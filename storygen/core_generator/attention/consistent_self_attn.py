"""
Consistent Self-Attention Mechanism
StoryDiffusion-style implementation for training-free identity consistency

This module implements attention-based mechanisms to maintain character
consistency across multiple generated images without requiring model fine-tuning.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusers.models.attention_processor import Attention, AttnProcessor


class ConsistentSelfAttentionProcessor(AttnProcessor):
    """
    StoryDiffusion-style consistent self-attention processor

    Core principle: Modify the QKV computation in self-attention to make
    images in the same batch attend to each other, sharing identity features.

    This enables training-free consistency where subsequent frames can
    "see" and reference features from previous generations.
    """

    def __init__(
        self,
        consistency_strength: float = 0.6,
        memory_bank_size: int = 4,
        device: str = "cuda"
    ):
        """
        Initialize the consistent attention processor

        Args:
            consistency_strength: How strongly frames attend to history (0.0-1.0)
                                Higher = more consistent but potentially less diverse
            memory_bank_size: Number of historical frames to remember
            device: Computation device
        """
        self.consistency_strength = consistency_strength
        self.memory_bank_size = memory_bank_size
        self.device = device

        # Historical feature storage (store downsampled pixel-space features instead of VAE latents)
        self.memory_bank = []
        
        # Projection: [B, H*W, 4] (VAE latent) -> [B, H*W, target_dim]
        self.feature_projector = None
        self._projector_target_dim = None

    def clear_memory(self):
        """Clear memory bank when starting a new story"""
        self.memory_bank = []

    def update_memory(self, new_features: torch.Tensor):
        """
        Add new frame features to memory bank

        Args:
            new_features: Feature tensor from current frame [B, Seq, Dim]
        """
        # Remove oldest if capacity exceeded
        if len(self.memory_bank) >= self.memory_bank_size:
            self.memory_bank.pop(0)

        # Add new features (detach to avoid gradient accumulation)
        # Ensure consistent dtype
        new_features = new_features.detach().to(dtype=torch.float16)
        self.memory_bank.append(new_features)

    def get_context_features(self) -> Optional[torch.Tensor]:
        """
        Get aggregated historical context features

        Returns:
            Aggregated context tensor or None if memory is empty
        """
        if not self.memory_bank:
            return None

        # Simple average aggregation (can be upgraded to attention-weighted)
        stacked = torch.stack(self.memory_bank, dim=0)  # [Mem, B, Seq, Dim]
        context = stacked.mean(dim=0)  # [B, Seq, Dim]

        return context

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute modified self-attention with consistency.
        Inherits from AttnProcessor to be compatible with set_attn_processor().
        Only applies consistency injection for self-attention (no encoder_hidden_states).
        Cross-attention layers are passed through unchanged.
        """
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        # For cross-attention, use encoder_hidden_states; for self-attention use hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states_for_kv = hidden_states
        else:
            encoder_hidden_states_for_kv = encoder_hidden_states
            if attn.norm_cross:
                encoder_hidden_states_for_kv = attn.norm_encoder_hidden_states(encoder_hidden_states_for_kv)

        key = attn.to_k(encoder_hidden_states_for_kv, *args)
        value = attn.to_v(encoder_hidden_states_for_kv, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # NOTE: Consistency injection temporarily disabled due to complex dimension handling
        # The memory bank stores VAE latent features (4-dim) which don't match UNet attention dims
        # This feature requires proper CLIP image encoder integration (IP-Adapter approach)
        # For now, using standard attention with no memory injection

        # Compute attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class RegionDisentangledProcessor(ConsistentSelfAttentionProcessor):
    """
    ReDiStory-style region disentangled processor

    Core improvement over base implementation:
    - Explicitly decomposes features into identity-related and scene-specific components
    - Applies different strategies at different layers
    - More fine-grained control over consistency vs diversity balance
    """

    def __init__(
        self,
        identity_weight: float = 0.7,
        scene_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.identity_weight = identity_weight
        self.scene_weight = scene_weight

        # Identity projection layers (learn to map features to identity subspace)
        hidden_dim = kwargs.get('dim', 320)
        self.identity_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.scene_projection = torch.nn.Linear(hidden_dim, hidden_dim)

    def disentangle_features(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feature disentanglement: Separate identity and scene components

        Args:
            features: Input feature tensor

        Returns:
            Tuple of (identity_features, scene_features)
        """
        identity = self.identity_projection(features)
        scene = self.scene_projection(features)

        # Orthogonality constraint can be added here for better separation
        # For simplicity, using direct decomposition

        return identity, scene

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        """
        Forward pass with feature disentanglement
        """
        # First execute standard consistent attention
        output = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        # Apply disentanglement recombination
        identity_feat, scene_feat = self.disentangle_features(output)

        # Weighted combination: stronger identity for consistency
        recombined = identity_feat * self.identity_weight + scene_feat * self.scene_weight

        return recombined


class ICSA_RACA_Processor:
    """
    TaleDiffusion-style Identity-Consistent Self-Attention and
    Region-Aware Cross-Attention processor

    Designed for multi-character scenes with spatial layout control:
    - ICSA: Maintains identity consistency for each character
    - RACA: Controls spatial relationships between characters
    """

    def __init__(
        self,
        num_characters: int = 2,
        spatial_weights: dict = None,
        **kwargs
    ):
        """
        Initialize multi-character attention processor

        Args:
            num_characters: Maximum number of characters to track
            spatial_weights: Dict mapping character pairs to attention weights
        """
        self.num_characters = num_characters
        self.spatial_weights = spatial_weights or {}
        super().__init__(**kwargs)

        # Character-specific feature banks
        self.character_features = {}

    def register_character(self, character_id: str, features: torch.Tensor):
        """Register a character's feature vector"""
        self.character_features[character_id] = features.detach().clone()

    def get_character_attention(
        self,
        query: torch.Tensor,
        character_id: str
    ) -> torch.Tensor:
        """
        Compute attention scores for a specific character

        Args:
            query: Query tensor
            character_id: Character identifier

        Returns:
            Attention scores for this character
        """
        if character_id not in self.character_features:
            return query

        char_features = self.character_features[character_id]

        # Compute similarity between query and character features
        # Simplified: using dot product attention
        similarity = torch.matmul(query, char_features.transpose(-2, -1))

        return similarity

    def apply_spatial_constraints(
        self,
        attention_scores: torch.Tensor,
        spatial_layout: dict
    ) -> torch.Tensor:
        """
        Apply spatial constraints to attention scores

        Args:
            attention_scores: Base attention scores
            spatial_layout: Dict describing relative positions

        Returns:
            Modified attention scores with spatial constraints
        """
        # Apply spatial weights based on layout
        for char_pair, weight in self.spatial_weights.items():
            # Parse character pair (e.g., "char1_char2")
            chars = char_pair.split("_")
            if len(chars) == 2 and all(c in self.character_features for c in chars):
                # Adjust attention based on spatial relationship
                attention_scores = attention_scores * weight

        return attention_scores
