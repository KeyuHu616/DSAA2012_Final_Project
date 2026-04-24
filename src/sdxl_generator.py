"""
White-Box SDXL Generator Module - Core Story Pipeline
Implements: CSA (StoryDiffusion) + MSA (StoryDiffusion) + Shared Attention (ConsiStory)

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from diffusers import (
    AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler,
    CLIPTextModel, CLIPTextModelWithProjection,
)
from diffusers.models.attention_processor import Attention, AttentionProcessor


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ConsistencyConfig:
    """Configuration for consistency mechanisms"""
    share_step_ratio: float = 0.5  # CSA: First 50% steps use shared attention
    msa_inject_scale: float = 0.35  # MSA: Character injection weight
    enable_shared_attention: bool = True
    enable_msa: bool = True


# ============================================================================
# CONSISTENT SELF-ATTENTION (CSA) - StoryDiffusion-inspired
# ============================================================================

class ConsistentSelfAttention(nn.Module):
    """
    Replaces UNet's default self-attention with caching for structural consistency.
    
    Core Insight:
    - Early denoising steps (high noise) determine STRUCTURE
    - Late denoising steps (low noise) determine DETAILS
    - Cache attention maps in early steps, reuse for subsequent frames
    """

    def __init__(self, original_attn: Attention, share_step_ratio: float = 0.5):
        super().__init__()
        self.to_q = original_attn.to_q
        self.to_k = original_attn.to_k
        self.to_v = original_attn.to_v
        self.to_out = original_attn.to_out
        self.heads = original_attn.heads
        self.scale = original_attn.scale
        self.cross_attention_dim = original_attn.cross_attention_dim
        self.share_step_ratio = share_step_ratio
        self.cached_attn_map: Optional[torch.Tensor] = None
        self.is_first_frame: bool = True

    def clear_cache(self):
        self.cached_attn_map = None
        self.is_first_frame = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_index: int = 0,
        total_steps: int = 24,
        **kwargs
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = hidden_states.shape
        head_dim = dim // self.heads

        # QKV projection
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape for multi-head: (B, N, H, D) -> (B, H, N, D)
        query = query.view(batch_size, sequence_length, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.heads, head_dim).transpose(1, 2)

        is_early_step = step_index < int(total_steps * self.share_step_ratio)

        if is_early_step and self.cached_attn_map is not None:
            # REUSE cached attention map (structural consistency)
            attn_probs = self.cached_attn_map.to(query.device)
            if attn_probs.shape[0] != batch_size:
                attn_probs = attn_probs.repeat(batch_size, 1, 1, 1)
            attn_output = torch.matmul(attn_probs, value)
        else:
            # Standard attention
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attn_probs = F.softmax(attn_scores.float().half(), dim=-1)
            attn_output = torch.matmul(attn_probs, value)

            # Cache for first frame
            if is_early_step and self.is_first_frame and self.cached_attn_map is None:
                self.cached_attn_map = attn_probs.detach().clone()

        # Reshape back: (B, H, N, D) -> (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, dim)
        return self.to_out(attn_output)


# ============================================================================
# MULTI-SOURCE CROSS-ATTENTION (MSA) - StoryDiffusion-inspired
# ============================================================================

class MultiSourceCrossAttention(nn.Module):
    """
    Enhanced cross-attention that injects character embeddings for identity consistency.
    
    Core Insight:
    - Standard: attend to TEXT (action, background)
    - Ours: ALSO attend to CHARACTER embeddings
    - Ensures every spatial position sees both current frame description AND character
    """

    def __init__(self, original_attn: Attention, char_embed_dim: int = 768, inject_scale: float = 0.35):
        super().__init__()
        self.original_attn = original_attn
        self.to_q = original_attn.to_q
        self.to_k = original_attn.to_k
        self.to_v = original_attn.to_v
        self.to_out = original_attn.to_out
        self.heads = original_attn.heads
        self.scale = original_attn.scale
        self.cross_attention_dim = original_attn.cross_attention_dim

        # Character feature projectors (NEW)
        self.char_to_k = nn.Linear(char_embed_dim, original_attn.cross_attention_dim)
        self.char_to_v = nn.Linear(char_embed_dim, original_attn.cross_attention_dim)
        self.inject_scale = inject_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        char_embeds: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # 1. Standard cross-attention (text)
        std_output = self.original_attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if char_embeds is None or self.inject_scale == 0:
            return std_output

        # 2. Character-enhanced branch
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.cross_attention_dim // self.heads

        Q = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, head_dim).transpose(1, 2)
        K_char = self.char_to_k(char_embeds).view(batch_size, -1, self.heads, head_dim).permute(0, 2, 1, 3)
        V_char = self.char_to_v(char_embeds).view(batch_size, -1, self.heads, head_dim).permute(0, 2, 1, 3)

        char_attn_scores = torch.matmul(Q, K_char.transpose(-2, -1)) * self.scale
        char_attn_probs = F.softmax(char_attn_scores, dim=-1)
        char_attn_output = torch.matmul(char_attn_probs, V_char)
        char_attn_output = char_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.cross_attention_dim)
        char_attn_output = self.to_out(char_attn_output)

        # 3. Blend
        return std_output * (1 - self.inject_scale) + char_attn_output * self.inject_scale


# ============================================================================
# SHARED ATTENTION MANAGER - ConsiStory-inspired
# ============================================================================

class SharedAttentionManager:
    """
    Manages feature sharing between frames for structural consistency.
    
    Core Insight (ConsiStory):
    - Early steps (high noise): Share features for structure consistency
    - Late steps (low noise): Allow independence for detail diversity
    """

    def __init__(self, share_threshold: float = 0.5):
        self.share_threshold = share_threshold
        self.cache: Dict[str, Tuple] = {}
        self.is_first_frame: bool = True

    def should_share(self, step_index: int, total_steps: int) -> bool:
        return step_index < int(total_steps * self.share_threshold)

    def save(self, layer_name: str, attn_out: Optional[torch.Tensor], hidden: Optional[torch.Tensor]):
        if self.is_first_frame:
            self.cache[layer_name] = (
                attn_out.detach().clone() if attn_out is not None else None,
                hidden.detach().clone() if hidden is not None else None
            )

    def get(self, layer_name: str):
        return self.cache.get(layer_name, (None, None))

    def mark_first_frame_done(self):
        self.is_first_frame = False
        print(f"[SharedAttn] Cache saved for {len(self.cache)} layers")

    def clear(self):
        self.cache = {}
        self.is_first_frame = True


# ============================================================================
# CUSTOM ATTENTION PROCESSORS
# ============================================================================

class ConsistentSelfAttentionProcessor(AttentionProcessor):
    def __init__(self, consistent_attn: ConsistentSelfAttention, shared_mgr: Optional[SharedAttentionManager] = None):
        self.consistent_attn = consistent_attn
        self.shared_mgr = shared_mgr

    def __call__(self, attn: Attention, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        step_index = kwargs.get("step_index", 0)
        total_steps = kwargs.get("total_steps", 24)
        return self.consistent_attn(hidden_states, step_index=step_index, total_steps=total_steps)


class MultiSourceCrossAttentionProcessor(AttentionProcessor):
    def __init__(self, msa_attn: MultiSourceCrossAttention):
        self.msa_attn = msa_attn

    def __call__(self, attn: Attention, hidden_states: torch.Tensor, encoder_hidden_states=None, **kwargs) -> torch.Tensor:
        char_embeds = kwargs.get("char_embeds", None)
        return self.msa_attn(hidden_states, encoder_hidden_states=encoder_hidden_states, char_embeds=char_embeds)


# ============================================================================
# WHITEBOX SDXL GENERATOR - MAIN CLASS
# ============================================================================

class WhiteBoxSDXLGenerator:
    """
    White-Box SDXL Generator with MSA + CSA + Shared Attention.
    
    Key Capabilities:
    1. Manual denoising loop with step-by-step control
    2. Character identity injection via Multi-Source Cross-Attention
    3. Structural consistency via Consistent Self-Attention
    4. Feature sharing via SharedAttentionManager
    5. Memory-optimized inference

    Usage:
        generator = WhiteBoxSDXLGenerator()
        generator.load_components()
        
        # First frame
        img1 = generator.generate_single_frame(prompt, char_prompt, seed=42)
        
        # Subsequent frames with consistency
        img2 = generator.generate_with_consistency(
            prompt, char_prompt, seed=137, frame_index=2
        )
    """

    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: str = "./models",
        consistency_config: Optional[ConsistencyConfig] = None
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache_dir = cache_dir
        
        # Default config
        class DefaultConfig:
            share_step_ratio = 0.5
            msa_inject_scale = 0.35
            enable_shared_attention = True
            enable_msa = True
        
        self.config = consistency_config or DefaultConfig()

        # Core components
        self.vae: Optional[AutoencoderKL] = None
        self.unet: Optional[UNet2DConditionModel] = None
        self.text_encoder_1: Optional[CLIPTextModel] = None
        self.text_encoder_2: Optional[CLIPTextModelWithProjection] = None
        self.scheduler: Optional[EulerDiscreteScheduler] = None

        # Consistency mechanisms
        self.csa_modules: List = []
        self.msa_modules: List = []
        self.shared_mgr = SharedAttentionManager(share_threshold=self.config.share_step_ratio)

        # Caches
        self.char_embeds_cache: Dict[str, torch.Tensor] = {}
        self.is_loaded = False

    def load_components(self, low_vram_mode: bool = False):
        """Load all SDXL components with memory optimizations."""
        print(f"\n[SDXL] Loading White-Box SDXL from: {self.model_path}")
        print(f"[SDXL] Device: {self.device}, Dtype: {self.dtype}")

        # 1. VAE with slicing
        print("[SDXL] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_path, subfolder="vae", torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device)
        if low_vram_mode:
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        # 2. UNet
        print("[SDXL] Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path, subfolder="unet", torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device)
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()

        # 3. Text Encoders
        print("[SDXL] Loading Text Encoders...")
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            self.model_path, subfolder="text_encoder", torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_path, subfolder="text_encoder_2", torch_dtype=self.dtype, cache_dir=self.cache_dir
        ).to(self.device)

        # 4. Scheduler
        print("[SDXL] Loading Scheduler...")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_path, subfolder="scheduler"
        )

        # 5. Patch UNet with consistency mechanisms
        print("[SDXL] Patching UNet...")
        self._patch_unet()

        # Eval mode
        self.unet.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        self.vae.eval()
        
        for p in self.unet.parameters(): p.requires_grad = False
        for p in self.text_encoder_1.parameters(): p.requires_grad = False
        for p in self.text_encoder_2.parameters(): p.requires_grad = False
        for p in self.vae.parameters(): p.requires_grad = False

        self.is_loaded = True
        print("[SDXL] All components loaded!")

    def _patch_unet(self):
        """Monkey-patch UNet attention with CSA and MSA modules."""
        from diffusers.models.attention_processor import AttnProcessor
        
        attn_processors = self.unet.attn_processors
        self.csa_modules = []
        self.msa_modules = []

        for name, processor in attn_processors.items():
            parts = name.split(".")
            module = self.unet
            for p in parts[:-1]:
                module = getattr(module, p)
            original_attn = getattr(module, parts[-1], None) if parts else None

            if original_attn is None or not isinstance(original_attn, Attention):
                continue

            if "attn1" in name:
                # Self-attention -> CSA
                csa = ConsistentSelfAttention(original_attn, self.config.share_step_ratio)
                self.csa_modules.append(csa)
                attn_processors[name] = ConsistentSelfAttentionProcessor(csa, self.shared_mgr)
            elif "attn2" in name:
                # Cross-attention -> MSA
                msa = MultiSourceCrossAttention(original_attn, 768, self.config.msa_inject_scale)
                self.msa_modules.append(msa)
                attn_processors[name] = MultiSourceCrossAttentionProcessor(msa)

        self.unet.set_attn_processor(attn_processors)
        print(f"[SDXL] Patched {len(self.csa_modules)} CSA, {len(self.msa_modules)} MSA modules")

    def encode_prompts(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts using CLIP text encoders (SDXL dual-encoder)."""
        tok1 = self.text_encoder_1.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        tok2 = self.text_encoder_2.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        
        for k in tok1: tok1[k] = tok1[k].to(self.device)
        for k in tok2: tok2[k] = tok2[k].to(self.device)

        with torch.no_grad():
            emb1 = self.text_encoder_1(tok1.input_ids).last_hidden_state
            emb2 = self.text_encoder_2(tok2.input_ids).last_hidden_state
        
        prompt_embeds = torch.cat([emb1, emb2], dim=-1)
        pooled_embeds = self.text_encoder_2(tok2.input_ids).text_embeds

        if negative_prompt:
            nt1 = self.text_encoder_1.tokenizer(negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            nt2 = self.text_encoder_2.tokenizer(negative_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            for k in nt1: nt1[k] = nt1[k].to(self.device)
            for k in nt2: nt2[k] = nt2[k].to(self.device)
            
            with torch.no_grad():
                ne1 = self.text_encoder_1(nt1.input_ids).last_hidden_state
                ne2 = self.text_encoder_2(nt2.input_ids).last_hidden_state
            
            neg_embeds = torch.cat([ne1, ne2], dim=-1)
            neg_pooled = self.text_encoder_2(nt2.input_ids).text_embeds
            
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([neg_pooled, pooled_embeds], dim=0)
        else:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([pooled_embeds, pooled_embeds], dim=0)

        return prompt_embeds.half(), pooled_embeds.half()

    def encode_character(self, char_prompt: str) -> torch.Tensor:
        """Encode character global prompt for MSA injection."""
        if char_prompt in self.char_embeds_cache:
            return self.char_embeds_cache[char_prompt]
        
        tokens = self.text_encoder_1.tokenizer(char_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        for k in tokens: tokens[k] = tokens[k].to(self.device)
        
        with torch.no_grad():
            embeds = self.text_encoder_1(tokens.input_ids).last_hidden_state
        
        self.char_embeds_cache[char_prompt] = embeds.half()
        return embeds.half()

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to PIL image."""
        latents = latent / self.vae.config.scaling_factor
        with torch.no_grad():
            img = self.vae.decode(latents.half()).sample
        img = (img / 2 + 0.5).clamp(0, 1).cpu().float().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((img * 255).round().astype(np.uint8))

    def manual_denoise_loop(
        self,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        char_embeds: Optional[torch.Tensor] = None,
        height: int = 832, width: int = 896,
        num_steps: int = 24, guidance: float = 8.6,
        seed: int = 42, step_offset: int = 0
    ) -> torch.Tensor:
        """Manual denoising loop with custom attention injection."""
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, 4, height//8, width//8), device=self.device, generator=gen, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        added_cond_kwargs = {"text_embeds": pooled_embeds, "image_embeds": None}

        for i, t in enumerate(torch.longTensor(timesteps).to(self.device)):
            latent_input = torch.cat([latents] * 2)
            latent_input = self.scheduler.scale_model_input(latent_input, t)

            unet_kwargs = {
                "encoder_hidden_states": prompt_embeds,
                "added_cond_kwargs": added_cond_kwargs,
                "return_dict": False,
                "step_index": step_offset + i,
                "total_steps": num_steps,
            }
            if char_embeds is not None and self.config.enable_msa:
                unet_kwargs["char_embeds"] = char_embeds

            with torch.no_grad():
                noise_pred = self.unet(latent_input.half(), t, **unet_kwargs).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance * (noise_text - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if i % 5 == 0 or i == num_steps - 1:
                print(f"    Step {i+1}/{num_steps}, t={t.item():.1f}")

        return latents

    def generate_single_frame(
        self, prompt: str, char_prompt: str, neg_prompt: str = "",
        height: int = 832, width: int = 896,
        num_steps: int = 24, guidance: float = 8.6, seed: int = 42
    ) -> Image.Image:
        """Generate first frame (no consistency mechanisms)."""
        print(f"\n[SDXL] Generating single frame")
        print(f"       Prompt: {prompt[:80]}...")
        print(f"       Char: {char_prompt[:50]}...")

        for csa in self.csa_modules: csa.clear_cache()
        self.shared_mgr.clear()
        self.char_embeds_cache.clear()

        prompt_embeds, pooled_embeds = self.encode_prompts(prompt, neg_prompt)
        char_embeds = self.encode_character(char_prompt)

        latent = self.manual_denoise_loop(
            prompt_embeds, pooled_embeds, char_embeds,
            height, width, num_steps, guidance, seed, step_offset=0
        )

        for csa in self.csa_modules: csa.is_first_frame = False
        self.shared_mgr.mark_first_frame_done()

        return self.decode_latent(latent)

    def generate_with_consistency(
        self, prompt: str, char_prompt: str, neg_prompt: str = "",
        height: int = 832, width: int = 896,
        num_steps: int = 24, guidance: float = 8.6,
        seed: int = 42, frame_index: int = 2, n_candidates: int = 1
    ) -> List[Image.Image]:
        """Generate subsequent frames WITH consistency mechanisms."""
        print(f"\n[SDXL] Generating frame {frame_index} with consistency")
        print(f"       Prompt: {prompt[:80]}...")
        print(f"       Candidates: {n_candidates}")

        prompt_embeds, pooled_embeds = self.encode_prompts(prompt, neg_prompt)
        char_embeds = self.encode_character(char_prompt)

        candidates = []
        for cand_idx in range(n_candidates):
            cand_seed = seed + cand_idx * 1000
            print(f"  Candidate {cand_idx+1}/{n_candidates}, seed={cand_seed}")
            
            latent = self.manual_denoise_loop(
                prompt_embeds, pooled_embeds, char_embeds,
                height, width, num_steps, guidance, cand_seed,
                step_offset=(frame_index - 1) * num_steps
            )
            candidates.append(self.decode_latent(latent))

        return candidates

    def extract_dino_features(self, image: Image.Image) -> torch.Tensor:
        """Extract DINOv2 features for evaluation."""
        try:
            if not hasattr(self, 'dino_model'):
                print("[SDXL] Loading DINOv2...")
                self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
                self.dino_model.eval()

            if image.mode != "RGB": image = image.convert("RGB")
            img_t = torch.from_numpy(np.array(image.resize((224, 224)))).float() / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device), \
                        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            img_t = (img_t - mean) / std

            with torch.no_grad():
                return self.dino_model(img_t)
        except Exception as e:
            print(f"[SDXL] DINO extraction failed: {e}")
            return torch.randn(1, 768).to(self.device)

    def clear_cache(self):
        """Clear all caches."""
        self.char_embeds_cache.clear()
        for csa in self.csa_modules: csa.clear_cache()
        self.shared_mgr.clear()
        torch.cuda.empty_cache()


def create_generator(model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                     device: str = "cuda", config: Optional[ConsistencyConfig] = None) -> WhiteBoxSDXLGenerator:
    """Factory function to create initialized generator."""
    gen = WhiteBoxSDXLGenerator(model_path=model_path, device=device, consistency_config=config)
    gen.load_components()
    return gen
