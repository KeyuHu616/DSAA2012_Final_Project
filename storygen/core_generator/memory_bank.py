"""
Memory Bank - Long-term consistency maintenance for story generation
Implements compressed visual memory for maintaining coherence across frames
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Single memory entry from a generated frame"""
    features: torch.Tensor
    frame_id: int
    importance_score: float = 1.0


class MemoryBank:
    """
    Compressed Visual Memory Bank

    Maintains a rolling history of visual features from generated frames
    to provide context for subsequent generations.

    Based on principles from STAGE and OneStory:
    - Feature compression to reduce memory footprint
    - Importance-weighted retrieval
    - Decay mechanism for older memories
    """

    def __init__(
        self,
        capacity: int = 5,
        decay_factor: float = 0.9,
        compression_dim: int = 128,
        device: str = "cuda"
    ):
        """
        Initialize memory bank

        Args:
            capacity: Maximum number of frames to remember
            decay_factor: How much older memories decay (0.0-1.0)
            compression_dim: Dimension for compressed feature representation
            device: Computation device
        """
        self.capacity = capacity
        self.decay_factor = decay_factor
        self.compression_dim = compression_dim
        self.device = device

        # Memory storage
        self.entries: List[MemoryEntry] = []

        # Projection layer for feature compression
        self.compressor = None
        self._init_compressor()

    def _init_compressor(self):
        """Initialize feature compression layer"""
        # Will be initialized on first use when feature dim is known
        pass

    def _ensure_compressor(self, feature_dim: int):
        """Ensure compressor is properly initialized"""
        if self.compressor is None or self.compressor.in_features != feature_dim:
            self.compressor = torch.nn.Linear(feature_dim, self.compression_dim).to(self.device)
            # Move to same dtype as device default (will be overwritten on first use)
            self.compressor = self.compressor.to(dtype=torch.float16)
            self.compressor = self.compressor.to(self.device)  # Ensure correct device

    def update(self, features: torch.Tensor, frame_id: Optional[int] = None):
        """
        Add new frame features to memory bank

        Args:
            features: Feature tensor [B, Seq, Dim] or [B, Dim]
            frame_id: Optional frame identifier
        """
        # Flatten sequence dimension if present
        if features.dim() == 3:
            # Take mean over sequence
            features = features.mean(dim=1)

        # Ensure proper device
        features = features.to(self.device)

        # Compress features
        self._ensure_compressor(features.shape[-1])
        compressed = self.compressor(features)

        # Compute importance score (higher for recent frames)
        importance = 1.0 if frame_id is None else 1.0 / (1.0 + frame_id * 0.1)

        # Apply decay to existing memories
        self._apply_decay()

        # Create new entry
        entry = MemoryEntry(
            features=compressed.detach(),
            frame_id=frame_id if frame_id is not None else len(self.entries),
            importance_score=importance
        )

        # Add to storage (remove oldest if at capacity)
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)

        self.entries.append(entry)

    def _apply_decay(self):
        """Apply decay factor to all memory entries"""
        for entry in self.entries:
            entry.importance_score *= self.decay_factor

    def retrieve(self) -> Optional[torch.Tensor]:
        """
        Retrieve aggregated memory context

        Returns:
            Aggregated memory tensor or None if empty
        """
        if not self.entries:
            return None

        # Weighted average of all entries
        total_weight = sum(e.importance_score for e in self.entries)
        if total_weight == 0:
            return None

        weighted_sum = sum(
            e.features * e.importance_score
            for e in self.entries
        ) / total_weight

        return weighted_sum

    def retrieve_top_k(self, k: int = 2) -> List[torch.Tensor]:
        """
        Retrieve top-k most important memory entries

        Args:
            k: Number of entries to retrieve

        Returns:
            List of feature tensors, sorted by importance
        """
        if not self.entries:
            return []

        # Sort by importance
        sorted_entries = sorted(self.entries, key=lambda e: e.importance_score, reverse=True)

        # Return top-k features
        return [e.features for e in sorted_entries[:k]]

    def get_temporal_context(self, window_size: int = 3) -> Optional[torch.Tensor]:
        """
        Get temporal context from recent frames only

        Args:
            window_size: Number of recent frames to consider

        Returns:
            Temporal context tensor
        """
        if not self.entries:
            return None

        recent = self.entries[-window_size:]
        stacked = torch.stack([e.features for e in recent])

        return stacked.mean(dim=0)

    def clear(self):
        """Clear all memory entries"""
        self.entries = []

    def __len__(self) -> int:
        """Return number of memory entries"""
        return len(self.entries)

    def get_info(self) -> dict:
        """Get memory bank status information"""
        return {
            "capacity": self.capacity,
            "current_size": len(self.entries),
            "decay_factor": self.decay_factor,
            "compression_dim": self.compression_dim,
            "total_importance": sum(e.importance_score for e in self.entries)
        }
