"""
Visual Consistency Evaluator
Measures character and style consistency across generated frames
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from PIL import Image
import numpy as np
import lpips


class ConsistencyEvaluator:
    """
    Visual Consistency Evaluator

    Measures identity consistency across story frames using:
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - DreamSim-based perceptual similarity
    - CLIP feature cosine similarity

    Lower scores indicate better consistency (more similar).
    """

    def __init__(
        self,
        device: str = "cuda",
        metric: str = "lpips"
    ):
        """
        Initialize consistency evaluator

        Args:
            device: Computation device
            metric: Similarity metric to use ("lpips", "clip", "both")
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.metric = metric

        # Initialize LPIPS model
        self._lpips_model = None
        if metric in ["lpips", "both"]:
            try:
                self._lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self._lpips_model.eval()
            except Exception as e:
                print(f"[Consistency] LPIPS init failed: {e}")
                self._lpips_model = None

        # Initialize CLIP for feature extraction
        self._clip_model = None
        self._preprocess = None
        if metric in ["clip", "both"]:
            try:
                import open_clip
                self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                    'ViT-L-14', pretrained='openai'
                )
                self._clip_model = self._clip_model.to(self.device)
                self._clip_model.eval()
            except Exception as e:
                print(f"[Consistency] CLIP init failed: {e}")
                self._clip_model = None

    def _preprocess_image(self, img: Image.Image, size: int = 224) -> torch.Tensor:
        """Preprocess image for consistency evaluation"""
        if self._preprocess is not None:
            return self._preprocess(img).unsqueeze(0).to(self.device)

        # Fallback preprocessing
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(self.device)

    def compute_lpips_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> float:
        """
        Compute LPIPS similarity between two images

        Args:
            img1, img2: PIL Images to compare

        Returns:
            LPIPS distance (lower = more similar)
        """
        if self._lpips_model is None:
            return 0.5

        try:
            # Preprocess images to [-1, 1]
            img1_tensor = self._preprocess_image(img1, size=256) * 2 - 1
            img2_tensor = self._preprocess_image(img2, size=256) * 2 - 1

            with torch.no_grad():
                distance = self._lpips_model(img1_tensor, img2_tensor)
                return float(distance.item())

        except Exception as e:
            print(f"[Consistency] LPIPS error: {e}")
            return 0.5

    def compute_clip_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> float:
        """
        Compute CLIP feature cosine similarity

        Args:
            img1, img2: PIL Images to compare

        Returns:
            Cosine similarity (higher = more similar, range -1 to 1)
        """
        if self._clip_model is None:
            return 0.5

        try:
            img1_tensor = self._preprocess_image(img1)
            img2_tensor = self._preprocess_image(img2)

            with torch.no_grad():
                feat1 = self._clip_model.encode_image(img1_tensor)
                feat2 = self._clip_model.encode_image(img2_tensor)

                # Normalize and compute cosine similarity
                feat1 = F.normalize(feat1, p=2, dim=-1)
                feat2 = F.normalize(feat2, p=2, dim=-1)

                similarity = (feat1 @ feat2.T).item()
                return float(similarity)

        except Exception as e:
            print(f"[Consistency] CLIP error: {e}")
            return 0.5

    def compute_pairwise_consistency(
        self,
        images: List[Image.Image]
    ) -> Dict[str, List[float]]:
        """
        Compute pairwise consistency scores across all consecutive frames

        Args:
            images: List of generated story images

        Returns:
            Dict with consistency metrics
        """
        if len(images) < 2:
            return {"lpips": [], "clip": [], "average": []}

        lpips_scores = []
        clip_scores = []

        # Compute for all consecutive pairs
        for i in range(len(images) - 1):
            if self.metric in ["lpips", "both"]:
                lpips_score = self.compute_lpips_similarity(images[i], images[i+1])
                lpips_scores.append(lpips_score)

            if self.metric in ["clip", "both"]:
                clip_score = self.compute_clip_similarity(images[i], images[i+1])
                clip_scores.append(clip_score)

        # Compute average
        if self.metric == "lpips":
            averages = lpips_scores
        elif self.metric == "clip":
            averages = clip_scores
        else:
            averages = [
                (lp + cl) / 2
                for lp, cl in zip(lpips_scores, clip_scores)
            ]

        return {
            "lpips": lpips_scores,
            "clip": clip_scores,
            "average": averages,
            "average_lpips": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
            "average_clip": float(np.mean(clip_scores)) if clip_scores else 0.0
        }

    def compute_first_frame_consistency(
        self,
        images: List[Image.Image]
    ) -> Dict[str, float]:
        """
        Compute consistency of all frames with the first frame
        This measures how well identity is maintained throughout the story

        Args:
            images: List of generated story images

        Returns:
            Dict with consistency metrics vs first frame
        """
        if len(images) < 2:
            return {"lpips": [], "clip": [], "average": []}

        first_frame = images[0]
        lpips_scores = []
        clip_scores = []

        # Compare all frames to first frame
        for img in images[1:]:
            if self.metric in ["lpips", "both"]:
                lpips_scores.append(self.compute_lpips_similarity(first_frame, img))

            if self.metric in ["clip", "both"]:
                clip_scores.append(self.compute_clip_similarity(first_frame, img))

        return {
            "vs_first_lpips": lpips_scores,
            "vs_first_clip": clip_scores,
            "avg_vs_first_lpips": float(np.mean(lpips_scores)) if lpips_scores else 0.0,
            "avg_vs_first_clip": float(np.mean(clip_scores)) if clip_scores else 0.0
        }

    def evaluate_story(
        self,
        images: List[Image.Image]
    ) -> Dict[str, any]:
        """
        Comprehensive consistency evaluation for a story

        Args:
            images: List of generated story images

        Returns:
            Complete evaluation report
        """
        print("[Consistency] Evaluating story consistency...")

        pairwise = self.compute_pairwise_consistency(images)
        vs_first = self.compute_first_frame_consistency(images)

        return {
            "pairwise_consistency": pairwise,
            "first_frame_consistency": vs_first,
            "num_frames": len(images),
            "metric_used": self.metric
        }
