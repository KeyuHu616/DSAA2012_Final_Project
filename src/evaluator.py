"""
Best-of-N Evaluator Module - GenEval-inspired Quality Assurance

This module implements multi-metric evaluation for story image quality:
1. DINOv2 Cosine Similarity: Object-level visual consistency across frames
2. CLIP Score: Text-image alignment with current frame prompt
3. Aesthetic Score: General image quality prediction

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import clip


# ============================================================================
# STORY EVALUATOR CLASS
# ============================================================================

class StoryEvaluator:
    """
    GenEval-inspired Best-of-N Evaluator for story image quality.
    
    Evaluates candidates using:
    - DINOv2: Visual consistency with previous frame
    - CLIP: Text-image alignment with current prompt
    - Aesthetic: General image quality
    """
    
    def __init__(
        self,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        use_aesthetic: bool = False
    ):
        self.device = device
        self.weights = weights or {"dino": 0.40, "clip": 0.35, "aesthetic": 0.25}
        self.use_aesthetic = use_aesthetic
        
        self._load_dino()
        self._load_clip()
        
        if use_aesthetic:
            self._load_aesthetic_scorer()
        else:
            self.aesthetic_model = None
            
        print(f"[Evaluator] Initialized with weights: {self.weights}")

    def _load_dino(self):
        """Load DINOv2 model."""
        try:
            print("[Evaluator] Loading DINOv2...")
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.dino = self.dino.to(self.device)
            self.dino.eval()
            print("[Evaluator] DINOv2 loaded")
        except Exception as e:
            print(f"[Evaluator] DINOv2 failed: {e}")
            self.dino = None

    def _load_clip(self):
        """Load CLIP model."""
        try:
            print("[Evaluator] Loading CLIP...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            print("[Evaluator] CLIP loaded")
        except Exception as e:
            print(f"[Evaluator] CLIP failed: {e}")
            self.clip_model = None
            self.clip_preprocess = None

    def _load_aesthetic_scorer(self):
        """Load aesthetic scorer (simple MLP on CLIP features)."""
        try:
            import torch.nn as nn
            self.aesthetic_model = nn.Sequential(
                nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 1)
            ).to(self.device)
            self.aesthetic_model.eval()
            print("[Evaluator] Aesthetic scorer ready")
        except Exception as e:
            print(f"[Evaluator] Aesthetic failed: {e}")
            self.aesthetic_model = None

    @torch.no_grad()
    def _extract_dino_features(self, image: Image.Image) -> torch.Tensor:
        """Extract DINOv2 features."""
        if self.dino is None:
            return torch.randn(1, 768).to(self.device)
        
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            img = image.resize((224, 224), Image.LANCZOS)
            img_t = torch.from_numpy(np.array(img)).float() / 255.0
            img_t = img_t.permute(2, 0, 1)
            mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device), \
                        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            img_t = ((img_t / 255.0) - mean) / std
            img_t = img_t.unsqueeze(0).to(self.device)
            return self.dino(img_t)
        except:
            return torch.randn(1, 768).to(self.device)

    @torch.no_grad()
    def _compute_clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP text-image similarity."""
        if self.clip_model is None:
            return 0.5
        
        try:
            img = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = clip.tokenize([text]).to(self.device)
            img_feat = F.normalize(self.clip_model.encode_image(img).float(), dim=-1)
            text_feat = F.normalize(self.clip_model.encode_text(text_tokens).float(), dim=-1)
            return ((img_feat @ text_feat.T).item() + 1) / 2
        except:
            return 0.5

    def _compute_aesthetic(self, image: Image.Image) -> float:
        """Compute aesthetic score."""
        if self.aesthetic_model is None or self.clip_model is None:
            return 0.5
        
        try:
            img = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(img).float()
            score = torch.sigmoid(self.aesthetic_model(features)).item()
            return score
        except:
            return 0.5

    @torch.no_grad()
    def evaluate_candidates(
        self,
        candidates: List[Image.Image],
        prev_best_img: Optional[Image.Image] = None,
        current_prompt: str = "",
        verbose: bool = True
    ) -> Tuple[int, Dict]:
        """
        Evaluate N candidates and select best.
        
        Args:
            candidates: List of PIL Images
            prev_best_img: Previous frame for consistency check
            current_prompt: Text prompt for alignment check
            
        Returns:
            (best_idx, all_scores)
        """
        if len(candidates) == 1:
            return 0, {"dino": [1.0], "clip": [0.5], "aesthetic": [0.5]}
        
        scores = {"dino": [], "clip": [], "aesthetic": []}
        
        ref_features = self._extract_dino_features(prev_best_img) if prev_best_img else None
        
        for img in candidates:
            # DINO consistency
            if ref_features is not None:
                curr = self._extract_dino_features(img)
                sim = F.cosine_similarity(ref_features, curr, dim=1).item()
                dino = max(0, min(1, (sim + 1) / 2))
            else:
                dino = 1.0
            scores["dino"].append(dino)
            
            # CLIP alignment
            clip_s = self._compute_clip_score(img, current_prompt) if current_prompt else 0.5
            scores["clip"].append(clip_s)
            
            # Aesthetic
            aes = self._compute_aesthetic(img) if self.use_aesthetic else 0.5
            scores["aesthetic"].append(aes)
        
        # Weighted composite
        w = self.weights
        composite = [
            w["dino"] * scores["dino"][i] +
            w["clip"] * scores["clip"][i] +
            w["aesthetic"] * scores["aesthetic"][i]
            for i in range(len(candidates))
        ]
        
        best_idx = int(np.argmax(composite))
        
        if verbose:
            print(f"\n[Evaluator] Results ({len(candidates)} candidates):")
            for i, (d, c, a, comp) in enumerate(zip(scores["dino"], scores["clip"], scores["aesthetic"], composite)):
                marker = " <- BEST" if i == best_idx else ""
                print(f"  #{i+1}: DINO={d:.3f} CLIP={c:.3f} AES={a:.3f} TOTAL={comp:.3f}{marker}")
        
        return best_idx, scores


# ============================================================================
# SIMPLE EVALUATOR (CLIP only, lightweight)
# ============================================================================

class SimpleEvaluator:
    """Lightweight evaluator using only CLIP."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
    
    @torch.no_grad()
    def evaluate_candidates(
        self,
        candidates: List[Image.Image],
        prev_best_img: Optional[Image.Image] = None,
        current_prompt: str = "",
        verbose: bool = True
    ) -> Tuple[int, Dict]:
        scores = []
        for img in candidates:
            if current_prompt:
                img_t = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                text_t = clip.tokenize([current_prompt]).to(self.device)
                img_feat = F.normalize(self.clip_model.encode_image(img_t).float(), dim=-1)
                text_feat = F.normalize(self.clip_model.encode_text(text_t).float(), dim=-1)
                sim = ((img_feat @ text_feat.T).item() + 1) / 2
            else:
                sim = 0.5
            scores.append(sim)
        
        best_idx = int(np.argmax(scores))
        
        if verbose and len(candidates) > 1:
            print(f"\n[SimpleEval]:")
            for i, s in enumerate(scores):
                print(f"  #{i+1}: {s:.3f}{' <- BEST' if i == best_idx else ''}")
        
        return best_idx, {"clip": scores}


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    print("Evaluator Test")
    evaluator = StoryEvaluator(use_aesthetic=False)
    test_images = [
        Image.new('RGB', (512, 512), color=(200, 100, 50)),
        Image.new('RGB', (512, 512), color=(100, 150, 200)),
    ]
    best_idx, scores = evaluator.evaluate_candidates(test_images, current_prompt="sunset colors")
    print(f"Best: #{best_idx + 1}")
