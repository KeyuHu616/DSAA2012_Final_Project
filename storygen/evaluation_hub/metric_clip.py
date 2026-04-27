"""
CLIP-based Text-Image Alignment Evaluator
Measures how well generated images match their text prompts
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from PIL import Image
import numpy as np


class CLIPEvaluator:
    """
    CLIP-based Prompt Alignment Evaluator

    Uses CLIP model to compute text-image similarity scores,
    measuring how well generated images match their intended prompts.

    Based on principles from TIFA (Text-to-Image Faithfulness) evaluation.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda"
    ):
        """
        Initialize CLIP evaluator

        Args:
            model_name: CLIP model variant
            device: Computation device
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load CLIP model"""
        if self._model is None:
            try:
                import open_clip
                self._model, _, _ = open_clip.create_model_and_transforms(
                    'ViT-L-14',
                    pretrained='openai'
                )
                self._model = self._model.to(self.device)
                self._model.eval()
            except ImportError:
                print("[CLIP] Warning: open_clip not available, using fallback")
                self._model = None
        return self._model

    @property
    def preprocess(self):
        """Get image preprocessing transform"""
        try:
            import open_clip
            _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
            return preprocess
        except:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])

    def compute_similarity(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> List[float]:
        """
        Compute text-image similarity scores

        Args:
            images: List of PIL Images
            prompts: List of text prompts

        Returns:
            List of similarity scores (0.0-1.0)
        """
        if self.model is None:
            print("[CLIP] Model not available, returning dummy scores")
            return [0.5] * len(images)

        scores = []

        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                try:
                    # Preprocess image
                    img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

                    # Encode image
                    image_features = self.model.encode_image(img_tensor)
                    image_features = F.normalize(image_features, p=2, dim=-1)

                    # Encode text
                    import open_clip
                    text_tokens = open_clip.tokenize([prompt]).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, p=2, dim=-1)

                    # Compute similarity
                    similarity = (image_features @ text_features.T).item()
                    scores.append(float(similarity))

                except Exception as e:
                    print(f"[CLIP] Error computing similarity: {e}")
                    scores.append(0.0)

        return scores

    def compute_average_score(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> float:
        """Compute average similarity score across all images"""
        scores = self.compute_similarity(images, prompts)
        return float(np.mean(scores)) if scores else 0.0

    def evaluate_story(
        self,
        images: List[Image.Image],
        panels: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate entire story for prompt alignment

        Args:
            images: List of generated images
            panels: List of panel info dicts with 'prompt' key

        Returns:
            Dict with evaluation metrics
        """
        prompts = [p.get('prompt', p.get('raw_scene', '')) for p in panels]

        scores = self.compute_similarity(images, prompts)

        return {
            "average_clip_score": float(np.mean(scores)),
            "min_clip_score": float(np.min(scores)),
            "max_clip_score": float(np.max(scores)),
            "per_frame_scores": scores
        }
