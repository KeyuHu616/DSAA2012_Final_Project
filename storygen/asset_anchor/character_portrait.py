"""
Character Portrait Generator - Visual anchor generation for identity consistency
Generates character reference portraits and extracts IP-Adapter features
"""

import torch
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torchvision.transforms as transforms


class CharacterPortraitGenerator:
    """
    Character Portrait Generator

    Generates standardized portrait images for each character in a story
    and extracts visual features for IP-Adapter-based consistency.

    The workflow:
    1. Generate a high-quality portrait for each character
    2. Extract CLIP visual features from the portraits
    3. Use these features in subsequent frames to maintain identity
    """

    def __init__(
        self,
        base_model: Any = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the portrait generator

        Args:
            base_model: Optional pre-loaded SDXL pipeline
            device: Computation device
            dtype: Data type for computations
        """
        self.device = device
        self.dtype = dtype
        self.base_model = base_model
        self._pipe = None

        # Storage for generated portraits and features
        self.portraits: Dict[str, Image.Image] = {}
        self.features: Dict[str, torch.Tensor] = {}

    @property
    def pipe(self):
        """Lazy load diffusion pipeline"""
        if self._pipe is None:
            if self.base_model is not None:
                self._pipe = self.base_model
            else:
                from diffusers import StableDiffusionXLPipeline
                from storygen.utils.mirror_config import get_models_cache_dir
                cache_dir = get_models_cache_dir()
                print(f"[PortraitGen] Loading SDXL model for portrait generation...")
                print(f"[PortraitGen] Using cache directory: {cache_dir}")
                self._pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    cache_dir=str(cache_dir)  # Use project ./models directory
                ).to(self.device)
        return self._pipe

    def generate_portrait(
        self,
        character_info: Dict,
        seed: int = 42
    ) -> Image.Image:
        """
        Generate a portrait for a single character

        Args:
            character_info: Character information dict
            seed: Random seed for reproducible generation

        Returns:
            Generated portrait as PIL Image
        """
        char_name = character_info.get('name', 'character')
        visual_desc = character_info.get('visual_description', 'a person')
        clothing = character_info.get('clothing', 'casual clothing')

        # Compose portrait prompt
        portrait_prompt = (
            f"portrait of {visual_desc}, {clothing}, "
            f"cinematic portrait photography, neutral expression, "
            f"front facing, soft studio lighting, clean background, "
            f"sharp focus, professional headshot, 8k quality, masterpiece"
        )

        negative_prompt = (
            "cartoon, anime, painting, drawing, artificial, deformed, "
            "bad anatomy, extra limbs, blurry, low quality, group shot"
        )

        print(f"[PortraitGen] Generating portrait for: {char_name}")

        # Generate portrait
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            output = self.pipe(
                prompt=portrait_prompt,
                negative_prompt=negative_prompt,
                height=768,
                width=768,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            )
            portrait = output.images[0]
        except Exception as e:
            print(f"[PortraitGen] Error generating portrait: {e}")
            # Return a placeholder
            portrait = Image.new('RGB', (512, 512), color=(100, 100, 100))

        return portrait

    def extract_clip_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLIP image features for IP-Adapter

        Args:
            image: Input portrait image

        Returns:
            CLIP feature tensor
        """
        try:
            # Image preprocessing for CLIP
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

            img_tensor = transform(image).unsqueeze(0).to(self.device, self.dtype)

            # Try to use CLIP model
            try:
                from transformers import CLIPVisionModelWithProjection
                from storygen.utils.mirror_config import get_models_cache_dir

                if not hasattr(self, '_clip_model'):
                    cache_dir = get_models_cache_dir()
                    print(f"[PortraitGen] Loading CLIP model... (cache: {cache_dir})")
                    self._clip_model = CLIPVisionModelWithProjection.from_pretrained(
                        "h94/IP-Adapter",
                        subfolder="image_encoder",
                        torch_dtype=self.dtype,
                        cache_dir=str(cache_dir)  # Use project ./models directory
                    ).to(self.device)

                with torch.no_grad():
                    image_embeds = self._clip_model(img_tensor).image_embeds

            except Exception:
                # Fallback: use VAE features as proxy
                print("[PortraitGen] Using VAE features as CLIP fallback")
                with torch.no_grad():
                    latent = self.pipe.vae.encode(
                        img_tensor * 2 - 1
                    ).latent_dist.sample()
                    image_embeds = latent.flatten(start_dim=1).mean(dim=1, keepdim=True)

            return image_embeds

        except Exception as e:
            print(f"[PortraitGen] Feature extraction error: {e}")
            # Return dummy features
            return torch.zeros(1, 768).to(self.device, self.dtype)

    def generate_all_portraits(
        self,
        characters: Dict[str, Dict],
        global_style: str = "cinematic",
        output_dir: str = "outputs/portraits"
    ) -> Dict[str, Tuple[Image.Image, torch.Tensor]]:
        """
        Generate portraits for all characters in a story

        Args:
            characters: Dict mapping character names to character info
            global_style: Overall visual style
            output_dir: Directory to save portraits

        Returns:
            Dict mapping character names to (portrait, features) tuples
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        for char_name, char_info in characters.items():
            print(f"\n[PortraitGen] Processing character: {char_name}")

            # Add name to character info
            char_info_with_name = {**char_info, 'name': char_name}

            # Generate portrait
            portrait = self.generate_portrait(char_info_with_name)

            # Save portrait
            portrait_path = Path(output_dir) / f"portrait_{char_name.replace(' ', '_')}.png"
            portrait.save(portrait_path)
            print(f"[PortraitGen] Portrait saved: {portrait_path}")

            # Extract features
            features = self.extract_clip_features(portrait)

            # Cache results
            self.portraits[char_name] = portrait
            self.features[char_name] = features
            results[char_name] = (portrait, features)

        return results

    def get_portrait(self, character_name: str) -> Optional[Image.Image]:
        """Get portrait for a specific character"""
        return self.portraits.get(character_name)

    def get_features(self, character_name: str) -> Optional[torch.Tensor]:
        """Get features for a specific character"""
        return self.features.get(character_name)

    def clear(self):
        """Clear all stored portraits and features"""
        self.portraits.clear()
        self.features.clear()
