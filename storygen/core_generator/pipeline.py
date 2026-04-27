"""
Narrative Generation Pipeline - Core Story Generation Engine
Integrates all SOTA techniques into a unified generation interface
"""

import torch
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image
import numpy as np
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup HF mirror for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from storygen.script_director.llm_parser import ProductionBoard, Panel
from storygen.asset_anchor.character_portrait import CharacterPortraitGenerator
from storygen.core_generator.attention.consistent_self_attn import ConsistentSelfAttentionProcessor
from storygen.core_generator.memory_bank import MemoryBank


class NarrativeGenerationPipeline:
    """
    Main Story Generation Pipeline

    This class orchestrates the entire story generation process, including:
    - LLM-directed production board parsing
    - Character portrait generation and feature extraction
    - Consistent image generation with memory
    - Multi-frame story creation

    Usage:
        pipeline = NarrativeGenerationPipeline(config)
        images = pipeline.generate_story(production_board)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generation pipeline

        Args:
            config: Configuration dictionary containing:
                - base_model: Path to SDXL model
                - consistency_mode: "storydiffusion" | "redistory" | "hybrid"
                - device: Computation device
                - generation_params: Generation settings
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if config.get("use_fp16", True) else torch.float32

        # Component initialization flags
        self._initialized = False
        self._base_pipe = None
        self._portrait_gen = None
        self._attn_processor = None
        self._memory_bank = None

        print("=" * 60)
        print("Narrative Weaver Pro - Generation Engine")
        print("=" * 60)

    @property
    def base_pipe(self):
        """Lazy load base diffusion pipeline with cache-first strategy"""
        if self._base_pipe is None:
            from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
            from storygen.utils.mirror_config import verify_model_integrity, get_models_cache_dir

            model_name = self.config.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
            cache_dir = get_models_cache_dir()
            print(f"[Pipeline] Loading SDXL Base Model: {model_name}")
            print(f"[Pipeline] Using cache directory: {cache_dir}")

            # Check cache integrity first
            is_complete = verify_model_integrity(model_name, cache_dir)

            if is_complete:
                print("[Pipeline] ✓ Using local cache (skip network verification)")
                load_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "variant": "fp16" if self.dtype == torch.float16 else None,
                    "local_files_only": True,  # Skip network verification
                    "low_cpu_mem_usage": True,  # Memory optimization
                    "cache_dir": str(cache_dir),  # Use project ./models directory
                }
            else:
                print("[Pipeline] ⚠ Cache incomplete/missing, downloading from mirror...")
                load_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "variant": "fp16" if self.dtype == torch.float16 else None,
                    "local_files_only": False,
                    "low_cpu_mem_usage": True,
                    "cache_dir": str(cache_dir),  # Use project ./models directory
                }

            self._base_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                **load_kwargs
            ).to(self.device)

            # Use DPM++ scheduler for faster convergence
            self._base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._base_pipe.scheduler.config
            )

            # Enable memory optimization
            if self.config.get("enable_model_cpu_offload", True):
                self._base_pipe.enable_model_cpu_offload()

        return self._base_pipe

    @property
    def portrait_gen(self):
        """Lazy load character portrait generator"""
        if self._portrait_gen is None:
            print("[Pipeline] Initializing Character Portrait Generator...")
            self._portrait_gen = CharacterPortraitGenerator(
                base_model=self.base_pipe,
                device=self.device,
                dtype=self.dtype
            )
        return self._portrait_gen

    @property
    def attn_processor(self):
        """Lazy load attention processor"""
        if self._attn_processor is None:
            consistency_strength = self.config.get("consistency_strength", 0.0)
            if consistency_strength > 0:
                print("[Pipeline] Setting up Consistent Self-Attention...")
                self._attn_processor = ConsistentSelfAttentionProcessor(
                    consistency_strength=consistency_strength,
                    memory_bank_size=self.config.get("memory_bank_size", 4),
                    device=self.device
                )
                self.base_pipe.unet.set_attn_processor(self._attn_processor)
            else:
                print("[Pipeline] Using default attention (consistency disabled)")
                self._attn_processor = None  # Will use default processor
        return self._attn_processor

    @property
    def memory_bank(self):
        """Lazy load memory bank"""
        if self._memory_bank is None:
            print("[Pipeline] Initializing Memory Bank...")
            self._memory_bank = MemoryBank(
                capacity=self.config.get("memory_bank_capacity", 5),
                decay_factor=self.config.get("memory_decay_factor", 0.9),
                device=self.device
            )
        return self._memory_bank

    def initialize(self):
        """Explicit initialization of all components"""
        if self._initialized:
            return

        # Trigger lazy loading of all components
        _ = self.base_pipe
        _ = self.portrait_gen
        _ = self.attn_processor
        _ = self.memory_bank

        self._initialized = True
        print("[Pipeline] All components initialized successfully!\n")

    def _compose_prompt(
        self,
        panel: Panel,
        global_style: str,
        characters: Dict
    ) -> str:
        """
        Compose final generation prompt optimized for SDXL.
        Prefer using enhanced_prompt from parser, supplemented with additional details.
        Limit to ~77 tokens to avoid CLIP truncation issues.
        """
        import re
        
        # If parser provided an enhanced prompt, use it as base
        if panel.enhanced_prompt and len(panel.enhanced_prompt) > 30:
            base_prompt = panel.enhanced_prompt
        else:
            # Fallback: build prompt from components
            base_prompt = self._build_prompt_from_components(panel, global_style, characters)
        
        # Truncate to avoid token limit (roughly 1 token ≈ 4 chars for CLIP)
        max_chars = 300  # ~75 tokens
        if len(base_prompt) > max_chars:
            base_prompt = base_prompt[:max_chars]
        
        # Add global style modifiers if not already present
        if global_style not in base_prompt.lower():
            base_prompt = f"{base_prompt}, {global_style}"
        
        # Final truncation after adding style
        if len(base_prompt) > max_chars + 30:
            base_prompt = base_prompt[:max_chars + 30]
        
        return base_prompt
    
    def _build_prompt_from_components(
        self,
        panel: Panel,
        global_style: str,
        characters: Dict
    ) -> str:
        """Build prompt from individual components (fallback)"""
        import re
        parts = []
        seen_parts = set()

        def add_unique(part: str):
            part_lower = part.lower().strip()
            if part_lower and part_lower not in seen_parts:
                seen_parts.add(part_lower)
                parts.append(part.strip())

        # Get characters in panel
        present_char_names = []
        for char_name, char_info in characters.items():
            if char_name in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                present_char_names.append((char_name, char_info))

        # Character descriptions
        if present_char_names:
            if len(present_char_names) == 2:
                add_unique("two young adults")
            elif len(present_char_names) == 1:
                add_unique("one person")
            
            for char_name, char_info in present_char_names:
                if hasattr(char_info, 'visual_description') and char_info.visual_description:
                    desc_parts = char_info.visual_description.split(",")
                    for desc in desc_parts[:3]:
                        add_unique(desc.strip())
                
                if hasattr(char_info, 'clothing') and char_info.clothing:
                    clothing = char_info.clothing.strip()
                    if len(clothing) > 5:
                        add_unique(clothing)

        # Setting
        if panel.setting:
            add_unique(panel.setting)
        else:
            raw_lower = panel.raw_prompt.lower()
            if "park" in raw_lower:
                add_unique("outdoor park with trees and green grass")
            elif "cafe" in raw_lower or "coffee" in raw_lower:
                add_unique("cozy cafe interior with wooden tables")
            elif "window" in raw_lower:
                add_unique("indoor room with window view")
            elif "exhibition" in raw_lower or "gallery" in raw_lower:
                add_unique("art gallery with paintings on walls")
            elif "bus" in raw_lower:
                add_unique("bus interior")
            elif "train" in raw_lower:
                add_unique("train interior")

        # Main action (cleaned)
        main_content = panel.raw_prompt
        main_content = re.sub(r'<[A-Z][a-z]+>\s*', '', main_content)  # Remove <Name> tags
        main_content = main_content.strip().rstrip('.')
        if main_content:
            add_unique(main_content)

        # Shot type
        shot_map = {
            "closeup": "close-up portrait",
            "medium": "medium shot",
            "wide": "wide angle shot",
            "extreme_closeup": "extreme close-up",
            "over_shoulder": "over-the-shoulder shot"
        }
        if panel.shot_type in shot_map:
            add_unique(shot_map[panel.shot_type])

        # Lighting
        if panel.lighting_mood and panel.lighting_mood != "natural":
            add_unique(f"{panel.lighting_mood} lighting")
        if panel.time_of_day:
            add_unique(panel.time_of_day)

        # Style and quality
        add_unique(global_style)
        add_unique("photorealistic, 8k, sharp focus, cinematic lighting")

        return ", ".join(filter(None, parts))

    def _extract_characters_from_panel(self, panel: Panel, all_characters: Dict) -> List[str]:
        """Extract character names appearing in this panel"""
        present_chars = []
        for char_name in all_characters.keys():
            if f"<{char_name}>" in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                present_chars.append(char_name)
        return present_chars

    @torch.inference_mode()
    def generate_story(
        self,
        production_board: ProductionBoard,
        seed: Optional[int] = None,
        return_portraits: bool = False
    ) -> Tuple[List[Image.Image], Optional[Dict]]:
        """
        Generate complete story as a sequence of images

        Args:
            production_board: LLM-produced production blueprint
            seed: Random seed for reproducibility
            return_portraits: Whether to return character portraits

        Returns:
            Tuple of (generated images list, optional portrait dict)
        """
        # Ensure initialization
        self.initialize()

        # Clear memory banks for new story (Bug 4 fix: ensure fresh start)
        if self._attn_processor is not None:
            self._attn_processor.clear_memory()
        if self._memory_bank is not None:
            self._memory_bank.clear()

        # Set seed for reproducibility
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[Generate] Starting story: {production_board.story_id}")
        print(f"[Generate] Total frames: {len(production_board.panels)}, Seed: {seed}")

        all_images = []
        portraits = {}
        first_image = None

        # Phase 1: Generate character portraits for feature extraction
        print("\n[Generate] Phase 1: Character Portrait Generation...")
        char_dict = {k: v.__dict__ for k, v in production_board.characters.items()}

        try:
            portraits = self.portrait_gen.generate_all_portraits(
                characters=char_dict,
                global_style=production_board.global_style,
                output_dir=f"outputs/portraits/{production_board.story_id}"
            )
            print(f"[Generate] Generated {len(portraits)} character portraits")
        except Exception as e:
            print(f"[Generate] Warning: Portrait generation failed: {e}")
            print("[Generate] Continuing without character portraits...")

        # Phase 2: Frame-by-frame generation
        print("\n[Generate] Phase 2: Frame Generation...")

        gen_params = self.config.get("generation_params", {
            "num_steps": 35,
            "guidance_scale": 7.5
        })

        for i, panel in enumerate(production_board.panels):
            print(f"\n{'=' * 50}")
            print(f"[Frame {i+1}/{len(production_board.panels)}]")
            print(f"Description: {panel.raw_prompt[:60]}...")
            print(f"{'=' * 50}")

            # Compose prompt
            prompt = self._compose_prompt(
                panel,
                production_board.global_style,
                production_board.characters
            )
            print(f"[Generate] Composed prompt: {prompt[:200]}...")

            negative_prompt = (
                "blurry, distorted, deformed, ugly, bad anatomy, "
                "extra limbs, watermark, text, signature, "
                "cartoon, anime style, low resolution"
            )

            # Get current characters in this panel
            current_chars = self._extract_characters_from_panel(panel, production_board.characters)

            # Generation parameters
            height = self.config.get("height", 1024)
            width = self.config.get("width", 1024)

            try:
                # Build call kwargs
                call_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": gen_params.get("num_steps", 35),
                    "guidance_scale": gen_params.get("guidance_scale", 7.5),
                    "generator": generator,
                }
                
                # For subsequent frames, use img2img with the first frame for consistency
                # This helps maintain character identity across scenes
                if i > 0 and first_image is not None:
                    call_kwargs["image"] = first_image
                    call_kwargs["strength"] = 0.3  # Lower = more consistent with first frame
                    print(f"[Generate] Using first frame for consistency (strength={call_kwargs['strength']})")
                
                output = self.base_pipe(**call_kwargs)

                current_image = output.images[0]
                all_images.append(current_image)

                # Update memory bank with current frame features
                self._update_memory(current_image)

                print(f"[Generate] Frame {i+1} completed successfully")

                # Save first frame reference (only once)
                if i == 0:
                    first_image = current_image

            except Exception as e:
                print(f"[Generate] Error generating frame {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Create placeholder for failed frame
                placeholder = Image.new('RGB', (height, width), color=(128, 128, 128))
                all_images.append(placeholder)

        print(f"\n{'=' * 60}")
        print(f"[Generate] Story generation complete! Generated {len(all_images)} frames")
        print(f"{'=' * 60}\n")

        if return_portraits:
            return all_images, portraits
        return all_images, None

    def _update_memory(self, image: Image.Image):
        """
        Update memory bank with features from generated image

        Args:
            image: PIL Image to extract features from
        """
        # Skip memory update if consistency is disabled
        if self._attn_processor is None and self._memory_bank is None:
            return
            
        try:
            # Extract simple features using VAE
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

            img_tensor = transform(image).unsqueeze(0).to(self.device, self.dtype)

            # Encode using VAE to get latent features
            with torch.no_grad():
                # Ensure VAE is on correct device and dtype (handles CPU offload)
                vae = self.base_pipe.vae.to(self.device)
                img_for_vae = img_tensor.to(dtype=vae.dtype)
                latent = vae.encode(
                    img_for_vae * 2 - 1  # Normalize to [-1, 1]
                ).latent_dist.sample()
                latent = latent * vae.config.scaling_factor

            # Ensure consistent dtype and device across all operations
            latent = latent.to(dtype=self.dtype, device=self.device)

            # Update attention processor memory
            if self._attn_processor is not None:
                b, c, h, w = latent.shape
                features = latent.view(b, c, h * w).permute(0, 2, 1)
                self._attn_processor.update_memory(features)

            # Update memory bank
            if self._memory_bank is not None:
                self._memory_bank.update(latent)

        except Exception as e:
            pass  # Silently skip memory update errors
            print(f"[Pipeline] Memory update warning: {e}")

    def save_story_images(
        self,
        images: List[Image.Image],
        story_id: str,
        panels: List[Panel],
        output_dir: str = "outputs/test_results"
    ) -> List[str]:
        """
        Save generated story images

        Args:
            images: List of generated PIL Images
            story_id: Story identifier
            panels: List of panels for metadata
            output_dir: Output directory

        Returns:
            List of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / f"{story_id}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, (img, panel) in enumerate(zip(images, panels)):
            # Save individual frame
            filename = f"frame_{i+1:02d}_{panel.shot_type}.png"
            filepath = save_dir / filename
            img.save(filepath)
            saved_paths.append(str(filepath))
            print(f"[Save] Saved: {filepath.name}")

        # Create and save storyboard
        storyboard = self._create_storyboard(images, panels)
        storyboard_path = save_dir / "storyboard.png"
        storyboard.save(storyboard_path)
        print(f"[Save] Storyboard saved: {storyboard_path.name}")

        return saved_paths

    def _create_storyboard(
        self,
        images: List[Image.Image],
        panels: List[Panel]
    ) -> Image.Image:
        """Create horizontal storyboard from generated images"""
        from PIL import ImageDraw, ImageFont

        # Target size
        target_height = 768
        target_width = int(target_height * 0.8)

        # Resize all images
        resized = [
            img.resize((target_width, target_height), Image.LANCZOS)
            for img in images
        ]

        # Horizontal layout
        spacing = 20
        total_width = sum(img.width for img in resized) + spacing * (len(resized) - 1)
        storyboard_height = target_height + 80

        storyboard = Image.new('RGB', (total_width, storyboard_height), color=(40, 40, 40))
        draw = ImageDraw.Draw(storyboard)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        x_offset = 0
        for i, (img, panel) in enumerate(zip(resized, panels)):
            storyboard.paste(img, (x_offset, 0))

            # Add frame number and scene description
            text = f"Scene {panel.panel_id}: {panel.shot_type}"
            draw.text((x_offset + 10, target_height + 10), text, fill=(200, 200, 200), font=font)

            x_offset += img.width + spacing

        return storyboard
