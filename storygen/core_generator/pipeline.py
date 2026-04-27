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
from storygen.utils.image_utils import remove_white_borders


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
        characters: Dict,
        panel_index: int = 0,
        all_panels: List[Panel] = None,
        consistency_constraints: List[str] = None
    ) -> str:
        """
        Compose final generation prompt optimized for SDXL.
        Structure: Character (START) > Scene Description > Key Objects > Time > Style
        
        CRITICAL: Each panel MUST have character description. If LLM output is incomplete,
        we fall back to character data or scene content.
        """
        import re
        
        # === STEP 1: Extract character information ===
        present_char_names = self._extract_characters_from_panel(panel, characters)
        
        # CRITICAL FIX: DO NOT hardcode character count!
        # Let LLM flexibility handle stories with "meet his friends" or other multi-person scenarios
        # Only use character count hint if we have explicit information
        
        # Build full character description from character data
        # CRITICAL FIX: Use visual_description as SINGLE SOURCE OF TRUTH
        # Key_attributes and clothing may contradict visual_description
        char_descriptions = []
        for idx, char_name in enumerate(present_char_names):
            if char_name in characters:
                char = characters[char_name]
                desc = None
                
                if hasattr(char, 'visual_description') and char.visual_description:
                    # visual_description is the source of truth
                    # Verify it's not just "A person" (too generic)
                    if "person" not in char.visual_description.lower()[:30]:
                        desc = char.visual_description
                
                if not desc:
                    # Fallback: construct from key_attributes and clothing
                    parts = [char_name]
                    if hasattr(char, 'key_attributes') and char.key_attributes:
                        # Filter out contradictory attributes
                        valid_attrs = []
                        for attr in char.key_attributes:
                            attr_lower = str(attr).lower()
                            # Only keep non-contradictory attributes
                            if not any(exclude in attr_lower for exclude in ['person', 'generic']):
                                valid_attrs.append(str(attr))
                        parts.extend(valid_attrs[:3])
                    if hasattr(char, 'clothing') and char.clothing:
                        parts.append(str(char.clothing))
                    desc = ", ".join(parts)
                
                char_descriptions.append(desc)
        
        # Combine: character descriptions (no hardcoded count constraint)
        if char_descriptions:
            combined_char = ", ".join(char_descriptions)
        else:
            combined_char = ""
        
        # === STEP 2: Build scene description ===
        # CRITICAL FIX: Extract ONLY the scene/action part from enhanced_prompt
        # Character description is already handled in char_descriptions
        scene_desc = ""
        
        if panel.enhanced_prompt and len(panel.enhanced_prompt) > 15:
            raw_scene = panel.enhanced_prompt
            
            # Pattern 1: "Name, description, action" - find action after comma
            # Pattern 2: "Name action" - remove name at start
            action_verbs = ["walks", "sits", "stands", "looks", "waits", "pauses", "gets", "runs", 
                          "eating", "reading", "driving", "talking", "playing", "resting", 
                          "laughing", "makes", "meets", "enters", "exits", "arrives", "leaves",
                          "holds", "carries", "takes", "places", "opens", "closes", "turns"]
            
            # Strategy: Find the first action verb and extract from there
            verb_pattern = r'\b(' + '|'.join(action_verbs) + r')\b'
            match = re.search(verb_pattern, raw_scene, re.IGNORECASE)
            
            if match:
                verb_pos = match.start()
                # Extract everything from the verb onwards, but only up to quality terms
                potential_scene = raw_scene[verb_pos:]
                
                # Find where quality terms start
                quality_terms = ["photorealistic", "realistic photography", "sharp focus", 
                               "8k detailed", "highly detailed", "masterpiece"]
                quality_pos = len(potential_scene)
                for term in quality_terms:
                    pos = potential_scene.lower().find(term.lower())
                    if pos != -1 and pos < quality_pos:
                        quality_pos = pos
                
                scene_desc = potential_scene[:quality_pos].strip()
                
                # If scene_desc is too short or weird, fall back to raw_prompt
                if len(scene_desc) < 20:
                    scene_desc = re.sub(r'<[A-Z][a-z]+>\s*', '', panel.raw_prompt).strip()
            else:
                # No action verb found, use raw_prompt
                scene_desc = re.sub(r'<[A-Z][a-z]+>\s*', '', panel.raw_prompt).strip()
        else:
            # Fall back to raw_prompt
            scene_desc = re.sub(r'<[A-Z][a-z]+>\s*', '', panel.raw_prompt).strip()
        
        # Remove photorealistic/quality terms from scene_desc (we add them later)
        quality_terms = ["photorealistic", "realistic photography", "sharp focus", 
                        "8k detailed", "highly detailed", "masterpiece"]
        for term in quality_terms:
            scene_desc = re.sub(rf',\s*{re.escape(term)}', '', scene_desc, flags=re.IGNORECASE)
            scene_desc = re.sub(rf'{re.escape(term)}\s*,\s*', '', scene_desc, flags=re.IGNORECASE)
        scene_desc = scene_desc.rstrip(',. ')
        
        # === STEP 3: Compose full prompt ===
        prompt_parts = []
        
        # 3a. Character at START (includes EXACTLY N person + all character descriptions)
        if combined_char:
            prompt_parts.append(combined_char)
        
        # 3b. Scene action
        if scene_desc:
            prompt_parts.append(scene_desc)
        
        # 3c. Key objects
        if hasattr(panel, 'key_objects') and panel.key_objects:
            prompt_parts.append(f"with {panel.key_objects}")
        
        # 3d. Time of day
        if hasattr(panel, 'time_of_day') and panel.time_of_day:
            prompt_parts.append(panel.time_of_day)
        
        # 3e. Setting
        if hasattr(panel, 'setting') and panel.setting:
            prompt_parts.append(f"in {panel.setting}")
        
        # 3f. Style (ALWAYS last)
        prompt_parts.append("photorealistic, realistic photography, sharp focus, 8k detailed")
        
        # === STEP 4: Combine with truncation ===
        base_prompt = ", ".join(prompt_parts)
        
        # SDXL CLIP limit ~77 tokens ≈ 350 chars
        # Ensure we keep character description even if truncating
        MAX_LEN = 380
        if len(base_prompt) > MAX_LEN:
            char_part = prompt_parts[0] if prompt_parts else ""
            style_part = "photorealistic, realistic photography, sharp focus, 8k detailed"
            
            # Reserve space for character (200 chars) and style (60 chars)
            available = MAX_LEN - len(char_part) - len(style_part) - 10
            if available > 100:
                middle_parts = ", ".join(prompt_parts[1:-1])
                if len(middle_parts) > available:
                    middle_parts = middle_parts[:available]
                base_prompt = f"{char_part}, {middle_parts}, {style_part}"
        
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
        """
        Extract character names appearing in this panel.
        
        CRITICAL FIX: For multi-panel stories, we need to infer character presence:
        - Single-character stories: Character is present in ALL panels
        - Multi-character stories: Check pronouns (they/she/he) to determine count
        """
        import re
        
        present_chars = []
        
        # First, check explicit name mentions in raw_prompt
        for char_name in all_characters.keys():
            if f"<{char_name}>" in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                present_chars.append(char_name)
        
        # CRITICAL FIX: If no explicit mentions found, check story-level heuristics
        if not present_chars:
            all_names = list(all_characters.keys())
            num_chars = len(all_names)
            
            # Single-character story: character is in ALL panels
            if num_chars == 1:
                present_chars = all_names
            
            # Multi-character story: check pronouns
            raw_lower = panel.raw_prompt.lower()
            if 'they' in raw_lower or 'they\'re' in raw_lower:
                # All characters present
                present_chars = all_names
            elif 'she' in raw_lower or 'he' in raw_lower or 'her ' in raw_lower or 'his ' in raw_lower:
                # One character (but which one?) - try to infer from context
                # For simplicity, use first character
                if all_names:
                    present_chars = [all_names[0]]
        
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
                panel=panel,
                global_style=production_board.global_style,
                characters=production_board.characters,
                panel_index=i,
                all_panels=production_board.panels,
                consistency_constraints=production_board.consistency_constraints
            )
            print(f"[Generate] Composed prompt: {prompt[:200]}...")

            # Enhanced negative prompt to prevent common issues
            negative_prompt = (
                "blurry, blurry hands, blurry face, distorted, deformed, ugly, bad anatomy, "
                "extra limbs, missing limbs, fused fingers, too many fingers, "
                "missing fingers, extra fingers, poorly drawn hands, poorly drawn face, "
                "watermark, text, signature, cropped, out of frame, "
                "low quality, worst quality, jpeg artifacts, "
                "cartoon, anime style, illustration, painting, drawing, sketch, "
                "anime, manga, comic, 2D art style, 3D render, CGI, "
                "plastic looking, toy-like, over-saturated, oversaturated colors"
            )

            # Generation parameters
            height = self.config.get("height", 1024)
            width = self.config.get("width", 1024)

            try:
                # Build call kwargs - pure txt2img for best quality
                call_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": gen_params.get("num_steps", 35),
                    "guidance_scale": gen_params.get("guidance_scale", 7.5),
                    "generator": generator,
                }
                
                output = self.base_pipe(**call_kwargs)

                current_image = output.images[0]
                
                # FIXED: Remove white/gray borders from generated image
                # This addresses SDXL VAE border issues
                try:
                    current_image = remove_white_borders(current_image, threshold=180)
                except Exception:
                    pass  # Keep original if border removal fails
                
                all_images.append(current_image)

                # Update memory bank with current frame features
                self._update_memory(current_image)

                print(f"[Generate] Frame {i+1} completed successfully")

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
        FIXED: Proper device handling for model_cpu_offload scenarios

        Args:
            image: PIL Image to extract features from
        """
        # Skip memory update if consistency is disabled
        if self._attn_processor is None and self._memory_bank is None:
            return
            
        try:
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

            # Create tensor on CPU first, then move to the same device as VAE
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                # Get VAE device safely
                vae = self.base_pipe.vae
                
                # Move VAE to CPU for encoding, then back (safer for CPU offload)
                vae_device = next(vae.parameters()).device if hasattr(vae, 'parameters') and len(list(vae.parameters())) > 0 else torch.device('cpu')
                
                # Move tensor to VAE's device
                img_for_vae = img_tensor.to(device=vae_device, dtype=vae.dtype)
                
                # Encode
                latent = vae.encode(
                    img_for_vae * 2 - 1  # Normalize to [-1, 1]
                ).latent_dist.sample()
                latent = latent * vae.config.scaling_factor

            # Move latent to pipeline device for consistency processing
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
            # Silently skip memory update errors - don't disrupt generation
            pass

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
