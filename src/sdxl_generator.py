#!/usr/bin/env python3
# File: src/sdxl_generator.py
# Description: Optimized SDXL 1.0 + IP-Adapter + ControlNet with Single Pipeline.
import os
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, ControlNetModel
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
import logging
from typing import Optional, Union

# CRITICAL FIX: Enable expandable memory segments to combat VRAM fragmentation
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Chinese network mirror for faster downloads
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDXLGenerator:
    def __init__(self, model_path: str = None, ip_adapter_path: str = None, device="cuda"):
        """
        Initialization: Single pipeline mode. IP-Adapter and ControlNet are NOT preloaded.
        """
        self.device = device
        self.model_path = model_path or Path(__file__).parent.parent / "models" / "sdxl"
        self.ip_adapter_path = ip_adapter_path or Path(__file__).parent.parent / "models" / "ip-adapter" / "sdxl_models" / "ip-adapter_sdxl.safetensors"
        
        # --- 1. Load SDXL base model from HuggingFace (China-friendly via HF-Mirror) ---
        print(f"[SDXL] Loading model from HuggingFace (single pipeline, energy-saving mode)")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            cache_dir="./models",
            variant="fp16"
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # --- 2. Status flags (all initially unloaded) ---
        self.ip_adapter_loaded = False
        self.controlnet = None
        self.controlnet_loaded = False
        self.default_control_weight = 0.37  # Balanced control strength

        print(f"[SDXL] Initialization complete. IP/ControlNet in sleep mode.")

    def _load_ip_adapter_if_needed(self):
        """Load IP-Adapter only when needed, save VRAM"""
        if self.ip_adapter_loaded:
            return
        print(f"[IP-Adapter] Loading on demand...")
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.safetensors",
                torch_dtype=torch.float16
            )
            self.ip_adapter_loaded = True
        except Exception as e:
            logger.error(f"[ERROR] IP-Adapter loading failed: {e}")

    def _load_controlnet_if_needed(self):
        """Load ControlNet only when needed"""
        if self.controlnet_loaded:
            return
        logger.info("[ControlNet] Loading on demand...")
        try:
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16,
                cache_dir="./models",
                variant="fp16"
            ).to(self.device)
            self.pipe.controlnet = self.controlnet
            self.controlnet_loaded = True
        except Exception as e:
            logger.error(f"[ERROR] ControlNet loading failed: {e}")

    @staticmethod
    def _create_clean_control_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """
        Create ControlNet control image: Keep background edges, remove character details.
        """
        # 1. Strong blur to denoise, weaken character textures
        gray = image.convert('L')
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2.5))
        # 2. Extract edges
        edge_img = blurred.filter(ImageFilter.FIND_EDGES)
        # 3. Binarization and denoising
        edge_array = np.array(edge_img)
        binary_array = np.where(edge_array > 100, 250, 50).astype(np.uint8)
        binary_img = Image.fromarray(binary_array)
        # 4. Morphological denoising - fixed: use odd size (3)
        denoised = binary_img.filter(ImageFilter.MinFilter(size=3))
        smoothed = denoised.filter(ImageFilter.MaxFilter(size=3))
        return smoothed.resize((width, height), Image.Resampling.LANCZOS)

    def _prepare_controlnet_input(self, ctrl_ref_image, width: int, height: int) -> Optional[torch.Tensor]:
        """Prepare ControlNet input tensor"""
        try:
            if isinstance(ctrl_ref_image, str) and os.path.exists(ctrl_ref_image):
                source_img = Image.open(ctrl_ref_image).convert("RGB")
            else:
                source_img = ctrl_ref_image
            source_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Generate control image
            control_pil = self._create_clean_control_image(source_img, width, height)
            
            # Convert to Tensor [1, 3, H, W]
            control_array = np.array(control_pil.convert('L'))
            control_array = np.stack([control_array] * 3, axis=-1) # 3 channels
            control_tensor = torch.from_numpy(control_array).float().div(255.0)
            control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            return control_tensor
        except Exception as e:
            logger.error(f"[ERROR] Control image generation failed: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        *,
        ip_ref_image: Optional[Union[str, Image.Image]] = None,
        ctrl_ref_image: Optional[Union[str, Image.Image]] = None,
        ctrl_weight: Optional[float] = None,
        width: int = 992,    # 992 / 8 = 124 (perfect alignment)
        height: int = 1024,  # 1024 / 8 = 128
        num_inference_steps: int = 24,  # Reduced steps for VRAM savings
        guidance_scale: float = 8.6,
    ) -> bool:
        """
        VRAM-safe image generation function.
        """
        logger.info(f"[GEN] Generating: {prompt[:55]}...")
        
        # Initialize variables
        conditioning_scale = 0.0
        ip_adapter_scale_val = 0.0
        control_image_tensor = None
        final_ip_image = None
        
        # Negative prompt
        base_neg_prompt = negative_prompt or ""
        base_neg_prompt += ", blurry, distorted, deformed, bad anatomy, extra limbs"

        try:
            # =============================================================
            # CASE A: First frame (no reference images, most VRAM-efficient path)
            # =============================================================
            if ip_ref_image is None and ctrl_ref_image is None:
                print("   [MODE-A] Lightweight first frame generation...")
                # Key: pipe has no IP or ControlNet attached, lowest VRAM usage
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=base_neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            # =============================================================
            # CASE B: Subsequent frames (dynamically attach heavy components)
            # =============================================================
            else:
                print("   [MODE-B] Enhanced generation mode...")
                
                # --- 1. ControlNet background control ---
                if ctrl_ref_image is not None:
                    self._load_controlnet_if_needed()
                    control_image_tensor = self._prepare_controlnet_input(ctrl_ref_image, width, height)
                    conditioning_scale = ctrl_weight or self.default_control_weight
                    if control_image_tensor is not None:
                        logger.info(f"[ControlNet] Strength: {conditioning_scale:.2f}")
                    else:
                        logger.warning("[WARN] Control image generation failed, skipping ControlNet")

                # --- 2. IP-Adapter character consistency ---
                if ip_ref_image is not None:
                    self._load_ip_adapter_if_needed()
                    if isinstance(ip_ref_image, str) and os.path.exists(ip_ref_image):
                        final_ip_image = load_image(ip_ref_image).convert("RGB")
                        final_ip_image = final_ip_image.resize((width, height), Image.Resampling.LANCZOS)
                    else:
                        final_ip_image = ip_ref_image
                    # Weight strategy
                    ip_adapter_scale_val = 0.71 if control_image_tensor is not None else 0.79
                    logger.info(f"[IP-Adapter] Strength: {ip_adapter_scale_val:.2f}")

                # --- 3. Narrative enhancement ---
                enhanced_prompt = prompt
                if "looks out the window" in prompt.lower():
                    enhanced_prompt += ", side view, gazing out window"
                if "sits down" in prompt.lower():
                    enhanced_prompt += ", seated, book on table"

                # --- 4. Pipeline call ---
                call_kwargs = {
                    "prompt": enhanced_prompt,
                    "negative_prompt": base_neg_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
                # Inject conditions
                if control_image_tensor is not None:
                    call_kwargs["image"] = control_image_tensor
                    call_kwargs["controlnet_conditioning_scale"] = conditioning_scale
                if final_ip_image is not None:
                    call_kwargs["ip_adapter_image"] = final_ip_image
                    call_kwargs["ip_adapter_scale"] = ip_adapter_scale_val

                result = self.pipe(**call_kwargs).images[0]
                
                # EMERGENCY VRAM RELEASE: Immediately free large tensors
                if control_image_tensor is not None:
                    del control_image_tensor

            # --- 5. Save result ---
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            logger.info(f"[SAVE] Image saved: {output_path}")
            return True

        except torch.cuda.OutOfMemoryError:
            logger.error("[OOM] CUDA Out of Memory! Suggestion: 1) Restart terminal to clear VRAM 2) Lower resolution to 896x832")
            torch.cuda.empty_cache()
            return False
        except Exception as e:
            logger.exception(f"[ERROR] Generation failed: {str(e)[:120]}")
            return False
