#!/usr/bin/env python3
# File: src/sdxl_generator.py
# Description: Enhanced SDXL 1.0 + IP-Adapter + ControlNet for Consistent Story Generation
# Strategy: Use pure SDXL for 1st frame (bypass IP-Adapter bug), use IP+ControlNet for subsequent frames.
import os
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image, ImageFilter
import logging
from typing import Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SDXLGenerator:
    def __init__(self, model_path: str = None, ip_adapter_path: str = None, device="cuda"):
        """
        初始化包含 IP-Adapter 和 ControlNet 能力的生成器。
        """
        self.device = device
        self.model_path = model_path or Path(__file__).parent.parent / "models" / "sdxl" / "sd_xl_base_1.0.safetensors"
        self.ip_adapter_path = ip_adapter_path or Path(__file__).parent.parent / "models" / "ip-adapter" / "sdxl_models" / "ip-adapter_sdxl.safetensors"
        
        # --- 1. 创建基础管线 (主要用于首帧) ---
        self.pipe_basic = StableDiffusionXLPipeline.from_single_file(
            self.model_path.as_posix(),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        ).to(self.device)
        self.pipe_basic.scheduler = EulerDiscreteScheduler.from_config(self.pipe_basic.scheduler.config)

        # --- 2. 创建增强管线 (用于后续帧：IP+ControlNet) ---
        self.pipe_enhanced = StableDiffusionXLPipeline.from_single_file(
            self.model_path.as_posix(),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        )
        # 仅增强管线加载 IP-Adapter
        self.pipe_enhanced.load_ip_adapter(
            self.ip_adapter_path.parent.parent.as_posix(),
            subfolder="sdxl_models",
            weight_name=self.ip_adapter_path.name, 
            local_files_only=True,
            torch_dtype=torch.float16
        )
        self.pipe_enhanced.to(self.device)
        self.pipe_enhanced.scheduler = EulerDiscreteScheduler.from_config(self.pipe_enhanced.scheduler.config)

        # --- 3. ControlNet 状态 (懒加载) ---
        self.controlnet = None
        self.is_controlnet_loaded = False
        self.default_control_weight = 0.34  # 经验值

        print(f"✅ SDXL 双模式加载完成。基础模式(首帧) + 增强模式(IP+ControlNet)。设备: {self.device}")

    def _load_controlnet(self):
        """懒加载 ControlNet 模型"""
        if self.is_controlnet_loaded:
            return
        logger.info("⏳ 正在懒加载 ControlNet (Canny-SDXL)...")
        try:
            from diffusers import ControlNetModel
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16,
                cache_dir="./models",
                variant="fp16"
            ).to(self.device)
            # 挂载到增强管线
            self.pipe_enhanced.controlnet = self.controlnet
            self.is_controlnet_loaded = True
        except Exception as e:
            logger.error(f"❌ ControlNet 加载失败: {e}")
            self.controlnet = None

    @staticmethod
    def _get_canny_edges(image: Image.Image, threshold: int = 95) -> Image.Image:
        """
        使用 PIL 生成简易 Canny-like 边缘图 (零OpenCV依赖)。
        """
        gray_img = image.convert('L')
        blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=1.5)) # 稍强模糊去噪
        edge_img = blurred.filter(ImageFilter.FIND_EDGES)
        return edge_img.point(lambda x: 255 if x > threshold else 0)

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        *,
        ip_ref_image: Optional[Union[str, Image.Image]] = None,   # IP-Adapter: 人物参考
        ctrl_ref_image: Optional[Union[str, Image.Image]] = None, # ControlNet: 结构参考
        ctrl_weight: Optional[float] = None,
        width: int = 960,   # 微调分辨率以适应显存和多人物
        height: int = 896,
        num_inference_steps: int = 24,
        guidance_scale: float = 8.75,
    ) -> bool:
        """
        增强版生成函数。
        """
        logger.info(f"🎨 生成: {prompt[:55]}...")
        
        # 🛡️ 智能负向提示防御
        base_neg_prompt = negative_prompt or ""
        if any(word in prompt.lower() for word in ['dog', 'bird', 'animal', 'pet']):
            base_neg_prompt += ", human body, human face, anthropomorphic"

        try:
            # =============================================================
            # 🅰️ 情况 A：首帧生成 (无任何参考图)
            # =============================================================
            if ip_ref_image is None and ctrl_ref_image is None:
                print("   🅰️ 使用【基础模式】生成首帧...")
                # 使用未加载IP-Adapter的管线，避开 encoder_hid_dim_type 冲突
                result = self.pipe_basic(
                    prompt=prompt,
                    negative_prompt=base_neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            
            # =============================================================
            # 🅱️ 情况 B：后续帧生成 (有IP参考 或 Control参考)
            # =============================================================
            else:
                print("   🅱️ 使用【增强模式】(IP+ControlNet)...")
                # --- 1. 预处理 ControlNet ---
                control_image_tensor = None
                conditioning_scale = 0.0
                if ctrl_ref_image is not None:
                    self._load_controlnet()
                    if self.controlnet is not None:
                        if isinstance(ctrl_ref_image, str) and os.path.exists(ctrl_ref_image):
                            source_img = Image.open(ctrl_ref_image).convert("RGB")
                        else:
                            source_img = ctrl_ref_image
                        source_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
                        control_image_tensor = self._get_canny_edges(source_img)
                        conditioning_scale = ctrl_weight or self.default_control_weight
                        logger.info(f"🎛️  ControlNet 启用，强度: {conditioning_scale:.2f}")

                # --- 2. 预处理 IP-Adapter ---
                cross_attention_kwargs = {}  # 保持为空即可
                final_ip_image = None
                ip_adapter_scale_val = None   # 初始化变量

                if ip_ref_image is not None:
                    if isinstance(ip_ref_image, str) and os.path.exists(ip_ref_image):
                        final_ip_image = load_image(ip_ref_image).convert("RGB")
                        final_ip_image = final_ip_image.resize((width, height), Image.Resampling.LANCZOS)
                    else:
                        final_ip_image = ip_ref_image
                    # ✅ 计算权重值，但不放入 cross_attention_kwargs
                    ip_adapter_scale_val = 0.64 if control_image_tensor is None else 0.57

                # --- 3. 调用增强管线 ---
                result = self.pipe_enhanced(
                    prompt=prompt,
                    negative_prompt=base_neg_prompt,
                    image=control_image_tensor,
                    controlnet_conditioning_scale=conditioning_scale,
                    ip_adapter_image=final_ip_image,
                    ip_adapter_scale=ip_adapter_scale_val,  # ✅ 直接作为独立参数传递
                    cross_attention_kwargs=cross_attention_kwargs, # 留空或保留其他必要参数
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

            # --- 4. 统一保存 ---
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            logger.info(f"💾 已保存: {output_path}")
            return True

        except Exception as e:
            logger.exception(f"❌ SDXL 生成失败: {str(e)[:100]}")
            return False