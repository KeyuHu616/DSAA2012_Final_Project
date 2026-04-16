#!/usr/bin/env python3
# File: src/sdxl_generator.py
# Description: SDXL 1.0 + IP-Adapter for Consistent Character Generation
import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image

class SDXLGenerator:
    def __init__(self, model_path: str = None, ip_adapter_path: str = None, device="cuda"):
        """
        初始化包含 IP-Adapter 的生成器。
        
        Args:
            model_path: SDXL 基础模型路径 (.safetensors 或 目录)
            ip_adapter_path: IP-Adapter 模型路径 (通常为 'ip-adapter_sdxl_vit-h.safetensors')
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path or Path(__file__).parent.parent / "models" / "sdxl" / "sd_xl_base_1.0.safetensors"
        self.ip_adapter_path = ip_adapter_path or Path(__file__).parent.parent / "models" / "ip-adapter" / "sdxl_models" / "ip-adapter_sdxl.safetensors"
        
        # 1. 初始化基础 Pipeline
        # 注意：我们使用 from_single_file 需要确保环境支持
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path.as_posix(),
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        )
        
        # 2. 加载 IP-Adapter
        # 参考文档: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
        # self.pipe.load_ip_adapter(
        #     "h94/IP-Adapter", # 如果你下载了本地文件，这里可以改为本地路径字符串，或者直接用下面的 `ip_adapter_ckpt` 参数指定本地路径
        #     subfolder="sdxl_models",
        #     weight_name="ip-adapter_sdxl_vit-h.safetensors", # 如果你指定了本地路径，请确保文件存在
        #     image_encoder_folder="models/image_encoder", # 如果有本地 image_encoder
        #     torch_dtype=torch.float16
        # )
        # 如果你是直接加载本地下载的 checkpoint，请使用以下方式代替上面的 load_ip_adapter：
        self.pipe.load_ip_adapter(self.ip_adapter_path.parent.parent.as_posix(),
                                  subfolder="sdxl_models",
                                  weight_name=self.ip_adapter_path.name, 
                                  local_files_only=True,
                                  torch_dtype=torch.float16)

        # 3. 启用必要的内存优化
        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload() # 自动管理 GPU/CPU 内存，对于大模型非常必要
        
        # 4. 设置调度器 (Scheduler)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

        # 5. 将 pipeline 移动到设备
        self.pipe.to(self.device)

        print(f"✅ SDXL + IP-Adapter 加载完成。设备: {self.device}")

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = None,
                       width: int = 1024, height: int = 1024, num_inference_steps: int = 30,
                       guidance_scale: float = 7.5, ip_adapter_scale: float = 0.6,
                       reference_image_path: str = None) -> bool:
        """
        核心生成函数。
        如果 reference_image_path 为 None (第一帧)，使用 txt2img。
        如果 reference_image_path 存在 (后续帧)，使用 img2img + IP-Adapter。

        Args:
            reference_image_path: 参考图像路径 (来自 JSON 的 reference_image 字段)
            ip_adapter_scale: IP-Adapter 的控制强度 (0.0 - 1.0, 0.6 通常效果较好)
        """
        print(f"🎨 正在生成: {prompt[:50]}...")
        print(f"💾 输出路径: {output_path}")
        
        try:
            dummy_image = Image.new('RGB', (512, 512), color='black')
            # 处理参考图像
            if reference_image_path and os.path.exists(reference_image_path):
                print(f"🔗 检测到参考图像: {reference_image_path} (启用 IP-Adapter)")
                # 加载图像
                reference_image = load_image(reference_image_path).convert("RGB")
                
                # 启用 IP-Adapter 并设置权重
                self.pipe.set_ip_adapter_scale(ip_adapter_scale)
                
                # 使用 img2img 进行生成 (因为需要参考图像)
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=reference_image, # 传入参考图
                    # 注意: IP-Adapter 会自动从 reference_image 提取特征
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    # img2img 特有参数
                    strength=0.4, # 噪声强度 (0.3-0.6), 控制参考图对新图的影响程度
                ).images[0]
            else:
                # 第一帧：纯文本生成
                print("🆕 无参考图像 (第一帧)，使用纯文本生成")
                self.pipe.set_ip_adapter_scale(0.0) # 关闭 IP-Adapter 影响
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=dummy_image, 
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

            # 保存图像
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            print(f"✅ 成功保存: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return False