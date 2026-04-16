#!/usr/bin/env python3
# File: src/pipeline_runner.py
# Description: 主运行脚本，连接 LLM 生成的 JSON 和 SDXL 生成器。
import json
import os
import sys
from pathlib import Path
from sdxl_generator import SDXLGenerator

class PipelineRunner:
    def __init__(self, json_file_path: str, results_root: str = "results"):
        """
        初始化运行器。
        
        Args:
            json_file_path: 思琪生成的 JSON 文件路径 (如 data/json_data/data01.json)
            results_root: 生成图像的根目录
        """
        self.json_file_path = Path(json_file_path)
        self.results_root = Path(results_root)
        
        # 解析 JSON 获取 Story ID
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.story_data = json.load(f)
        
        self.story_id = self.story_data['story_id']
        self.generator = None # 延迟初始化或在 run() 中初始化

    def run(self, model_path=None, ip_adapter_path=None):
        """执行完整的生成流程"""
        print(f"🚀 开始运行 Pipeline 生成故事: {self.story_id}")
        
        # 1. 初始化生成器
        self.generator = SDXLGenerator(model_path, ip_adapter_path)
        
        # 2. 遍历每一帧
        panels = self.story_data['panels']
        for panel in panels:
            success = self._generate_single_panel(panel)
            if not success:
                print(f"🛑 第 {panel['index']} 帧生成失败，流程终止。")
                sys.exit(1)
                
        print(f"🎉 故事 {self.story_id} 生成完毕！")

    def _generate_single_panel(self, panel: dict) -> bool:
        """
        处理单个 Panel 的生成逻辑。
        这里负责路径拼接和参数映射。
        """
        index = panel['index']
        prompt = panel['expanded_prompt']
        negative_prompt = panel['negative_prompt']
        ref_image_path = panel['reference_image'] # 这个路径在 JSON 里可能是相对的或旧的，我们需要根据 results_root 重算
        
        # --- 路径计算逻辑 ---
        # 我们希望输出到: results_root/{story_id}/panel_{index}.png
        output_subdir = self.results_root / self.story_id
        output_filename = f"panel_{index}.png"
        output_full_path = output_subdir / output_filename
        
        # --- 参考图像路径修正 ---
        # JSON 里的 reference_image 是给下一张图用的提示，我们需要根据当前生成的路径来构建它
        # 例如：当前生成 panel_2，它需要参考 panel_1，路径应为 output_subdir/panel_1.png
        actual_ref_path = None
        if ref_image_path: 
            # 如果 JSON 里有路径，尝试用新路径代替（或者直接构建新路径）
            # 这里我们直接构建标准路径：如果生成 panel_2，参考图就是 panel_1
            ref_index = index - 1
            potential_ref = output_subdir / f"panel_{ref_index}.png"
            if potential_ref.exists():
                actual_ref_path = str(potential_ref)
            else:
                print(f"⚠️ 警告: 参考图像 {potential_ref} 不存在，将进行纯文本生成。")
                # 如果文件不存在（比如第一次跑），降级为 txt2img

        print(f"\n--- 正在处理 Panel {index} ---")
        
        # 3. 调用生成器
        return self.generator.generate_image(
            prompt=prompt,
            output_path=str(output_full_path),
            negative_prompt=negative_prompt,
            reference_image_path=actual_ref_path, # 传给 SDXL
            # 可以在这里覆盖默认参数
            ip_adapter_scale=0.4 if actual_ref_path else None # 第一帧不需要 scale
        )

# --- Main Entry ---
if __name__ == "__main__":
    # 简单的命令行接口
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--results", type=str, default="results", help="输出结果目录")
    args = parser.parse_args()

    runner = PipelineRunner(args.json, args.results)
    runner.run()