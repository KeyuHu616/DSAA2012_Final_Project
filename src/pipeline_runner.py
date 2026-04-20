#!/usr/bin/env python3
"""
enhanced_pipeline_runner.py
重构增强版Pipeline Runner：从.txt输入直接驱动LLM+SDXL生成故事图片，支持JSON中间文件持久化
用法示例：
    python enhanced_pipeline_runner.py \
        --input data/TaskA/01.txt \
        --llm_path ./models/Qwen2.5-7B-Instruct \
        --sdxl_model ./models/sd_xl_base_1.0.safetensors \
        --ip_adapter ./models/ip-adapter_sdxl.safetensors \
        --save_json  # 可选保存中间JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path

from llm_processor import parse_raw_input, derive_story_id, load_llm, run_llm_inference, parse_llm_output, post_process
from sdxl_generator import SDXLGenerator


class EnhancedPipelineRunner:
    def __init__(self, llm_path: str, sdxl_model_path: str, ip_adapter_path: str, results_root: str = "results"):
        self.llm_path = llm_path
        self.sdxl_model_path = sdxl_model_path
        self.ip_adapter_path = ip_adapter_path
        self.results_root = Path(results_root)
        
        # 延迟初始化组件
        self.llm_tokenizer = None
        self.llm_model = None
        self.sdxl_gen = None
    
    def init_llm(self):
        """懒加载LLM"""
        if self.llm_tokenizer is None or self.llm_model is None:
            self.llm_tokenizer, self.llm_model = load_llm(self.llm_path)
    
    def init_sdxl(self):
        """懒加载SDXL"""
        if self.sdxl_gen is None:
            self.sdxl_gen = SDXLGenerator(self.sdxl_model_path, self.ip_adapter_path)
    
    def run_from_txt(self, input_txt_path: str, save_json: bool = False) -> bool:
        """
        从txt文件运行完整流水线
        Args:
            input_txt_path: 输入.txt场景描述文件
            save_json: 是否将LLM输出的JSON保存到磁盘
        Returns: 是否全部生成成功
        """
        input_path = Path(input_txt_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_txt_path}")
        
        # === 阶段1: LLM解析txt生成结构化JSON ===
        print(f"\n📖 解析输入: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        parsed_info = parse_raw_input(raw_text)
        story_id = derive_story_id(str(input_path))
        print(f"   StoryID: {story_id}, 主体: {parsed_info['subject_name']}, 共{len(parsed_info['scenes'])}帧")
        
        # 构造LLM提示词（复用llm_processor内部函数）
        from llm_processor import build_user_prompt, SYSTEM_PROMPT
        user_prompt = build_user_prompt(parsed_info["subject_name"], parsed_info["scenes"])
        
        print("🧠 运行LLM生成视觉提示...")
        self.init_llm()
        llm_raw_out = run_llm_inference(self.llm_tokenizer, self.llm_model, SYSTEM_PROMPT, user_prompt)
        
        story_data = parse_llm_output(llm_raw_out)
        if story_data is None:
            print("❌ LLM输出解析失败")
            return False
        
        # 后处理：注入真实路径与修正raw_text
        story_data = post_process(story_data, story_id, str(self.results_root), parsed_info["scenes"])
        
        # 可选保存JSON
        if save_json:
            json_dir = self.results_root / "json_data"
            json_dir.mkdir(exist_ok=True)
            json_path = json_dir / f"data{story_id.zfill(2)}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)
            print(f"💾 已保存中间JSON: {json_path}")
        
        # === 阶段2: SDXL逐帧生成图片 ===
        print(f"\n🚀 启动SDXL生成故事 {story_id}...")
        self.init_sdxl()
        panels = story_data["panels"]
        
        for panel in sorted(panels, key=lambda x: x["index"]):  # 按index排序保证顺序
            success = self._generate_panel(panel, story_id)
            if not success:
                print(f"🛑 Panel {panel['index']} 生成失败，终止流程")
                return False
        
        print(f"🎉 故事 {story_id} 所有帧生成完成！输出至: {self.results_root / story_id}")
        return True
    
    def _generate_panel(self, panel: dict, story_id: str) -> bool:
        idx = panel["index"]
        prompt = panel["expanded_prompt"]
        neg_prompt = panel["negative_prompt"]

        # 计算输出路径
        out_dir = self.results_root / story_id
        out_path = out_dir / f"panel_{idx}.png"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 全局画风前缀
        GLOBAL_STYLE = "Clean storyboard-style digital illustration, soft ink outlines, flat-wash color fills, mild cel-shading, warm and approachable color palette, 2d art style"
        full_prompt = f"{GLOBAL_STYLE}, {prompt}"

        print(f"\n🎬 生成 Panel {idx}: {prompt[:40]}...")

        # ========== 首帧处理 ==========
        if idx == 1:
            print("   🆕 首帧模式 (无参考图)")
            # ✅ 修正：使用新参数名，显式置空
            success = self.sdxl_gen.generate_image(
                prompt=full_prompt,
                output_path=str(out_path),
                negative_prompt=neg_prompt,
                ip_ref_image=None,    # 新参数名
                ctrl_ref_image=None   # 新参数名
            )
            
            # ✅ 首帧生成后，立即派生 ControlNet 控制图
            if success and out_path.exists():
                control_path = out_dir / f"panel_{idx}_control.png"
                try:
                    from PIL import Image, ImageFilter
                    img = Image.open(out_path).convert("RGB")
                    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                    edges = img.filter(ImageFilter.FIND_EDGES)
                    edges.save(control_path)
                    print(f"💡 已生成控制图: {control_path.name}")
                except Exception as e:
                    print(f"⚠️ 控制图生成失败: {e}")
            return success

        # ========== 后续帧处理 ==========
        else:
            print("   🔗 连续帧模式 (IP+ControlNet)")
            # ✅ 修正：分离两种参考图
            ip_ref_path = out_dir / f"panel_{idx-1}.png"       # IP用：上一帧
            ctrl_ref_path = out_dir / "panel_1_control.png"    # Control用：首帧边缘

            # 防御性检查
            if not ip_ref_path.exists():
                print(f"⚠️  IP参考图缺失: {ip_ref_path}, 降级为文本生成")
                ip_ref_path = None
            if not ctrl_ref_path.exists():
                print(f"⚠️  控制图缺失: {ctrl_ref_path}, ControlNet将关闭")
                ctrl_ref_path = None

            # ✅ 调用新接口
            return self.sdxl_gen.generate_image(
                prompt=full_prompt,
                output_path=str(out_path),
                negative_prompt=neg_prompt,
                ip_ref_image=str(ip_ref_path) if ip_ref_path else None,
                ctrl_ref_image=str(ctrl_ref_path) if ctrl_ref_path else None
            )

def main():
    parser = argparse.ArgumentParser(description="增强版Pipeline：从txt直接生成故事图片")
    parser.add_argument("--input", type=str, required=True, help="输入.txt场景文件路径")
    parser.add_argument("--llm_path", type=str, default="./models/llm/Qwen2.5-7B-Instruct", help="本地LLM模型路径")
    parser.add_argument("--sdxl_model", type=str, default="", help="SDXL模型路径")
    parser.add_argument("--ip_adapter", type=str, default="", help="IP-Adapter模型路径")
    parser.add_argument("--results_root", type=str, default="results", help="图片输出根目录")
    parser.add_argument("--save_json", action="store_true", help="是否保存LLM生成的中间JSON文件")
    
    args = parser.parse_args()
    
    runner = EnhancedPipelineRunner(
        llm_path=args.llm_path,
        sdxl_model_path=args.sdxl_model,
        ip_adapter_path=args.ip_adapter,
        results_root=args.results_root
    )
    
    success = runner.run_from_txt(args.input, args.save_json)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()