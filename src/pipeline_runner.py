#!/usr/bin/env python3
"""
enhanced_pipeline_runner.py
Refactored Pipeline Runner: Drives LLM+SDXL to generate story images from .txt input,
supports JSON intermediate file persistence.
Usage:
    python enhanced_pipeline_runner.py \
        --input data/TaskA/01.txt \
        --llm_path ./models/Qwen2.5-7B-Instruct \
        --save_json
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
        
        # Lazy initialization
        self.llm_tokenizer = None
        self.llm_model = None
        self.sdxl_gen = None
    
    def init_llm(self):
        """Lazy load LLM"""
        if self.llm_tokenizer is None or self.llm_model is None:
            self.llm_tokenizer, self.llm_model = load_llm(self.llm_path)
    
    def init_sdxl(self):
        """Lazy load SDXL"""
        if self.sdxl_gen is None:
            self.sdxl_gen = SDXLGenerator(self.sdxl_model_path, self.ip_adapter_path)
    
    def run_from_txt(self, input_txt_path: str, save_json: bool = False) -> bool:
        """
        Run full pipeline from txt file.
        Args:
            input_txt_path: Input .txt scene description file
            save_json: Whether to save LLM output JSON to disk
        Returns: Whether all generations succeeded
        """
        input_path = Path(input_txt_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_txt_path}")
        
        # === Phase 1: LLM parses txt to structured JSON ===
        print(f"\n[INPUT] Parsing: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        parsed_info = parse_raw_input(raw_text)
        story_id = derive_story_id(str(input_path))
        print(f"   StoryID: {story_id}, Subject: {parsed_info['subject_name']}, Frames: {len(parsed_info['scenes'])}")
        
        # Construct LLM prompt (reuse llm_processor functions)
        from llm_processor import build_user_prompt, SYSTEM_PROMPT
        user_prompt = build_user_prompt(parsed_info["subject_name"], parsed_info["scenes"])
        
        print("[LLM] Running inference...")
        self.init_llm()
        llm_raw_out = run_llm_inference(self.llm_tokenizer, self.llm_model, SYSTEM_PROMPT, user_prompt)
        
        story_data = parse_llm_output(llm_raw_out)
        if story_data is None:
            print("[ERROR] LLM output parsing failed")
            return False
        
        # Post-processing: inject real paths and correct raw_text
        story_data = post_process(story_data, story_id, str(self.results_root), parsed_info["scenes"])
        
        # Optional JSON save
        if save_json:
            json_dir = self.results_root / "json_data"
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"data{story_id.zfill(2)}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2, ensure_ascii=False)
            print(f"[JSON] Saved: {json_path}")
        
        # === Phase 2: SDXL generates images frame by frame ===
        print(f"\n[SDXL] Starting story generation: {story_id}...")
        self.init_sdxl()
        panels = story_data["panels"]
        
        for panel in sorted(panels, key=lambda x: x["index"]):  # Sort by index for order
            success = self._generate_panel(panel, story_id)
            if not success:
                print(f"[STOP] Panel {panel['index']} generation failed, aborting")
                return False
        
        print(f"[DONE] Story {story_id} complete! Output: {self.results_root / story_id}")
        return True
    
    def _generate_panel(self, panel: dict, story_id: str) -> bool:
        idx = panel["index"]
        prompt = panel["expanded_prompt"]
        neg_prompt = panel["negative_prompt"]

        # Calculate output path
        out_dir = self.results_root / story_id
        out_path = out_dir / f"panel_{idx}.png"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Global style prefix - PREPENDED so CLIP truncates style, not content
        # CLIP truncates from END, so we put less important style at the START
        GLOBAL_STYLE = "2d storyboard art, clean lines, flat colors, cel-shading, warm palette"
        full_prompt = f"{GLOBAL_STYLE}, {prompt}"

        print(f"\n[GENERATING] Panel {idx}: {prompt[:40]}...")

        # ========== First frame processing ==========
        if idx == 1:
            print("   [MODE] First frame (no reference)")
            success = self.sdxl_gen.generate_image(
                prompt=full_prompt,
                output_path=str(out_path),
                negative_prompt=neg_prompt,
                ip_ref_image=None,
                ctrl_ref_image=None
            )
            
            # After first frame generation, derive ControlNet control image
            if success and out_path.exists():
                control_path = out_dir / f"panel_{idx}_control.png"
                try:
                    from PIL import Image, ImageFilter
                    img = Image.open(out_path).convert("RGB")
                    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                    edges = img.filter(ImageFilter.FIND_EDGES)
                    edges.save(control_path)
                    print(f"[CONTROL] Control image generated: {control_path.name}")
                except Exception as e:
                    print(f"[WARN] Control image generation failed: {e}")
            return success

        # ========== Subsequent frame processing ==========
        else:
            print("   [MODE] Sequential frame (IP+ControlNet)")
            ip_ref_path = out_dir / f"panel_{idx-1}.png"       # IP: previous frame
            ctrl_ref_path = out_dir / "panel_1_control.png"    # Control: first frame edges

            # Defensive checks
            if not ip_ref_path.exists():
                print(f"[WARN] IP reference missing: {ip_ref_path}, falling back to text-only generation")
                ip_ref_path = None
            if not ctrl_ref_path.exists():
                print(f"[WARN] Control image missing: {ctrl_ref_path}, ControlNet will be disabled")
                ctrl_ref_path = None

            return self.sdxl_gen.generate_image(
                prompt=full_prompt,
                output_path=str(out_path),
                negative_prompt=neg_prompt,
                ip_ref_image=str(ip_ref_path) if ip_ref_path else None,
                ctrl_ref_image=str(ctrl_ref_path) if ctrl_ref_path else None
            )

def main():
    parser = argparse.ArgumentParser(description="Enhanced Pipeline: Generate story images from txt")
    parser.add_argument("--input", type=str, required=True, help="Input .txt scene file path")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model repo_id or local path")
    parser.add_argument("--sdxl_model", type=str, default="", help="SDXL model path")
    parser.add_argument("--ip_adapter", type=str, default="", help="IP-Adapter model path")
    parser.add_argument("--results_root", type=str, default="results", help="Image output root directory")
    parser.add_argument("--save_json", action="store_true", help="Save intermediate LLM JSON")
    
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
