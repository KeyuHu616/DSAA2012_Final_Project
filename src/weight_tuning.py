#!/usr/bin/env python3
"""
Weight Tuning Experiment Script
Runs SDXL generation with different ControlNet and IP-Adapter weight combinations.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sdxl_generator import SDXLGenerator
from llm_processor import parse_raw_input, derive_story_id, load_llm, run_llm_inference, parse_llm_output, post_process, build_user_prompt, SYSTEM_PROMPT


class WeightTuningRunner:
    """Run experiments with different weight configurations."""
    
    # Predefined weight combinations
    WEIGHT_COMBINATIONS = {
        "D_current": {"controlnet": 0.37, "ip_with_control": 0.71, "ip_without_control": 0.79},
        "A_conservative": {"controlnet": 0.45, "ip_with_control": 0.75, "ip_without_control": 0.83},
        "B_balanced": {"controlnet": 0.30, "ip_with_control": 0.80, "ip_without_control": 0.87},
        "C_aggressive": {"controlnet": 0.25, "ip_with_control": 0.85, "ip_without_control": 0.90},
    }
    
    def __init__(self, llm_path: str, base_results_root: str = "results"):
        self.llm_path = llm_path
        self.base_results_root = Path(base_results_root)
        self.llm_tokenizer = None
        self.llm_model = None
        
    def init_llm(self):
        if self.llm_tokenizer is None:
            self.llm_tokenizer, self.llm_model = load_llm(self.llm_path)
    
    def run_experiment(self, input_txt: str, weight_key: str, story_ids: list = None) -> dict:
        """Run single weight combination experiment."""
        
        if weight_key not in self.WEIGHT_COMBINATIONS:
            raise ValueError(f"Unknown weight key: {weight_key}. Available: {list(self.WEIGHT_COMBINATIONS.keys())}")
        
        weights = self.WEIGHT_COMBINATIONS[weight_key]
        results_dir = self.base_results_root / f"tuning_{weight_key}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {weight_key}")
        print(f"ControlNet: {weights['controlnet']}, IP-Adapter: {weights['ip_with_control']}/{weights['ip_without_control']}")
        print(f"{'='*60}")
        
        # Initialize SDXL with custom weights
        generator = SDXLGenerator()
        generator.default_control_weight = weights['controlnet']
        
        # Load LLM
        self.init_llm()
        
        # Determine which stories to run
        if story_ids:
            story_files = [f"data/TaskA/{sid}.txt" for sid in story_ids]
        else:
            story_files = [
                "data/TaskA/01.txt",  # Single character, female
                "data/TaskA/02.txt",  # Gender-sensitive name (Ryan)
                "data/TaskA/06.txt",  # Two characters
            ]
        
        results = {"weight_key": weight_key, "weights": weights, "stories": {}}
        
        for story_file in story_files:
            story_path = Path(story_file)
            if not story_path.exists():
                print(f"[SKIP] File not found: {story_file}")
                continue
                
            story_id = derive_story_id(str(story_path))
            print(f"\n--- Processing Story: {story_id} ---")
            
            # Parse input
            with open(story_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            parsed_info = parse_raw_input(raw_text)
            user_prompt = build_user_prompt(parsed_info["subject_name"], parsed_info["scenes"])
            
            # LLM inference
            print(f"[LLM] Generating prompts for {story_id}...")
            llm_out = run_llm_inference(self.llm_tokenizer, self.llm_model, SYSTEM_PROMPT, user_prompt)
            story_data = parse_llm_output(llm_out)
            
            if story_data is None:
                print(f"[ERROR] LLM parsing failed for {story_id}")
                continue
            
            story_data = post_process(story_data, story_id, str(results_dir), parsed_info["scenes"])
            
            # Generate panels
            panels = story_data["panels"]
            panel_results = {"success": [], "failed": []}
            
            GLOBAL_STYLE = "Clean storyboard-style digital illustration, soft ink outlines, flat-wash color fills, mild cel-shading, warm and approachable color palette, 2d art style"
            
            for panel in sorted(panels, key=lambda x: x["index"]):
                idx = panel["index"]
                full_prompt = f"{GLOBAL_STYLE}, {panel['expanded_prompt']}"
                out_path = results_dir / story_id / f"panel_{idx}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Determine IP and ControlNet references
                ip_ref = None
                ctrl_ref = None
                
                if idx > 1:
                    ip_ref = results_dir / story_id / f"panel_{idx-1}.png"
                    if not ip_ref.exists():
                        ip_ref = None
                    ctrl_ref = results_dir / story_id / "panel_1_control.png"
                    if not ctrl_ref.exists():
                        ctrl_ref = None
                
                # Update IP weight based on ControlNet presence
                ip_weight = weights['ip_without_control'] if ctrl_ref is None else weights['ip_with_control']
                
                print(f"  Panel {idx}: CN={weights['controlnet']}, IP={ip_weight}")
                
                success = generator.generate_image(
                    prompt=full_prompt,
                    output_path=str(out_path),
                    negative_prompt=panel['negative_prompt'],
                    ip_ref_image=str(ip_ref) if ip_ref else None,
                    ctrl_ref_image=str(ctrl_ref) if ctrl_ref else None,
                    ctrl_weight=weights['controlnet'],
                )
                
                if success and out_path.exists():
                    panel_results["success"].append(idx)
                    # Generate control image for next frame
                    if idx == 1:
                        control_path = out_path.parent / "panel_1_control.png"
                        try:
                            from PIL import Image, ImageFilter
                            img = Image.open(out_path).convert("RGB")
                            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                            edges = img.filter(ImageFilter.FIND_EDGES)
                            edges.save(control_path)
                        except Exception as e:
                            print(f"[WARN] Control image failed: {e}")
                else:
                    panel_results["failed"].append(idx)
            
            results["stories"][story_id] = panel_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Weight Tuning Experiment")
    parser.add_argument("--weight", type=str, required=True, 
                        choices=["D_current", "A_conservative", "B_balanced", "C_aggressive", "all"],
                        help="Weight combination to test")
    parser.add_argument("--stories", type=str, default="01,02,06",
                        help="Comma-separated story IDs (default: 01,02,06)")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM model path")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Base results directory")
    
    args = parser.parse_args()
    
    runner = WeightTuningRunner(args.llm_path, args.results_root)
    story_ids = args.stories.split(",") if args.stories else None
    
    if args.weight == "all":
        all_results = {}
        for weight_key in ["D_current", "A_conservative", "B_balanced", "C_aggressive"]:
            print(f"\n\n{'#'*70}")
            print(f"# RUNNING EXPERIMENT: {weight_key}")
            print(f"{'#'*70}")
            result = runner.run_experiment(None, weight_key, story_ids)
            all_results[weight_key] = result
        print("\n\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*70)
        for key, res in all_results.items():
            print(f"\n{key}:")
            for sid, panels in res["stories"].items():
                print(f"  {sid}: Success={panels['success']}, Failed={panels['failed']}")
    else:
        result = runner.run_experiment(None, args.weight, story_ids)
        print(f"\nExperiment complete: {result}")


if __name__ == "__main__":
    main()
