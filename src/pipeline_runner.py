"""
Pipeline Runner Module - White-Box Story Pipeline Orchestrator

Main orchestrator that coordinates:
1. LLM semantic decoupling (llm_processor)
2. White-box SDXL generation with MSA/CSA (sdxl_generator)
3. Best-of-N evaluation and selection (evaluator)

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
import time
from dataclasses import dataclass

from llm_processor import LLMProcessor, StoryData, Character, Frame
from sdxl_generator import WhiteBoxSDXLGenerator, ConsistencyConfig
from evaluator import StoryEvaluator, SimpleEvaluator


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the story pipeline."""
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    llm_path: str = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir: str = "./models"
    output_dir: str = "./results"
    
    height: int = 832
    width: int = 896
    num_inference_steps: int = 24
    guidance_scale: float = 8.6
    base_seed: int = 42
    n_candidates: int = 2
    
    share_step_ratio: float = 0.5
    msa_inject_scale: float = 0.35
    
    use_evaluator: bool = True
    use_aesthetic: bool = False
    eval_weights: Optional[Dict] = None
    
    low_vram_mode: bool = True
    max_characters: int = 2
    
    def __post_init__(self):
        if self.eval_weights is None:
            self.eval_weights = {"dino": 0.40, "clip": 0.35, "aesthetic": 0.25}


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class WhiteBoxStoryPipeline:
    """
    Main orchestrator for the White-Box Story Pipeline.
    
    Coordinates LLM processing, SDXL generation, and evaluation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.llm: Optional[LLMProcessor] = None
        self.generator: Optional[WhiteBoxSDXLGenerator] = None
        self.evaluator = None
        self.is_initialized = False
        self.current_story_data = None

    def initialize(self):
        """Initialize all components."""
        if self.is_initialized:
            return
        
        print("=" * 60)
        print("Initializing White-Box Story Pipeline")
        print("=" * 60)
        
        consistency_config = ConsistencyConfig(
            share_step_ratio=self.config.share_step_ratio,
            msa_inject_scale=self.config.msa_inject_scale,
            enable_shared_attention=True,
            enable_msa=True
        )
        
        # LLM Processor
        print("\n[Pipeline] Initializing LLM Processor...")
        try:
            self.llm = LLMProcessor(model_path=self.config.llm_path, device="cuda")
            print("[Pipeline] LLM ready")
        except Exception as e:
            print(f"[Pipeline] LLM init failed: {e}")
            self.llm = None
        
        # SDXL Generator
        print("\n[Pipeline] Initializing SDXL Generator...")
        try:
            self.generator = WhiteBoxSDXLGenerator(
                model_path=self.config.model_path,
                device="cuda",
                consistency_config=consistency_config
            )
            self.generator.load_components(low_vram_mode=self.config.low_vram_mode)
            print("[Pipeline] SDXL ready")
        except Exception as e:
            print(f"[Pipeline] SDXL init failed: {e}")
            raise RuntimeError(f"SDXL initialization failed: {e}")
        
        # Evaluator
        if self.config.use_evaluator:
            print("\n[Pipeline] Initializing Evaluator...")
            try:
                eval_class = StoryEvaluator if self.config.use_aesthetic else SimpleEvaluator
                self.evaluator = eval_class(weights=self.config.eval_weights)
                print("[Pipeline] Evaluator ready")
            except Exception as e:
                print(f"[Pipeline] Evaluator init failed: {e}")
                self.evaluator = None
        else:
            self.evaluator = None
        
        self.is_initialized = True
        print("\n" + "=" * 60)
        print("Pipeline initialized!")
        print("=" * 60)

    def run(self, input_path: str, story_id: Optional[str] = None) -> bool:
        """Run complete pipeline on an input story file."""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        story_id = story_id or Path(input_path).stem
        
        print(f"\n{'='*60}")
        print(f"Running Pipeline: {story_id}")
        print(f"{'='*60}")
        
        # Setup output
        output_dir = os.path.join(self.config.output_dir, story_id)
        os.makedirs(output_dir, exist_ok=True)
        json_dir = os.path.join(self.config.output_dir, "json_data")
        os.makedirs(json_dir, exist_ok=True)
        
        # Step 1: LLM semantic decoupling
        print("\n[Step 1/4] LLM Semantic Decoupling...")
        try:
            if self.llm is not None:
                story_data = self.llm.process(input_path)
            else:
                story_data = self._simple_parse(input_path, story_id)
        except Exception as e:
            print(f"[Pipeline] LLM failed: {e}")
            story_data = self._simple_parse(input_path, story_id)
        
        self.current_story_data = story_data
        
        # Save parsed JSON
        json_path = os.path.join(json_dir, f"data{story_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(story_data.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Saved] JSON: {json_path}")
        
        characters = story_data.characters
        frames = story_data.frames
        
        if not characters or not frames:
            print("[Pipeline] FATAL: No characters or frames")
            return False
        
        primary_char = characters[0]
        char_prompt = primary_char.global_prompt
        print(f"\n[Character] {primary_char.id}: {char_prompt}")
        
        # Step 2: Generate first frame
        print("\n[Step 2/4] Generating First Frame...")
        frame1_data = frames[0]
        
        try:
            frame1_img = self.generator.generate_single_frame(
                prompt=frame1_data.full_prompt,
                char_prompt=char_prompt,
                neg_prompt=frame1_data.negative_prompt,
                height=self.config.height,
                width=self.config.width,
                num_steps=self.config.num_inference_steps,
                guidance=self.config.guidance_scale,
                seed=self.config.base_seed
            )
            
            frame1_path = os.path.join(output_dir, f"panel_{frame1_data.index:02d}.png")
            frame1_img.save(frame1_path)
            print(f"[Saved] Frame 1: {frame1_path}")
            
            prev_best_img = frame1_img
        except Exception as e:
            print(f"[Pipeline] Frame 1 failed: {e}")
            return False
        
        # Step 3: Generate subsequent frames
        print("\n[Step 3/4] Generating Subsequent Frames...")
        generated_frames = [frame1_img]
        
        for i, frame_data in enumerate(frames[1:], start=2):
            print(f"\n--- Frame {i}/{len(frames)} ---")
            
            try:
                candidates = self.generator.generate_with_consistency(
                    prompt=frame_data.full_prompt,
                    char_prompt=char_prompt,
                    neg_prompt=frame_data.negative_prompt,
                    height=self.config.height,
                    width=self.config.width,
                    num_steps=self.config.num_inference_steps,
                    guidance=self.config.guidance_scale,
                    seed=self.config.base_seed + i * 100,
                    frame_index=i,
                    n_candidates=self.config.n_candidates
                )
                
                # Select best
                if self.evaluator and len(candidates) > 1:
                    best_idx, scores = self.evaluator.evaluate_candidates(
                        candidates=candidates,
                        prev_best_img=prev_best_img,
                        current_prompt=frame_data.full_prompt
                    )
                    selected_img = candidates[best_idx]
                else:
                    selected_img = candidates[0]
                    best_idx = 0
                
                frame_path = os.path.join(output_dir, f"panel_{frame_data.index:02d}.png")
                selected_img.save(frame_path)
                print(f"[Saved] Frame {i}: {frame_path}")
                
                generated_frames.append(selected_img)
                prev_best_img = selected_img
                
                if i % 5 == 0:
                    self.generator.clear_cache()
                
            except Exception as e:
                print(f"[Pipeline] Frame {i} failed: {e}")
                continue
        
        # Step 4: Save summary
        print("\n[Step 4/4] Saving Summary...")
        
        elapsed = time.time() - start_time
        summary = {
            "story_id": story_id,
            "total_frames": len(frames),
            "generated_frames": len(generated_frames),
            "character": primary_char.to_dict(),
            "config": {
                "height": self.config.height,
                "width": self.config.width,
                "num_steps": self.config.num_inference_steps,
                "n_candidates": self.config.n_candidates,
                "share_step_ratio": self.config.share_step_ratio,
                "msa_inject_scale": self.config.msa_inject_scale
            },
            "timing": {"total_seconds": elapsed, "seconds_per_frame": elapsed / len(frames) if frames else 0}
        }
        
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Complete: {story_id} - {len(generated_frames)}/{len(frames)} frames - {elapsed:.1f}s")
        print(f"{'='*60}")
        
        return True

    def _simple_parse(self, input_path: str, story_id: str) -> StoryData:
        """Simple fallback parser without LLM."""
        print("[Pipeline] Using simple fallback parser")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        import re
        scenes = re.split(r'\[SCENE-\d+\]|\[SEP\]', text)
        
        frames = []
        for i, scene in enumerate(scenes):
            scene = scene.strip()
            if len(scene) > 5:
                frames.append(Frame(
                    index=i + 1,
                    action=scene[:200],
                    background="indoor scene",
                    expression="neutral",
                    camera_angle="medium shot",
                    mood="natural",
                    negative_prompt="blurry, distorted, deformed"
                ))
        
        char = Character(
            id="CHAR_001", name="Character", gender="person",
            appearance={"hair": "short hair", "build": "average", "skin_tone": "medium"},
            outfit="casual clothing",
            signature_feature="casual appearance",
            global_prompt="a person with short hair wearing casual clothing"
        )
        
        story = StoryData(story_id=story_id, global_style="clean illustration", characters=[char], frames=frames)
        story.assemble_all_prompts()
        return story

    def cleanup(self):
        """Release resources."""
        if self.generator:
            self.generator.clear_cache()
        torch.cuda.empty_cache()


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Process multiple stories in batch."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.pipeline = WhiteBoxStoryPipeline(self.config)
        
    def process_directory(self, input_dir: str, pattern: str = "*.txt") -> Dict[str, bool]:
        """Process all TXT files in a directory."""
        input_path = Path(input_dir)
        files = sorted(input_path.glob(pattern))
        
        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(files)} files")
        print(f"{'='*60}")
        
        results = {}
        self.pipeline.initialize()
        
        for file_path in files:
            story_id = file_path.stem
            print(f"\n{'='*60}")
            print(f"Processing: {story_id}")
            print(f"{'='*60}")
            
            try:
                success = self.pipeline.run(str(file_path), story_id)
                results[story_id] = success
            except Exception as e:
                print(f"[Batch] FAILED: {e}")
                results[story_id] = False
            
            self.pipeline.cleanup()
            import gc
            gc.collect()
        
        success_count = sum(1 for v in results.values() if v)
        print(f"\n{'='*60}")
        print(f"Batch Complete: {success_count}/{len(files)} successful")
        print(f"{'='*60}")
        
        return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="White-Box Story Pipeline")
    parser.add_argument("input", help="Input TXT file or directory")
    parser.add_argument("--output", "-o", default="./results", help="Output directory")
    parser.add_argument("--model", "-m", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--llm", "-l", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--steps", "-s", type=int, default=24)
    parser.add_argument("--candidates", "-n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--batch", action="store_true")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        model_path=args.model, llm_path=args.llm, output_dir=args.output,
        num_inference_steps=args.steps, base_seed=args.seed,
        n_candidates=args.candidates, use_evaluator=not args.no_eval
    )
    
    if args.batch:
        processor = BatchProcessor(config)
        results = processor.process_directory(args.input)
        with open(os.path.join(args.output, "batch_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        pipeline = WhiteBoxStoryPipeline(config)
        success = pipeline.run(args.input)
        if not success:
            print("\n[Pipeline] FAILED")
            exit(1)


if __name__ == "__main__":
    main()
