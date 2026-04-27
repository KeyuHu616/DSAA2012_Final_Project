#!/usr/bin/env python3
"""
TaskA Full Test - Batch evaluation on all TaskA data files
Tests CLIP alignment (panel vs text) and LPIPS consistency (panel to panel)
"""
import sys
import os
sys.path.insert(0, '.')

from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

import torch
from PIL import Image
from pathlib import Path
import json
import time
import gc
from datetime import datetime

from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator
from storygen.utils.image_utils import create_storyboard


def test_single_file(script_file: str, output_dir: str, device: str = "cuda:5") -> dict:
    """Test a single TaskA file and return metrics"""
    print(f"\n{'='*60}")
    print(f"Testing: {script_file}")
    print(f"{'='*60}")

    script_path = Path(script_file)
    output_path = Path(output_dir) / script_path.stem
    os.makedirs(output_path, exist_ok=True)

    # Step 1: Parse script
    start = time.time()
    parser = create_qwen_parser()

    with parser:
        board = parser.process_script_file(script_file)
        parser.save_production_board(board, str(output_path / "production_board.json"))

    parse_time = time.time() - start
    del parser
    torch.cuda.empty_cache()

    print(f"  Parsed: {len(board.panels)} panels, {len(board.characters)} characters")

    # Step 2: Setup pipeline
    config = {
        "device": device,
        "use_fp16": True,
        "consistency_strength": 0.0,
        "memory_bank_size": 4,
        "generation_params": {"num_steps": 40, "guidance_scale": 7.5},  # Higher CFG for better prompt adherence
        "height": 1024,
        "width": 1024,
        "enable_model_cpu_offload": True,
    }

    pipeline = NarrativeGenerationPipeline(config)

    # Step 3: Generate
    gen_start = time.time()
    images, _ = pipeline.generate_story(board, seed=42)
    gen_time = time.time() - gen_start
    print(f"  Generated {len(images)} frames in {gen_time:.1f}s")

    # Save images
    for i, img in enumerate(images, 1):
        img.save(output_path / f"frame_{i:02d}.png")

    # Create storyboard
    storyboard = create_storyboard(
        images,
        [f"Scene {i+1}: {p.shot_type}" for i, p in enumerate(board.panels)],
        image_size=(512, 512)
    )
    storyboard.save(output_path / "storyboard.png")

    # Step 4: Evaluate
    clip_eval = CLIPEvaluator(device=device)
    clip_scores = []
    for i, panel in enumerate(board.panels):
        prompt = panel.enhanced_prompt or panel.raw_prompt
        score = clip_eval.compute_similarity([images[i]], [prompt])[0]
        clip_scores.append(score)
        print(f"  Frame {i+1} CLIP: {score:.3f} | {prompt[:50]}...")
    
    del clip_eval
    torch.cuda.empty_cache()

    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0

    # LPIPS consistency
    if len(images) > 1:
        consistency_eval = ConsistencyEvaluator(device=device, metric="lpips")
        lpips_scores = []
        for i in range(len(images) - 1):
            dist = consistency_eval.compute_lpips_similarity(images[i], images[i + 1])
            lpips_scores.append(1 - dist)
        avg_consistency = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 1.0
        del consistency_eval
        torch.cuda.empty_cache()
    else:
        avg_consistency = 1.0

    overall = 0.6 * avg_clip + 0.4 * avg_consistency

    metrics = {
        "script": str(script_file),
        "num_panels": len(board.panels),
        "num_characters": len(board.characters),
        "clip_scores": [float(s) for s in clip_scores],
        "avg_clip_score": float(avg_clip),
        "avg_consistency": float(avg_consistency),
        "overall_score": float(overall),
        "parse_time_s": round(parse_time, 1),
        "generation_time_s": round(gen_time, 1),
    }

    # Save metrics
    with open(output_path / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Results: CLIP={avg_clip:.3f}, Consistency={avg_consistency:.3f}, Overall={overall:.3f}")

    # Cleanup
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


def run_taska_batch(output_dir: str = "outputs/taskA_batch"):
    """Run batch test on all TaskA files"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   TaskA Full Test - Batch Evaluation                       ║
║   CLIP Alignment + LPIPS Consistency                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Find all TaskA files
    taska_dir = Path("data/TaskA")
    taska_files = sorted(taska_dir.glob("*.txt"))
    print(f"Found {len(taska_files)} TaskA files")

    # Target thresholds
    CLIP_THRESHOLD = 0.30
    CONSISTENCY_THRESHOLD = 0.30

    all_results = []
    os.makedirs(output_dir, exist_ok=True)

    for i, script_file in enumerate(taska_files, 1):
        print(f"\n[{i}/{len(taska_files)}]")
        try:
            metrics = test_single_file(
                str(script_file),
                output_dir=output_dir,
                device="cuda:5"
            )
            all_results.append(metrics)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "script": str(script_file),
                "error": str(e),
                "overall_score": 0.0
            })

    # Summary
    print("\n" + "="*70)
    print("TASKA BATCH TEST SUMMARY")
    print("="*70)

    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        avg_clip = sum(r["avg_clip_score"] for r in valid_results) / len(valid_results)
        avg_consistency = sum(r["avg_consistency"] for r in valid_results) / len(valid_results)
        avg_overall = sum(r["overall_score"] for r in valid_results) / len(valid_results)

        print(f"\nTotal: {len(taska_files)}, Successful: {len(valid_results)}, Failed: {len(all_results) - len(valid_results)}")
        print(f"\n  Average CLIP:     {avg_clip:.3f} (threshold: {CLIP_THRESHOLD})")
        print(f"  Average Consist:  {avg_consistency:.3f} (threshold: {CONSISTENCY_THRESHOLD})")
        print(f"  Average Overall: {avg_overall:.3f}")

        for r in valid_results:
            status = "PASS" if r["overall_score"] >= 0.3 else "FAIL"
            print(f"  [{status}] {Path(r['script']).name}: CLIP={r['avg_clip_score']:.3f}, C={r['avg_consistency']:.3f}")
    else:
        print("  All files failed!")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(taska_files),
        "successful": len(valid_results),
        "results": all_results
    }
    with open(Path(output_dir) / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_taska_batch()
