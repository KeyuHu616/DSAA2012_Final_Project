#!/usr/bin/env python3
"""
Quick Test - 从3 panels文本快速生成故事并评估
用法: python quick_test.py data/TaskA/06.txt
"""

import sys
import os
sys.path.insert(0, '.')

# Setup HF mirror and cache directories
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

import torch
from PIL import Image
from pathlib import Path
import json
import time

from storygen.script_director.llm_parser_local import create_qwen_parser
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.evaluation_hub.metric_consistency import ConsistencyEvaluator
from storygen.evaluation_hub.metric_clip import CLIPEvaluator
from storygen.utils.image_utils import create_storyboard


def quick_test(script_file: str, output_dir: str = "outputs/quick_test"):
    """快速测试：解析 → 生成 → 评估"""
    print("=" * 60)
    print("Quick Test: 3-Panel Story Generation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)

    # Step 1: 解析脚本
    print("\n[Step 1/4] Parsing script...")
    start = time.time()
    parser = create_qwen_parser()

    with parser:
        board = parser.process_script_file(script_file)
        parser.save_production_board(board, str(output_path / "production_board.json"))

    parse_time = time.time() - start
    print(f"  ✅ Parsed: {len(board.panels)} panels, {len(board.characters)} characters")
    print(f"  ⏱️  Parse time: {parse_time:.1f}s")

    # 释放 Qwen 内存
    del parser
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 2: 配置生成管道
    print("\n[Step 2/4] Setting up generation pipeline...")
    config = {
        "device": "cuda",  # Use GPU 2 (isolated via CUDA_VISIBLE_DEVICES in shell)
        "use_fp16": True,
        "consistency_strength": 0.0,  # Disabled - needs proper dimension handling for UNet attention
        "memory_bank_size": 4,
        "generation_params": {"num_steps": 40, "guidance_scale": 7.5},
        "height": 1024,
        "width": 1024,
        "enable_model_cpu_offload": True,
    }

    pipeline = NarrativeGenerationPipeline(config)

    # Step 3: 生成图像
    print("\n[Step 3/4] Generating images...")
    gen_start = time.time()

    images, _ = pipeline.generate_story(board, seed=42)

    gen_time = time.time() - gen_start
    print(f"  ✅ Generated {len(images)} frames")
    print(f"  ⏱️  Generation time: {gen_time:.1f}s")

    # 保存图像
    for i, img in enumerate(images, 1):
        img.save(output_path / f"frame_{i:02d}.png")

    # Step 4: 评估
    print("\n[Step 4/4] Evaluating...")

    device = config["device"]

    # CLIP 对齐评估 - 使用 production_board 中的 enhanced_prompts
    clip_eval = CLIPEvaluator(device=device)
    clip_scores = []
    for i, panel in enumerate(board.panels):
        # Use enhanced_prompt from production board for evaluation
        eval_prompt = panel.enhanced_prompt if panel.enhanced_prompt else panel.raw_prompt
        score = clip_eval.compute_similarity([images[i]], [eval_prompt])[0]
        clip_scores.append(score)
        print(f"  Frame {i+1} CLIP score: {score:.3f}")
        print(f"  Eval prompt: {eval_prompt[:100]}...")

    avg_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0

    # 一致性评估
    if len(images) > 1:
        consistency_eval = ConsistencyEvaluator(device="cuda", metric="lpips")
        lpips_scores = []
        for i in range(len(images) - 1):
            dist = consistency_eval.compute_lpips_similarity(images[i], images[i + 1])
            lpips_scores.append(1 - dist)  # 转换为相似度
            print(f"  Frame {i+1}-{i+2} LPIPS similarity: {lpips_scores[-1]:.3f}")

        avg_consistency = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 1.0
    else:
        avg_consistency = 1.0

    # 创建故事板
    storyboard = create_storyboard(
        images,
        [f"Scene {i+1}: {p.shot_type}" for i, p in enumerate(board.panels)],
        image_size=(384, 384)
    )
    storyboard.save(output_path / "storyboard.png")

    # 保存评估结果
    metrics = {
        "script": script_file,
        "num_panels": len(board.panels),
        "num_characters": len(board.characters),
        "global_style": board.global_style,
        "clip_scores": [float(s) for s in clip_scores],
        "avg_clip_score": float(avg_clip),
        "avg_consistency": float(avg_consistency),
        "overall_score": float(0.6 * avg_clip + 0.4 * avg_consistency),
        "timing": {
            "parse_time_s": round(parse_time, 1),
            "generation_time_s": round(gen_time, 1),
            "total_time_s": round(parse_time + gen_time, 1)
        }
    }

    with open(output_path / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  CLIP Alignment:  {avg_clip:.3f}")
    print(f"  Consistency:     {avg_consistency:.3f}")
    print(f"  Overall Score:  {metrics['overall_score']:.3f}")
    print(f"  Total Time:      {metrics['timing']['total_time_s']:.1f}s")
    print("=" * 60)
    print(f"  Output: {output_path}/")

    return metrics


if __name__ == "__main__":
    # 默认测试文件
    script_file = sys.argv[1] if len(sys.argv) > 1 else "data/TaskA/06.txt"

    if not Path(script_file).exists():
        print(f"❌ File not found: {script_file}")
        sys.exit(1)

    quick_test(script_file)
