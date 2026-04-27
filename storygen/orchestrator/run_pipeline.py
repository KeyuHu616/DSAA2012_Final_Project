"""
Pipeline Orchestrator - Main Execution Entry Point
Coordinates the complete story generation workflow
"""

import argparse
import sys
import json
import time
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Narrative Weaver Pro - SOTA Story Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input settings
    parser.add_argument(
        '--script_dir', type=str, default='../data/TaskA',
        help='Directory containing story scripts'
    )
    parser.add_argument('--single', type=str, help='Process only specified sample ID')
    parser.add_argument('--batch', action='store_true', help='Process all samples')

    # LLM settings
    parser.add_argument('--llm_model', type=str, default='local:llama3:70b',
        help='LLM model (format: backend:model_name)')

    # Generation settings
    parser.add_argument('--consistency_mode', type=str, default='storydiffusion',
        choices=['storydiffusion', 'redistory', 'hybrid'])
    parser.add_argument('--consistency_strength', type=float, default=0.6)
    parser.add_argument('--ip_adapter_scale', type=float, default=0.6)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=None)

    # Output settings
    parser.add_argument('--output_dir', type=str, default='outputs/test_results')
    parser.add_argument('--save_intermediate', action='store_true')

    # Evaluation
    parser.add_argument('--enable_eval', action='store_true')
    parser.add_argument('--eval_only', action='store_true')

    # Other
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry_run', action='store_true')

    return parser.parse_args()


def discover_scripts(script_dir: str, single: str = None) -> List[Path]:
    """Discover script files to process"""
    dir_path = Path(script_dir)

    if single:
        target = dir_path / f"{single}.txt"
        if target.exists():
            return [target]
        for ext in ['.txt', '.md']:
            alt = dir_path / f"{single}{ext}"
            if alt.exists():
                return [alt]
        raise FileNotFoundError(f"Script not found: {target}")

    scripts = sorted(dir_path.glob("*.txt"))
    if not scripts:
        raise ValueError(f"No .txt files found in {script_dir}")
    return scripts


def build_pipeline_config(args) -> Dict[str, Any]:
    """Build pipeline configuration from arguments"""
    return {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "device": "cuda",
        "use_fp16": True,
        "consistency_mode": args.consistency_mode,
        "consistency_strength": args.consistency_strength,
        "memory_bank_size": 4,
        "memory_bank_capacity": 5,
        "memory_decay_factor": 0.9,
        "ip_adapter_scale": args.ip_adapter_scale,
        "generation_params": {
            "num_steps": args.steps,
            "guidance_scale": args.guidance_scale,
        },
        "height": 768,
        "width": 768,
        "enable_model_cpu_offload": True,
    }


def run_pipeline():
    """Main execution function"""
    import argparse
    from storygen.script_director.llm_parser import LLMScriptParser
    from storygen.script_director.prompt_enhancer import PromptEnhancer
    from storygen.core_generator.pipeline import NarrativeGenerationPipeline

    args = parse_args()

    print("""
    ============================================================
        Narrative Weaver Pro v1.0
        SOTA Story Generation Pipeline
    ============================================================
    """)

    # Parse LLM configuration
    if ':' in args.llm_model:
        llm_backend, llm_model = args.llm_model.split(':', 1)
    else:
        llm_backend, llm_model = 'local', args.llm_model

    # Discover scripts
    try:
        scripts = discover_scripts(args.script_dir, args.single)
        print(f"\nFound {len(scripts)} scripts to process")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize components
    print(f"\nInitializing components...")
    llm_parser = LLMScriptParser(llm_backend=llm_backend, model_name=llm_model)
    enhancer = PromptEnhancer()

    if not args.eval_only:
        config = build_pipeline_config(args)
        pipeline = NarrativeGenerationPipeline(config)
    else:
        pipeline = None

    # Process scripts
    all_results = []
    for i, script_path in enumerate(scripts, 1):
        print(f"\n{'━' * 60}")
        print(f"Progress: [{i}/{len(scripts)}]")
        print(f"{'━' * 60}")

        story_id = script_path.stem
        result = {"story_id": story_id, "status": "pending", "processing_time": 0}
        start_time = time.time()

        try:
            # LLM Parsing
            print(f"[Step 1] LLM Script Parsing...")
            board = llm_parser.process_script_file(str(script_path))

            # Prompt Enhancement
            print(f"[Step 2] Prompt Enhancement...")
            enhanced_prompts = enhancer.process_entire_story(board)

            if args.dry_run:
                result["status"] = "dry_run_completed"
                continue

            # Image Generation
            if pipeline:
                print(f"[Step 3] Image Generation...")
                images, _ = pipeline.generate_story(board, seed=args.seed)

                # Save Results
                print(f"[Step 4] Saving Output...")
                pipeline.save_story_images(
                    images=images,
                    story_id=story_id,
                    panels=board.panels,
                    output_dir=args.output_dir
                )

            result["status"] = "success"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            if args.debug:
                traceback.print_exc()

        result["processing_time"] = time.time() - start_time
        all_results.append(result)

    # Summary
    success_count = sum(1 for r in all_results if r["status"] == "success")
    print(f"\n{'=' * 60}")
    print(f"FINAL: {success_count}/{len(all_results)} successful")
    print(f"{'=' * 60}")

    # Save report
    report_path = Path(args.output_dir) / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    run_pipeline()
