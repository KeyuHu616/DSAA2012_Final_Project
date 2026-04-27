"""
Enhanced Story Generation Test with Auto-Optimization
- Uses China mirror for faster downloads
- Uses local Qwen2.5-7B for script parsing
- Automatic quality assessment
"""

import sys
import os
sys.path.insert(0, '.')

# Setup HF mirror and unified cache directories
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()
print("[Setup] Unified cache directory: ./models")

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import json
import time
import gc
from datetime import datetime

# Import pipeline components
from storygen.script_director.llm_parser_local import LocalQwenParser, create_qwen_parser
from storygen.script_director.prompt_enhancer import PromptEnhancer
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.utils.image_utils import create_storyboard


class GenerationQualityAssessor:
    """
    Assess generation quality using multiple metrics
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model = None

    def _load_clip(self):
        """Lazy load CLIP model for quality assessment"""
        if self.clip_model is None:
            try:
                import open_clip
                self.clip_model = open_clip.create_model('ViT-B-32', pretrained='openai').to(self.device)
                self.clip_preprocess = open_clip.transform.create_model_transform('ViT-B-32')
                print("[Assessor] Open-CLIP model loaded for quality assessment")
            except Exception as e:
                print(f"[Assessor] CLIP load failed: {e}")

    def assess_single_image(self, image: Image.Image, prompt: str) -> float:
        """
        Assess quality of a single image against its prompt

        Returns:
            Quality score (0-1)
        """
        if self.clip_model is None:
            self._load_clip()

        if self.clip_model is None:
            return 0.5  # Default score if CLIP unavailable

        try:
            import open_clip

            # Preprocess image
            img = image.convert('RGB')
            img_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            # Tokenize text
            text_input = open_clip.tokenize([prompt]).to(self.device)

            # Get features and similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(img_input)
                text_features = self.clip_model.encode_text(text_input)

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).mean()
                score = similarity.item()

            return score
        except Exception as e:
            print(f"[Assessor] Assessment error: {e}")
            return 0.5

    def assess_story_consistency(self, images: list) -> float:
        """
        Assess visual consistency across story frames

        Returns:
            Consistency score (0-1)
        """
        if len(images) < 2:
            return 1.0

        try:
            import lpips
            import torchvision.transforms as T

            # Load LPIPS model
            lpips_model = lpips.LPIPS(net='alex').to(self.device)
            lpips_model.eval()

            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

            consistency_scores = []

            for i in range(len(images) - 1):
                img1 = transform(images[i]).unsqueeze(0).to(self.device) * 2 - 1
                img2 = transform(images[i + 1]).unsqueeze(0).to(self.device) * 2 - 1

                with torch.no_grad():
                    dist = lpips_model(img1, img2)
                    # Lower distance = higher consistency
                    score = 1.0 - dist.item()
                    consistency_scores.append(score)

            return np.mean(consistency_scores) if consistency_scores else 0.5

        except ImportError:
            print("[Assessor] LPIPS not available, skipping consistency check")
            return 0.7  # Default score
        except Exception as e:
            print(f"[Assessor] Consistency check error: {e}")
            return 0.5

    def comprehensive_assessment(
        self,
        images: list,
        prompts: list
    ) -> dict:
        """
        Comprehensive quality assessment

        Returns:
            Dict with individual and overall scores
        """
        print("\n[Assessor] Running comprehensive quality assessment...")

        # Individual image quality
        quality_scores = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            score = self.assess_single_image(img, prompt)
            quality_scores.append(score)
            print(f"  Frame {i+1} quality: {score:.3f}")

        avg_quality = np.mean(quality_scores)

        # Consistency
        consistency = self.assess_story_consistency(images)
        print(f"  Story consistency: {consistency:.3f}")

        # Overall score (weighted)
        overall = 0.6 * avg_quality + 0.4 * consistency

        results = {
            "quality_scores": quality_scores,
            "avg_quality": avg_quality,
            "consistency": consistency,
            "overall_score": overall
        }

        print(f"  Overall score: {overall:.3f}")

        return results


class AutoOptimizedPipeline:
    """
    Pipeline with automatic optimization
    - Tries multiple parameter combinations
    - Selects best results
    - Iteratively improves
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.assessor = GenerationQualityAssessor(device)

    def _create_config_variations(self) -> list:
        """Generate parameter variations for optimization"""
        variations = [
            # Variation 1: Default high quality
            {
                "num_steps": 30,
                "guidance_scale": 7.5,
                "ip_adapter_scale": 0.7,
                "seed_offset": 0,
                "description": "High quality default"
            },
            # Variation 2: Higher consistency
            {
                "num_steps": 25,
                "guidance_scale": 8.5,
                "ip_adapter_scale": 0.85,
                "seed_offset": 100,
                "description": "High consistency mode"
            },
            # Variation 3: Artistic interpretation
            {
                "num_steps": 35,
                "guidance_scale": 6.5,
                "ip_adapter_scale": 0.6,
                "seed_offset": 200,
                "description": "Artistic interpretation"
            },
        ]
        return variations

    def generate_with_optimization(
        self,
        test_file: str,
        output_dir: str = "outputs/optimized_test",
        max_attempts: int = 3,
        use_local_qwen: bool = True
    ) -> dict:
        """
        Generate with automatic optimization

        Args:
            test_file: Input script file
            output_dir: Output directory
            max_attempts: Max optimization iterations
            use_local_qwen: Use local Qwen2.5-7B for parsing

        Returns:
            Best generation result
        """
        print("\n" + "=" * 70)
        print("AUTO-OPTIMIZED STORY GENERATION")
        print("=" * 70)

        os.makedirs(output_dir, exist_ok=True)

        # Initialize parser and enhancer
        enhancer = PromptEnhancer()

        # Parse script with Qwen
        print(f"\n[Step 1/4] Parsing script: {test_file}")

        if use_local_qwen:
            parser = create_qwen_parser()
        else:
            from storygen.script_director.llm_parser import LLMScriptParser
            parser = LLMScriptParser(llm_backend='local', model_name='llama3:70b')

        with parser:
            board = parser.process_script_file(test_file)
            board_path = Path(output_dir) / "production_board.json"
            parser.save_production_board(board, str(board_path))

        # Free Qwen memory before image generation
        del parser
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Characters: {list(board.characters.keys())}")
        print(f"  Panels: {len(board.panels)}")
        print(f"  Style: {board.global_style}")

        # Enhance prompts
        print("\n[Step 2/4] Enhancing prompts...")
        enhanced_prompts = enhancer.process_entire_story(board)
        prompts_list = [p["prompt"] for p in enhanced_prompts]

        # Get parameter variations
        variations = self._create_config_variations()

        best_result = None
        best_score = 0.0

        print(f"\n[Step 3/4] Testing {len(variations)} parameter variations...")

        for attempt, var in enumerate(variations[:max_attempts], 1):
            print(f"\n{'─' * 60}")
            print(f"Variation {attempt}/{min(len(variations), max_attempts)}: {var['description']}")
            print(f"  Steps: {var['num_steps']}, CFG: {var['guidance_scale']}, "
                  f"IP-Adapter: {var['ip_adapter_scale']}")

            # Create pipeline with variation
            var_config = self.config.copy()
            var_config["generation_params"] = {
                "num_steps": var["num_steps"],
                "guidance_scale": var["guidance_scale"],
            }
            var_config["ip_adapter_scale"] = var["ip_adapter_scale"]

            # Override base model path if local
            var_config["base_model"] = "stabilityai/stable-diffusion-xl-base-1.0"

            pipeline = NarrativeGenerationPipeline(var_config)

            try:
                # Generate with variation
                seed = self.config.get("base_seed", 42) + var["seed_offset"]
                images, _ = pipeline.generate_story(
                    board,
                    seed=seed,
                    return_portraits=False
                )

                # Save individual images
                var_dir = Path(output_dir) / f"attempt_{attempt}"
                var_dir.mkdir(parents=True, exist_ok=True)

                for i, img in enumerate(images, 1):
                    img.save(var_dir / f"frame_{i:02d}.png")

                # Assess quality
                metrics = self.assessor.comprehensive_assessment(images, prompts_list)
                metrics["variation"] = var
                metrics["images"] = [str(var_dir / f"frame_{i:02d}.png") for i in range(1, len(images)+1)]

                # Create storyboard
                storyboard = create_storyboard(
                    images,
                    [f"Scene {i+1}" for i in range(len(images))],
                    image_size=(512, 512)
                )
                storyboard.save(var_dir / "storyboard.png")

                print(f"\n  Variation {attempt} Results:")
                print(f"    Quality: {metrics['avg_quality']:.3f}")
                print(f"    Consistency: {metrics['consistency']:.3f}")
                print(f"    Overall: {metrics['overall_score']:.3f}")

                # Check if this is the best result
                if metrics["overall_score"] > best_score:
                    best_score = metrics["overall_score"]
                    best_result = metrics.copy()
                    best_result["attempt"] = attempt

            except Exception as e:
                print(f"\n  Error in variation {attempt}: {e}")
                import traceback
                traceback.print_exc()

        # Save best result
        print("\n[Step 4/4] Saving best results...")

        if best_result:
            # Copy best images to main output
            for i, img_path in enumerate(best_result["images"], 1):
                img = Image.open(img_path)
                img.save(Path(output_dir) / f"best_frame_{i:02d}.png")

            # Create best storyboard
            best_images = [Image.open(p) for p in best_result["images"]]
            best_storyboard = create_storyboard(
                best_images,
                [f"Scene {i+1}" for i in range(len(best_images))],
                image_size=(512, 512)
            )
            best_storyboard.save(Path(output_dir) / "best_storyboard.png")

            # Save metrics
            with open(Path(output_dir) / "best_metrics.json", "w") as f:
                # Convert numpy types to native Python
                result_for_json = {
                    "overall_score": float(best_result["overall_score"]),
                    "avg_quality": float(best_result["avg_quality"]),
                    "consistency": float(best_result["consistency"]),
                    "quality_scores": [float(s) for s in best_result["quality_scores"]],
                    "variation": best_result["variation"],
                    "attempt": best_result["attempt"],
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(result_for_json, f, indent=2)

            print(f"\n{'=' * 70}")
            print("✅ BEST RESULT FOUND")
            print(f"{'=' * 70}")
            print(f"  Overall Score: {best_result['overall_score']:.3f}")
            print(f"  Quality: {best_result['avg_quality']:.3f}")
            print(f"  Consistency: {best_result['consistency']:.3f}")
            print(f"  Variation: {best_result['variation']['description']}")
            print(f"  Output: {output_dir}/")
            print(f"{'=' * 70}")

        else:
            print("\n❌ All variations failed!")

        return best_result


def main():
    """Main execution"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🎬 Narrative Weaver Pro - Auto-Optimized Generator      ║
║   With China Mirror & Quality Assessment                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Configuration
    config = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_fp16": True,
        "consistency_mode": "storydiffusion",
        "consistency_strength": 0.7,  # Higher for better consistency
        "memory_bank_size": 4,
        "memory_bank_capacity": 5,
        "memory_decay_factor": 0.9,
        "ip_adapter_scale": 0.7,
        "generation_params": {
            "num_steps": 25,
            "guidance_scale": 7.5,
        },
        "height": 1024,  # SDXL works best at 1024x1024
        "width": 1024,
        "enable_model_cpu_offload": True,
        "base_seed": 42,
    }

    print(f"Device: {config['device']}")
    print(f"Image Size: {config['height']}x{config['width']}")

    # Create optimized pipeline
    optimizer = AutoOptimizedPipeline(config, device=config["device"])

    # Test file
    test_file = 'data/TaskA/06.txt'
    output_dir = "outputs/optimized_test"

    # Run optimized generation
    start_time = time.time()
    result = optimizer.generate_with_optimization(
        test_file=test_file,
        output_dir=output_dir,
        max_attempts=3,
        use_local_qwen=True
    )

    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return result


if __name__ == "__main__":
    main()
