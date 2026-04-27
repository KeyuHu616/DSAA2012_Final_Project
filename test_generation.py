"""
Full pipeline test with image generation
"""

import sys
sys.path.insert(0, '.')

import torch
from storygen.script_director.llm_parser import LLMScriptParser
from storygen.script_director.prompt_enhancer import PromptEnhancer
from storygen.core_generator.pipeline import NarrativeGenerationPipeline
from storygen.utils.image_utils import create_storyboard

# Force rule-based parsing by using a mock LLM
class MockLLMScriptParser(LLMScriptParser):
    """Parser that always uses rule-based parsing"""
    def call_llm_for_analysis(self, parsed_script):
        print("[MockParser] Using rule-based parsing (LLM unavailable)")
        return self._rule_based_parse(parsed_script)

# Configuration
config = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_fp16": True,
    "consistency_mode": "storydiffusion",
    "consistency_strength": 0.6,
    "memory_bank_size": 4,
    "memory_bank_capacity": 5,
    "memory_decay_factor": 0.9,
    "ip_adapter_scale": 0.6,
    "generation_params": {
        "num_steps": 20,  # Reduced for testing
        "guidance_scale": 7.5,
    },
    "height": 512,  # Smaller for testing
    "width": 512,
    "enable_model_cpu_offload": True,
}

print("=" * 60)
print("Narrative Weaver Pro - Full Generation Test")
print("=" * 60)
print(f"\nDevice: {config['device']}")
print(f"Image Size: {config['height']}x{config['width']}")

# Initialize parser and enhancer
parser = MockLLMScriptParser(llm_backend='local', model_name='llama3:70b')
enhancer = PromptEnhancer()

# Initialize pipeline
print("\nInitializing pipeline...")
pipeline = NarrativeGenerationPipeline(config)

# Parse script
test_file = 'data/TaskA/06.txt'
print(f"\nParsing script: {test_file}")
board = parser.process_script_file(test_file)

print(f"\nProduction Board:")
print(f"  Story ID: {board.story_id}")
print(f"  Characters: {list(board.characters.keys())}")
print(f"  Panels: {len(board.panels)}")
print(f"  Style: {board.global_style}")

# Enhance prompts
print("\nEnhancing prompts...")
enhanced_prompts = enhancer.process_entire_story(board)

# Generate story
print("\n" + "=" * 60)
print("Starting Image Generation")
print("=" * 60)

try:
    images, portraits = pipeline.generate_story(
        board,
        seed=42,
        return_portraits=True
    )

    print(f"\nGenerated {len(images)} images successfully!")

    # Create storyboard
    print("\nCreating storyboard...")
    labels = [f"Scene {i+1}" for i in range(len(images))]
    storyboard = create_storyboard(images, labels, image_size=(512, 512))

    # Save outputs
    output_dir = "outputs/test_generation"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    for i, img in enumerate(images, 1):
        img.save(f"{output_dir}/frame_{i:02d}.png")

    # Save storyboard
    storyboard.save(f"{output_dir}/storyboard.png")

    print(f"\nOutputs saved to: {output_dir}/")
    print("  - frame_01.png")
    print("  - frame_02.png")
    print("  - frame_03.png")
    print("  - storyboard.png")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

except Exception as e:
    print(f"\nGeneration error: {e}")
    import traceback
    traceback.print_exc()
