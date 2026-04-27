#!/usr/bin/env python3
"""Test basic SDXL generation without custom attention processor"""
import sys
sys.path.insert(0, '.')

from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from storygen.utils.mirror_config import get_models_cache_dir
import os

cache_dir = get_models_cache_dir()

print("Loading SDXL pipeline on GPU 2...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
    local_files_only=True,
    cache_dir=str(cache_dir),
).to('cuda')

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
print("Pipeline loaded!")

# Test 1: Basic 1024x1024 generation
print("\nTest 1: Generating 1024x1024 image...")
result = pipe(
    prompt='a person sitting in a park, cinematic, photorealistic',
    num_inference_steps=20,
    height=1024,
    width=1024,
)
result.images[0].save('outputs/simple_test_1024.png')
print(f"SUCCESS: {result.images[0].size}")

# Test 2: Generate 3 frames with different prompts
print("\nTest 2: Generating 3 frames...")
prompts = [
    'a person sitting in a park, cinematic photorealistic',
    'a person drinking coffee in a cafe, cinematic photorealistic',
    'a person looking at an art exhibition, cinematic photorealistic',
]

images = []
for i, prompt in enumerate(prompts):
    print(f"  Frame {i+1}...")
    result = pipe(
        prompt=prompt,
        num_inference_steps=20,
        height=1024,
        width=1024,
    )
    result.images[0].save(f'outputs/simple_test_frame_{i+1}.png')
    images.append(result.images[0])
    print(f"  Done!")

# Create storyboard
from storygen.utils.image_utils import create_storyboard
board = create_storyboard(
    images,
    [f"Scene {i+1}" for i in range(len(images))],
    image_size=(512, 512)
)
board.save('outputs/simple_test_storyboard.png')
print("\nStoryboard saved!")
print("ALL TESTS PASSED!")
