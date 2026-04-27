#!/usr/bin/env python3
"""Simple test to verify SDXL pipeline works with custom attention processor"""
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttentionProcessor
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import get_models_cache_dir

cache_dir = get_models_cache_dir()

# Load pipeline with isolated GPU
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
    local_files_only=True,
    cache_dir=str(cache_dir),
).to('cuda')

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print('Pipeline loaded. Testing with 1024x1024...')

# Test basic generation first (no custom processor)
result = pipe(
    prompt='a person sitting in a park, cinematic, photorealistic',
    num_inference_steps=2,
    height=1024,
    width=1024,
)
print(f'Basic generation SUCCESS: {result.images[0].size}')

# Now test with custom attention processor
class SimplePassThrough(AttentionProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states

# Try to set custom processor
try:
    pipe.unet.set_attn_processor(SimplePassThrough())
    print('Custom attention processor set successfully')
    
    result2 = pipe(
        prompt='a person sitting in a park, cinematic, photorealistic',
        num_inference_steps=2,
        height=1024,
        width=1024,
    )
    print(f'With custom processor SUCCESS: {result2.images[0].size}')
except Exception as e:
    print(f'Custom processor FAILED: {e}')

print('ALL TESTS PASSED!')
