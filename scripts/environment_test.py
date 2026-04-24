#!/usr/bin/env python3
"""
Environment Test Script for White-Box Story Pipeline

Tests that all required packages and models are available.

Usage:
    python scripts/environment_test.py
"""

import sys
import os

print("=" * 60)
print("White-Box Story Pipeline - Environment Test")
print("=" * 60)

# Track errors
errors = []
warnings = []

# 1. Python version
print("\n[1] Python Version")
py_version = sys.version_info
print(f"    Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version < (3, 9):
    errors.append(f"Python 3.9+ required, got {py_version.major}.{py_version.minor}")

# 2. PyTorch
print("\n[2] PyTorch")
try:
    import torch
    print(f"    PyTorch: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        warnings.append("CUDA not available - will use CPU (slow)")
except ImportError as e:
    errors.append(f"PyTorch not installed: {e}")

# 3. Diffusers
print("\n[3] Diffusers")
try:
    import diffusers
    print(f"    Diffusers: {diffusers.__version__}")
except ImportError as e:
    errors.append(f"Diffusers not installed: {e}")

# 4. Transformers
print("\n[4] Transformers")
try:
    import transformers
    print(f"    Transformers: {transformers.__version__}")
except ImportError as e:
    errors.append(f"Transformers not installed: {e}")

# 5. CLIP
print("\n[5] CLIP")
try:
    import clip
    print(f"    CLIP: Available")
except ImportError as e:
    warnings.append(f"CLIP not installed: {e}")
    print(f"    CLIP: NOT AVAILABLE (evaluation will use fallback)")

# 6. Other dependencies
print("\n[6] Other Dependencies")
deps = ["numpy", "PIL", "torchvision", "accelerate", "tqdm"]
for dep in deps:
    try:
        __import__(dep)
        print(f"    {dep}: OK")
    except ImportError:
        errors.append(f"{dep} not installed")
        print(f"    {dep}: MISSING")

# 7. Test SDXL model loading (without actual inference)
print("\n[7] SDXL Model Check")
model_path = "stabilityai/stable-diffusion-xl-base-1.0"
cache_dir = "./models"
print(f"    Model path: {model_path}")
print(f"    Cache dir: {cache_dir}")

# Check if model files exist
sdxl_components = [
    "vae/config.json",
    "unet/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json"
]

model_exists = False
for component in sdxl_components:
    full_path = os.path.join(cache_dir, model_path.replace("/", "--"), component)
    if os.path.exists(full_path):
        model_exists = True
        break

if model_exists:
    print(f"    SDXL model files: FOUND")
else:
    warnings.append("SDXL model not downloaded - will download on first run")
    print(f"    SDXL model files: NOT FOUND (will download)")

# 8. Import our modules
print("\n[8] Custom Modules")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src import (
        LLMProcessor,
        WhiteBoxSDXLGenerator,
        StoryEvaluator,
        WhiteBoxStoryPipeline,
    )
    print(f"    llm_processor: OK")
    print(f"    sdxl_generator: OK")
    print(f"    evaluator: OK")
    print(f"    pipeline_runner: OK")
except ImportError as e:
    errors.append(f"Custom module import failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Environment Test Summary")
print("=" * 60)

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")

if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")

if not errors and not warnings:
    print("\n  All checks passed!")
    print("\n  Ready to run:")
    print("    python run_pipeline.py data/TaskA/01.txt")
elif not errors:
    print("\n  Ready to run (with warnings)")
    print("  Some features may be limited")
else:
    print("\n  Please fix errors before running")

print("\n" + "=" * 60)
