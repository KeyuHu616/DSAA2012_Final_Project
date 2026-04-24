#!/bin/bash
# ============================================================================
# Model Download Script for White-Box Story Pipeline
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODEL_DIR="./models"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo "Model Download Script"
echo "============================================"

download_model() {
    local model_name="$1"
    local cache_dir="$MODEL_DIR"
    echo "Downloading: $model_name"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$model_name" --cache-dir "$cache_dir" --local-dir-use-symlinks False
    else
        echo "WARNING: huggingface-cli not found"
        echo "Please install: pip install huggingface_hub"
    fi
}

# SDXL Base 1.0
echo ""
echo "[1/2] SDXL Base 1.0"
download_model "stabilityai/stable-diffusion-xl-base-1.0"

# CLIP models
echo ""
echo "[2/2] CLIP ViT-B/32"
download_model "openai/clip-vit-base-patch32"

echo ""
echo "============================================"
echo "Download complete!"
echo "DINOv2 will be downloaded via torch.hub"
echo "============================================"
