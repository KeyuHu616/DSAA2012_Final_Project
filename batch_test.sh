#!/bin/bash
# ============================================================================
# White-Box Story Pipeline - Batch Test Script
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
INPUT_DIR="${1:-./data/TaskA}"
OUTPUT_DIR="${2:-./results}"
MODEL_PATH="${3:-stabilityai/stable-diffusion-xl-base-1.0}"

echo "============================================"
echo "White-Box Story Pipeline - Batch Test"
echo "============================================"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Model Path: $MODEL_PATH"
echo "============================================"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "PyTorch not installed"; exit 1; }
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" || { echo "Diffusers not installed"; exit 1; }

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run batch processing
echo ""
echo "Starting batch processing..."
echo "============================================"

python -u run_pipeline.py "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --steps 24 \
    --candidates 2 \
    --seed 42 \
    --batch

echo ""
echo "============================================"
echo "Batch processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"

# Show summary if available
if [ -f "$OUTPUT_DIR/batch_results.json" ]; then
    echo ""
    echo "Summary:"
    cat "$OUTPUT_DIR/batch_results.json"
fi
