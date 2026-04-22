#!/bin/bash

# Enable extended globbing (for excluding files)
shopt -s extglob

# Default GPU device (GPU 0 for testing)
DEFAULT_CUDA_DEVICE="0"

# Allow command line override, e.g.: ./batch_test.sh --gpu 0
# Simple arg parsing: if first arg is --gpu, second arg is device number
if [[ "$1" == "--gpu" && -n "$2" ]]; then
    CUDA_DEVICE="$2"
    shift 2
else
    CUDA_DEVICE="$DEFAULT_CUDA_DEVICE"
fi

# Export to environment variable
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Input directory
INPUT_DIR="data/TaskA"

# Iterate through all .txt files excluding extra_ prefix
found=0
for file in "$INPUT_DIR"/!(extra_*.txt); do
    if [[ -f "$file" && "$file" == *.txt ]]; then
        ((found++))
        echo "Processing: $file with GPU $CUDA_VISIBLE_DEVICES"
        python src/pipeline_runner.py --input "$file" --save_json
        if [[ $? -ne 0 ]]; then
            echo "Error processing $file, exiting..."
            exit 1
        fi
    fi
done

if [[ "$found" -eq 0 ]]; then
    echo "No matching .txt files found in $INPUT_DIR"
    exit 1
fi
