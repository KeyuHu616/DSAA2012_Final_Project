#!/bin/bash

# File: scripts/download_models_incremental.sh
# Description: Incremental download script that skips existing files.

echo "🚀 Starting incremental download of Project 2 model weights..."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "📁 Project Root: $ROOT_DIR"
cd "$ROOT_DIR/models" || exit 1

# 错误计数器（用于后台任务）
ERRORS=0
ERROR_LOCK=$(mktemp -u)

# 线程安全的错误计数增加函数
increment_error() {
    # 简单的文件锁机制防止竞态条件
    (
        flock -x 200
        ERRORS=$((ERRORS + 1))
    ) 200>"$ERROR_LOCK"
}

# ==========================================
# Helper: Parallel Download (only if file doesn't exist)
# ==========================================
download_async_if_missing() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "⏭️  [SKIP] $output (already exists)"
        return 0 # 文件已存在，不计入错误
    fi
    echo "🔽 [ASYNC] $output"
    # -x 8 足够了，因为多个文件同时多线程，总数会很大
    aria2c -x 8 -s 8 -c --quiet "$url" -o "$output"
    # 检查返回值
    if [ $? -ne 0 ]; then
        echo "❌ Failed: $output"
        increment_error
    fi
}

# ==========================================
# 1. Start SDXL Download (Background)
# ==========================================
echo "📦 [1/3] Queuing SDXL..."
SDXL_DIR="sdxl"
mkdir -p "$SDXL_DIR"
(
    cd "$SDXL_DIR" || exit 1
    # Config
    if [ ! -f "model_index.json" ]; then
        echo "🔽 [CONFIG] model_index.json"
        aria2c -x 4 -c --quiet "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json" -o "model_index.json" || increment_error
    fi
    # Checkpoint
    download_async_if_missing "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" "sd_xl_base_1.0.safetensors"
) &

# ==========================================
# 2. Fixed IP-Adapter Download (SDXL Only)
# ==========================================
##TODO

# ==========================================
# 3. Improved LLM Download (串行小文件 + 并行大文件) with Skip Logic
# ==========================================
echo "📦 [3/3] Downloading LLM (Qwen2.5 7B)..."
LLM_DIR="llm/Qwen2.5-7B-Instruct"
mkdir -p "$LLM_DIR"
cd "$LLM_DIR" || exit 1

BASE_URL="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main"

# --- Key Improvement: Check for critical config files first ---
CRITICAL_CONFIGS=("config.json" "tokenizer_config.json")
MISSING_CONFIGS=()
for conf in "${CRITICAL_CONFIGS[@]}"; do
    if [ ! -f "$conf" ]; then
        MISSING_CONFIGS+=("$conf")
    fi
done

if [ ${#MISSING_CONFIGS[@]} -gt 0 ]; then
    echo "📝 [NEED CONFIG] Downloading missing config/tokenizer files: ${MISSING_CONFIGS[*]}"
    # Define all required small files
    SMALL_FILES=(
        "config.json"
        "generation_config.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
        "model.safetensors.index.json"
    )

    # Iterate and download only missing ones
    for file in "${SMALL_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "🔽 [SERIAL] $file"
            aria2c -x 4 -c --quiet "$BASE_URL/$file" -o "$file"
            if [ $? -ne 0 ]; then
                echo "❌ Failed to download small file: $file"
                increment_error
            fi
        else
            echo "⏭️  [SKIP] $file (already exists)"
        fi
    done
else
    echo "⏭️  [SKIP] All critical config files already exist, proceeding to shards."
fi

# --- Check for model shards ---
SHARDS=(
    "model-00001-of-00004.safetensors"
    "model-00002-of-00004.safetensors"
    "model-00003-of-00004.safetensors"
    "model-00004-of-00004.safetensors"
)

MISSING_SHARDS=()
for shard in "${SHARDS[@]}"; do
    if [ ! -f "$shard" ]; then
        MISSING_SHARDS+=("$shard")
    fi
done

if [ ${#MISSING_SHARDS[@]} -gt 0 ]; then
    echo "🚀 [NEED SHARDS] Downloading missing model shards: ${MISSING_SHARDS[*]}"
    # Download remaining shards in parallel
    for shard in "${SHARDS[@]}"; do
        download_async_if_missing "${BASE_URL}/${shard}" "$shard"
    done
else
    echo "⏭️  [SKIP] All model shards already exist."
fi

# ==========================================
# 4. Wait for all tasks and finalize
# ==========================================
echo "⏳ Waiting for all downloads to complete... (This may take a while)"
wait

# 清理锁文件
rm -f "$ERROR_LOCK"

# 检查是否有错误
if [ $ERRORS -gt 0 ]; then
    echo "❌ CRITICAL: Download completed with $ERRORS errors. Please check logs."
    exit 1
else
    echo "✅ ALL downloads completed successfully or skipped (if already present)! Enjoy your models."
fi