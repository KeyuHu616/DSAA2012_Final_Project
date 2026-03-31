#!/bin/bash

# scripts/verify_models.sh
# 用途: 在 Docker 构建后或运行前检查模型是否存在

MODEL_DIR="./models"

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 缺少关键文件 $1"
        exit 1
    fi
}

cd $MODEL_DIR

# 检查 SDXL 是否有 config.json 或 diffusion_pytorch_model.bin
if [ -d "sdxl-base-1.0" ]; then
    check_file "sdxl-base-1.0/config.json"
    echo "✅ SDXL 模型文件结构正常"
else
    echo "❌ 错误: 未找到 SDXL 模型目录"
    exit 1
fi

# 检查 IP-Adapter
if [ -d "ip-adapter" ]; then
    if ls ip-adapter/*.safetensors 1> /dev/null 2>&1; then
        echo "✅ IP-Adapter 权重文件 (.safetensors) 找到"
    else
        echo "❌ 错误: IP-Adapter 目录下没有 .safetensors 文件"
        exit 1
    fi
fi

echo "🎉 所有模型文件完整性检查通过！"