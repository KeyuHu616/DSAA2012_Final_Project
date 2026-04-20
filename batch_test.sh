#!/bin/bash

# 启用扩展通配符（用于排除文件）
shopt -s extglob

# 设置默认 GPU 设备号
DEFAULT_CUDA_DEVICE="2"

# 允许通过命令行参数覆盖，例如：./script.sh --gpu 0
# 简单参数解析：如果第一个参数是 --gpu，则取第二个参数作为设备号
if [[ "$1" == "--gpu" && -n "$2" ]]; then
    CUDA_DEVICE="$2"
    shift 2   # 把已处理的参数移走
else
    CUDA_DEVICE="$DEFAULT_CUDA_DEVICE"
fi

# 导出到环境变量
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 指定输入目录
INPUT_DIR="data/TaskA"

# 遍历目录下所有非 extra_ 开头的 .txt 文件
found=0
for file in "$INPUT_DIR"/!(extra_*.txt); do
    if [[ -f "$file" && "$file" == *.txt ]]; then
        ((found++))
        echo "Processing: $file with GPU $CUDA_VISIBLE_DEVICES"
        python src/pipeline_runner.py --input "$file"
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