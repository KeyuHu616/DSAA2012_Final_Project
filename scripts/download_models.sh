#!/bin/bash

# =============================================================================
# 🚀 项目: Task1-StoryGen - 模型下载脚本
# 角色: Zhenzhuo (扩散模型核心)
# 功能: 下载 SDXL 基础模型和 IP-Adapter 依赖
# 路径: 对应 Dockerfile 中的 /app/models 目录
# =============================================================================

echo "🚀 开始下载模型文件..."
echo "💡 提示: 如果下载中断，可以手动运行此脚本，或使用 aria2 进行多线程下载。"

# --- 配置区域 ---
# 设置 HuggingFace 国内镜像源 (关键! 解决中国网络慢的问题)
export HF_ENDPOINT="https://hf-mirror.com"

# 设置 HuggingFace Token (用于下载需要授权的模型，如 Llama, SDXL-Turbo 等)
# 如果你没有特定 Token，可以留空或注释掉 export 语句，公开模型不需要 Token
# export HUGGING_FACE_HUB_TOKEN=""

# 定义模型存储目录 (必须与 Dockerfile 中的 VOLUME 或 WORKDIR 一致)
# 根据文档，Keyu 的 Dockerfile 应该会 COPY 这个目录
MODEL_DIR="./models"

# --- 核心逻辑 ---
# 1. 创建目录
mkdir -p $MODEL_DIR
cd $MODEL_DIR

# 2. 定义下载函数 (使用 huggingface_hub 的命令行工具)
download_model() {
    local repo_id=$1
    local local_dir=$2
    echo "🔽 正在下载模型: $repo_id"
    echo "   保存路径: $local_dir"
    
    # 使用 hf_transfer (更快) 或 huggingface-cli
    # 如果报错请先: pip install huggingface_hub
    huggingface-cli download $repo_id --local-dir $local_dir --resume-download
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功: $repo_id"
    else
        echo "❌ 失败: $repo_id (请检查网络或 Token)"
        # exit 1
    fi
}

# --- 模型列表 ---
# 根据《详细计划.md》文档要求，我们需要以下模型：

# 1. SDXL 基础模型 (文档阶段 1.1)
download_model "stabilityai/stable-diffusion-xl-base-1.0" "sdxl-base-1.0"

# 2. IP-Adapter (文档阶段 2.1)
# 参考: https://huggingface.co/tencent-ailab/IP-Adapter
download_model "TencentARC/IP-Adapter" "ip-adapter"

# 3. (可选) IP-Adapter 需要的 ViT-G 模型 (通常包含在上面的库中，但单独列出以防万一)
# download_model "h94/IP-Adapter" "ip-adapter-files" 

# 4. (可选) ControlNet Canny (文档阶段 2.2)
download_model "lllyasviel/controlnet-canny-sdxl-1.0" "controlnet-canny"

# 5. (可选) LCM-LoRA (文档阶段 3.1)
download_model "latent-consistency/lcm-lora-sdxl" "lcm-lora-sdxl"

echo "🎉 所有模型下载任务完成！"
echo "📌 请确保这些文件在 Docker 构建上下文中。"