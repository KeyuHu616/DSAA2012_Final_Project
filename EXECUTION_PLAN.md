# DSAA2012 Final Project 2: 执行计划

## 1. 项目架构概览 (Pipeline Architecture)

本项目采用 **"编排-生成"双阶段架构**，核心在于利用 LLM 生成包含视觉锚点的 Prompt，并通过 IP-Adapter 实现跨帧一致性。

### 1.1 文件架构映射

```
.
├── environment.yml          # (已就绪) Conda 环境定义文件，锁定 PyTorch/CUDA 版本
├── README.md                # 项目说明与合规声明
├── scripts/                 # 脚本目录
│   ├── setup_environment.sh # 调用 conda env create -f environment.yml
│   ├── download_models.sh   # 下载 SDXL/IP-Adapter/LLM 权重
│   └── verify_models.sh     # 验证模型哈希值，确保完整性
├── data/                    # 输入数据目录
│   └── test_0/              # 测试案例
│       ├── blockA.txt       # 原始故事分块 (Raw Input)
│       ├── blockB.txt       # 原始故事分块
│       └── data.json        # LLM 处理后的结构化 Prompt (Intermediate)
├── models/                  # 模型权重 (本地隔离)
│   ├── sdxl/                # SDXL Base/Refiner
│   ├── ip_adapter/          # IP-Adapter 模型
│   └── llm/                 # 本地 LLM (Qwen2.5/LLaMA3)
└── src/                     # 核心源码
    ├── llm_processor.py     # 任务1: LLM 扩展 Prompt 并生成 JSON
    ├── sdxl_generator.py    # 任务2: SDXL + IP-Adapter 图像生成
    └── pipeline_runner.py   # 任务3: 主控流程 (Orchestrator)
```

### 1.2 数据流 (Data Flow)

1. **Input**: `blockA.txt`, `blockB.txt` (原始剧情文本)。
2. **LLM Processing**: `llm_processor.py` 读取文本，生成包含 `ip_adapter_ref` 字段的结构化 `data.json`。
3. **Consistency Logic**: `pipeline_runner.py` 检查 `data.json`，若当前 Panel 需要一致性，则加载前一帧图像。
4. **SDXL Generation**: `sdxl_generator.py` 接收 Text Prompt + Reference Image (via IP-Adapter)，输出图像。

---

## 2. 详细执行阶段 (Timeline: 4月3日 - 4月26日)

### 🗓️ 阶段 1：环境固化与基线测试 (4月3日 - 4月6日)

**目标**：基于现有的 `environment.yml` 搭建可复现环境，跑通第一个 "Hello World" 图像，签署合规声明。

**Checkpoint 1.1: 环境复现 (4月3日)**

- **动作**：
    - 确认 `scripts/setup_environment.sh` 内容为 `conda env create -f environment.yml`。
    - 确认 `environment.yml` 中明确指定了 `pytorch::pytorch`, `pytorch::torchaudio`, `nvidia::cudatoolkit` 版本。
- **验证**：团队成员 B 和 C 在本地执行 `conda env remove` 后，仅通过 `setup_environment.sh` 重建环境，确保无版本冲突。

**Checkpoint 1.2: 模型下载与验证 (4月4日)**

- **文件**：`scripts/download_models.sh`
- **内容**：下载 SDXL 1.0, IP-Adapter (Face/Full), LLM (Qwen2.5 7B)。
- **动作**：运行脚本，检查 `models/` 目录下的文件大小和哈希值。

**Checkpoint 1.3: 基线推理 (4月5日)**

- **文件**：`src/sdxl_generator.py`
- **代码要求**：
    - 实现 `generate_image(prompt, output_path)` 函数。
    - 禁用所有网络请求 (`import requests` 报错处理)。
- **验证**：输入 "A photo of a cat"，输出 `results/test_cat.png`。
- **合规动作**：在 `README.md` 中添加 "本项目完全离线运行，无外部 API 调用" 声明。

---

### 🗓️ 阶段 2：LLM 提示词工程与结构化 (4月7日 - 4月10日)

**目标**：实现 LLM 对 Prompt 的自动扩充，并生成包含一致性指令的 JSON。

**Checkpoint 2.1: JSON Schema 定义 (4月7日)**

- **文件**：`data/test_0/data.json` (Schema)
- **结构**：
    
    ```json
    {
      "story_id": "test_0",
      "panels": [
        {
          "index": 1,
          "raw_text": "A man in a room",
          "expanded_prompt": "cinematic photo, 8k, [Character_001] wearing red shirt, sitting on a wooden chair, dim lighting, realistic",
          "negative_prompt": "blurry, cartoon, text",
          "reference_image": null // 第一帧无参考
        },
        {
          "index": 2,
          "raw_text": "The man stands up",
          "expanded_prompt": "cinematic photo, 8k, [Character_001] wearing red shirt, standing up, same room, looking at camera",
          "negative_prompt": "blurry, cartoon, text",
          "reference_image": "results/test_0/panel_1.png" // 指向第一帧
        }
      ]
    }
    ```
    

**Checkpoint 2.2: LLM 编排逻辑 (4月8日-9日)**

- **文件**：`src/llm_processor.py`
- **核心逻辑**：
    - **角色锚定**：LLM 必须在第一帧生成时，为角色创建唯一 ID (如 `[Character_001]`) 并在后续帧复用。
    - **引用注入**：LLM 或 Runner 脚本需自动填充 `reference_image` 字段为上一帧的路径。
- **验证**：运行脚本，检查生成的 `data.json` 中第二帧是否包含第一帧的路径。

**Checkpoint 2.3: 端到端文本流 (4月10日)**

- **文件**：`src/pipeline_runner.py` (Part 1)
- **动作**：编写脚本读取 `data/test_0/blockA.txt` -> 调用 LLM -> 生成 `data.json`。

---

### 🗓️ 阶段 3：一致性核心 (IP-Adapter + ControlNet) (4月13日 - 4月16日)

**目标**：解决多图生成中的 "角色变形" 问题，实现跨 Panel 一致性。

**Checkpoint 3.1: IP-Adapter 集成 (4月13日)**

- **文件**：`src/sdxl_generator.py`
- **代码要求**：
    - 修改 `generate_image` 函数，增加参数 `reference_img_path`。
    - 若 `reference_img_path` 不为空，加载 IP-Adapter 权重，提取图像特征注入 UNet。
- **验证**：使用同一个 `[Character_001]` Prompt 生成两张图，对比人脸特征相似度 (DINO Score)。

**Checkpoint 3.2: 时序调度器 (4月14日)**

- **文件**：`src/pipeline_runner.py`
- **核心逻辑**：
    - **顺序执行**：必须按 Panel 1 -> Panel 2 的顺序执行。
    - **状态传递**：Panel 1 生成后，将其保存路径写入内存，供 Panel 2 的 `reference_image` 字段使用。
- **验证**：输入包含 3 个 Panel 的故事，输出 3 张图，角色外观保持一致。

**Checkpoint 3.3: ControlNet 辅助 (4月15日-16日)**

- **文件**：`src/sdxl_generator.py`
- **策略**：针对背景一致性，集成 ControlNet (Canny)。
- **逻辑**：第一帧生成后，提取 Canny Edge 作为后续帧的 Control 条件 (权重 0.5)。

---

### 🗓️ 阶段 4：评估与优化 (4月19日 - 4月22日)

**目标**：引入量化指标，证明系统的优越性。

**Checkpoint 4.1: 自动化评估脚本 (4月19日)**

- **文件**：`scripts/evaluate_results.py`
- **指标**：
    - **CLIP Score**：计算 Image 和 Text 的相关性 (验证 Prompt Adherence)。
    - **DINO Score**：计算 Panel 1 和 Panel 2 的图像特征余弦相似度 (验证 Consistency)。

**Checkpoint 4.2: 极速推理 (4月20日)**

- **策略**：集成 LCM-LoRA。
- **动作**：修改 `sdxl_generator.py`，支持加载 LCM LoRA，将推理步数降至 4-8 步。
- **验证**：生成速度提升 5 倍以上，视觉质量无明显下降。

---

### 🗓️ 阶段 5：封箱与提交 (4月24日 - 4月26日)

**目标**：确保 TA 能一键复现。

**Checkpoint 5.1: 最终复现测试 (4月24日)**

- **动作**：在一台全新的机器上，仅执行以下命令：
    1. `conda env create -f environment.yml`
    2. `bash scripts/download_models.sh`
    3. `python src/pipeline_runner.py`
- **标准**：从零环境到输出图像，全程无人工干预，结果与提交样本一致。

**Checkpoint 5.2: 报告撰写 (4月25日)**

- **重点**：详细描述 `data.json` 的生成逻辑，以及 IP-Adapter 在流水线中的具体位置。

**Checkpoint 5.3: 提交 (4月26日)**

- **打包**：`code/`, `report/`, `results/`, `environment.yml`。

