# White-Box Story Pipeline

SOTA story image generation using White-Box SDXL with consistency mechanisms.

## Architecture

**Core Innovations:**
- **CSA (Consistent Self-Attention)** - StoryDiffusion-inspired: Caches attention maps from first frame, reuses in early denoising steps
- **MSA (Multi-Source Cross-Attention)** - StoryDiffusion-inspired: Injects character embeddings into UNet cross-attention layers
- **Shared Attention** - ConsiStory-inspired: Manages feature sharing for structural consistency
- **Best-of-N Evaluation** - GenEval-inspired: Multi-metric scoring (DINOv2, CLIP, Aesthetic)

## Project Structure

```
src/
├── llm_processor.py      # LLM semantic decoupling (Qwen2.5-7B)
├── sdxl_generator.py     # White-Box SDXL with CSA/MSA
├── evaluator.py          # Best-of-N quality evaluation
└── pipeline_runner.py    # Main orchestrator

scripts/
├── download_models.sh    # Download required models
└── environment_test.py   # Environment verification
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate story-pipeline

# Or with pip
pip install -r requirements.txt
```

### 2. Download Models

```bash
bash scripts/download_models.sh
```

### 3. Test Environment

```bash
python scripts/environment_test.py
```

### 4. Run Pipeline

```bash
# Single story
python run_pipeline.py data/TaskA/01.txt

# Batch processing
python run_pipeline.py data/TaskA/ --batch

# Custom settings
python run_pipeline.py data/TaskA/01.txt --steps 20 --candidates 3 --seed 123
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--steps` | Denoising steps | 24 |
| `--candidates` | Best-of-N candidates | 2 |
| `--seed` | Base random seed | 42 |
| `--no-eval` | Disable evaluator | False |
| `--batch` | Batch mode | False |

## Output

Results saved to `./results/`:
```
results/
├── 01/
│   ├── panel_01.png
│   ├── panel_02.png
│   └── summary.json
└── json_data/
    └── data01.json
```

## Technical Details

### White-Box SDXL Pipeline

Unlike the previous black-box approach (`pipe.generate()`), this implementation provides full control over the denoising process:

1. **Manual Denoising Loop**: Step-by-step control over UNet inference
2. **Attention Injection**: Custom CSA/MSA processors replace default attention
3. **Feature Caching**: First frame features cached for subsequent frames
4. **Character Injection**: Global character prompt injected via MSA

### Consistency Mechanisms

**CSA (Consistent Self-Attention)**:
- Caches attention maps during first frame generation
- Reuses structural attention patterns in early denoising steps (50% by default)
- Allows detail divergence in later steps

**MSA (Multi-Source Cross-Attention)**:
- Adds character embeddings as additional K/V source
- Ensures character identity appears in all frames
- Injection weight: 0.35 (configurable)

## Requirements

- Python 3.10+
- CUDA-capable GPU (24GB VRAM recommended)
- ~50GB disk space for models

## References

- StoryDiffusion: Consistent Self-Attention for Controllable Generation
- ConsiStory: Training-Free Consistent Story Generation
- GenEval: An Objective Framework for Evaluating Generative Models
