#!/usr/bin/env python3
"""
White-Box Story Pipeline - Main Entry Point

SOTA Story Image Generation using:
- CSA (Consistent Self-Attention) - StoryDiffusion-inspired
- MSA (Multi-Source Cross-Attention) - StoryDiffusion-inspired
- Shared Attention - ConsiStory-inspired
- Best-of-N Evaluation - GenEval-inspired

Usage:
    # Single story
    python run_pipeline.py data/TaskA/01.txt

    # Batch processing
    python run_pipeline.py data/TaskA/ --batch

    # Custom settings
    python run_pipeline.py data/TaskA/01.txt --steps 20 --candidates 3 --seed 123

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline_runner import main

if __name__ == "__main__":
    main()
