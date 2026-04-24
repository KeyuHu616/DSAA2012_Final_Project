"""
White-Box Story Pipeline - Source Package

Modules:
- llm_processor: LLM semantic decoupling
- sdxl_generator_part2: White-box SDXL with CSA/MSA
- evaluator: Best-of-N quality evaluation
- pipeline_runner: Main orchestrator

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

from .llm_processor import LLMProcessor, StoryData, Character, Frame
from .sdxl_generator import WhiteBoxSDXLGenerator, ConsistencyConfig
from .evaluator import StoryEvaluator, SimpleEvaluator
from .pipeline_runner import WhiteBoxStoryPipeline, BatchProcessor, PipelineConfig

__all__ = [
    "LLMProcessor",
    "StoryData", 
    "Character",
    "Frame",
    "WhiteBoxSDXLGenerator",
    "ConsistencyConfig",
    "StoryEvaluator",
    "SimpleEvaluator",
    "WhiteBoxStoryPipeline",
    "BatchProcessor",
    "PipelineConfig",
]

__version__ = "1.0.0"
