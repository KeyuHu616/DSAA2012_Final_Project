"""
Story Generation Pipeline - Main Package
Narrative Weaver Pro: SOTA-based multi-image story generation system
"""

from .script_director.llm_parser import LLMScriptParser, ProductionBoard
from .script_director.prompt_enhancer import PromptEnhancer
from .core_generator.pipeline import NarrativeGenerationPipeline
from .orchestrator.run_pipeline import run_pipeline as run_pipeline

__version__ = "1.0.0"
__author__ = "Narrative Weaver Pro Team"

__all__ = [
    "LLMScriptParser",
    "ProductionBoard",
    "PromptEnhancer",
    "NarrativeGenerationPipeline",
    "run_pipeline",
]
