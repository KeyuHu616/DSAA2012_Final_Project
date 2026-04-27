"""
Prompt Enhancer - Advanced prompt engineering for SDXL image generation
Transforms production board data into optimized prompts for diffusion models
"""

import re
from typing import Dict, List, Optional
from .llm_parser import ProductionBoard, Panel, Character


class PromptEnhancer:
    """
    Prompt Enhancement Engine

    CRITICAL FIX: Ensure photorealistic style, block non-realistic styles.
    """

    # SDXL-optimized quality prefixes
    QUALITY_PREFIXES = [
        "masterpiece",
        "best quality",
        "ultra-detailed",
        "8k uhd",
        "highly detailed",
        "professional photography"
    ]

    # Global style modifiers - ALL converted to photorealistic
    # CRITICAL: Remove terms that might cause animation/illustration styles
    STYLE_MODIFIERS = {
        "warm_cinematic_lifestyle": (
            "photorealistic photography, cinematic lighting, warm color grading, "
            "shallow depth of field, golden hour tones, natural shadows"
        ),
        "urban_drama": (
            "photorealistic photography, dramatic lighting, high contrast, "
            "cool color palette, neon accents, gritty urban atmosphere"
        ),
        "whimsical_illustration": (
            # BLOCKED - Convert to photorealistic
            "photorealistic photography, warm tones, natural lighting, "
            "natural colors, realistic proportions"
        ),
        "photorealistic_documentary": (
            "photorealistic photography, documentary style, natural lighting, "
            "candid moment, realistic textures"
        ),
        "cinematic_realistic": (
            "photorealistic photography, cinematic composition, natural lighting, "
            "shallow depth of field, professional color grading"
        ),
        "cinematic_photorealistic": (
            "photorealistic photography, cinematic composition, natural lighting, "
            "shallow depth of field, professional color grading"
        )
    }

    # Shot type modifiers
    SHOT_TYPE_MODIFIERS = {
        "extreme_closeup": "macro photography, extreme detail, intimate framing",
        "closeup": "portrait orientation, facial focus, emotional expression visible",
        "medium": "half-body shot, balanced composition, subject and context visible",
        "wide": "full body shot, environmental storytelling, establishing view",
        "over_shoulder": "two-shot composition, conversational framing, depth layering",
        "establishing": "wide angle, location establishment, atmospheric perspective"
    }

    def __init__(self):
        """Initialize the prompt enhancer"""
        self.generated_prompts = []

    def enhance_panel_prompt(
        self,
        panel: Panel,
        characters: Dict[str, Character],
        global_style: str,
        is_first_frame: bool = False
    ) -> str:
        """
        Enhance a single panel prompt with modifiers.
        CRITICAL FIX: Ensure photorealistic style at the end.
        """
        components = []

        # 1. Main content FIRST - this is the most important part
        main_content = panel.enhanced_prompt if panel.enhanced_prompt else panel.raw_prompt
        if main_content:
            components.append(main_content)

        # 2. Style modifier (add after main content)
        # Get style and ensure it's photorealistic
        style_mod = self.STYLE_MODIFIERS.get(global_style, 
            "photorealistic photography, cinematic quality")
        components.append(style_mod)

        # 3. Shot type modifier
        shot_mod = self.SHOT_TYPE_MODIFIERS.get(panel.shot_type, "")
        if shot_mod:
            components.append(shot_mod)

        # 4. Quality suffixes - include photorealistic emphasis
        components.append("photorealistic, sharp focus, 8k detailed")

        # Combine into final prompt
        final_prompt = ", ".join(filter(None, components))

        return final_prompt

    def create_negative_prompt(self) -> str:
        """
        Create universal negative prompt for SDXL
        ENHANCED: Block animation styles and common generation issues
        """
        negative_elements = [
            "low quality",
            "blurry",
            "blurry hands",
            "blurry face",
            "distorted",
            "deformed",
            "ugly",
            "bad anatomy",
            "extra limbs",
            "missing limbs",
            "fused fingers",
            "too many fingers",
            "missing fingers",
            "extra fingers",
            "poorly drawn hands",
            "poorly drawn face",
            "watermark",
            "text",
            "signature",
            "cropped",
            "out of frame",
            "worst quality",
            "jpeg artifacts",
            "cartoon",
            "anime style",
            "illustration",
            "painting",
            "drawing",
            "sketch",
            "anime",
            "manga",
            "comic",
            "2D art style",
            "3D render",
            "CGI",
            "plastic looking",
            "toy-like",
            "over-saturated colors"
        ]
        return ", ".join(negative_elements)

    def process_entire_story(
        self,
        production_board: ProductionBoard
    ) -> List[Dict[str, str]]:
        """Process all panels in a story - SIMPLIFIED"""
        enhanced_prompts = []
        negative_prompt = self.create_negative_prompt()

        for i, panel in enumerate(production_board.panels):
            enhanced = self.enhance_panel_prompt(
                panel=panel,
                characters=production_board.characters,
                global_style=production_board.global_style,
                is_first_frame=(i == 0)
            )

            enhanced_prompts.append({
                "panel_id": panel.panel_id,
                "prompt": enhanced,
                "negative_prompt": negative_prompt,
                "raw_scene": panel.raw_prompt,
                "shot_type": panel.shot_type,
                "is_first_frame": i == 0
            })

            print(f"[Enhancer] Frame {i+1}: {len(enhanced)} chars")

        self.generated_prompts = enhanced_prompts
        return enhanced_prompts

    def get_prompt_by_index(self, index: int) -> Optional[Dict[str, str]]:
        """Get enhanced prompt by panel index"""
        if 0 <= index < len(self.generated_prompts):
            return self.generated_prompts[index]
        return None


if __name__ == "__main__":
    # Simple test
    from llm_parser import LLMScriptParser

    parser = LLMScriptParser()
    enhancer = PromptEnhancer()

    try:
        board = parser.process_script_file("/home/KeyuHu/code/DSAA2012FinalNew/data/TaskA/06.txt")
        prompts = enhancer.process_entire_story(board)

        print("\n[Test] Sample enhanced prompts:")
        for p in prompts[:2]:
            print(f"\nPanel {p['panel_id']}:")
            print(f"  Shot: {p['shot_type']}")
            print(f"  Prompt: {p['prompt'][:200]}...")
    except Exception as e:
        print(f"[Test] Test failed: {e}")
