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

    Transforms structured production board data into high-quality,
    SDXL-optimized prompts with proper style modifiers, quality enhancers,
    and character tokens for consistency.
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

    # Global style modifiers
    STYLE_MODIFIERS = {
        "warm_cinematic_lifestyle": (
            "cinematic lighting, warm color grading, film grain, "
            "shallow depth of field, golden hour tones"
        ),
        "urban_drama": (
            "dramatic lighting, high contrast, cool color palette, "
            "neon accents, gritty urban atmosphere"
        ),
        "whimsical_illustration": (
            "digital art, vibrant colors, soft lighting, "
            "storybook illustration style, dreamy atmosphere"
        ),
        "photorealistic_documentary": (
            "documentary photography, natural lighting, "
            "candid moment, realistic textures, photojournalistic"
        ),
        "cinematic_realistic": (
            "cinematic composition, natural lighting, film grain, "
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
        Enhance a single panel prompt with all modifiers

        Args:
            panel: Panel/scene object
            characters: Character dictionary
            global_style: Global visual style
            is_first_frame: Whether this is the first frame

        Returns:
            Enhanced prompt string ready for SDXL
        """
        components = []

        # 1. Quality prefixes (take top 3 most important)
        components.extend(self.QUALITY_PREFIXES[:3])

        # 2. Style modifier
        style_mod = self.STYLE_MODIFIERS.get(global_style, "cinematic quality")
        components.append(style_mod)

        # 3. Shot type modifier
        shot_mod = self.SHOT_TYPE_MODIFIERS.get(panel.shot_type, "")
        if shot_mod:
            components.append(shot_mod)

        # 4. Main content (from enhanced prompt or raw text)
        main_content = panel.enhanced_prompt if panel.enhanced_prompt else panel.raw_prompt
        components.append(main_content)

        # 5. Inject character tokens for consistency
        char_tokens = []
        for char_name, char_info in characters.items():
            # Only inject if character appears in this panel
            if f"<{char_name}>" in panel.raw_prompt or char_name.lower() in panel.raw_prompt.lower():
                char_tokens.append(char_info.token)
                # Add key attributes for consistency reinforcement
                if char_info.key_attributes:
                    components.append(", ".join(char_info.key_attributes[:2]))

        # Add character tokens at the beginning
        if char_tokens:
            components.insert(0, f"[{' + '.join(char_tokens)}]")

        # 6. Lighting mood
        if panel.lighting_mood and panel.lighting_mood != "natural":
            components.append(panel.lighting_mood)

        # 7. Technical suffixes
        technical_suffixes = [
            "sharp focus",
            "professional color grading",
            "film still",
            "award-winning photography"
        ]
        components.extend(technical_suffixes)

        # Combine into final prompt
        final_prompt = ", ".join(filter(None, components))

        return final_prompt

    def create_negative_prompt(self) -> str:
        """
        Create universal negative prompt for SDXL

        Returns:
            Negative prompt string to avoid common issues
        """
        negative_elements = [
            "low quality",
            "blurry",
            "distorted",
            "deformed",
            "ugly",
            "bad anatomy",
            "extra limbs",
            "missing fingers",
            "watermark",
            "text",
            "signature",
            "cropped",
            "out of frame",
            "worst quality",
            "jpeg artifacts",
            "duplicate",
            "morbid",
            "mutilated",
            "mutation",
            "cartoon",
            "anime style"
        ]
        return ", ".join(negative_elements)

    def process_entire_story(
        self,
        production_board: ProductionBoard
    ) -> List[Dict[str, str]]:
        """
        Process all panels in a story

        Args:
            production_board: Complete production blueprint

        Returns:
            List of enhanced prompt dictionaries
        """
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

            print(f"[Enhancer] Frame {i+1}/{len(production_board.panels)}: "
                  f"{len(enhanced)} characters")

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
