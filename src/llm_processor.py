"""
LLM Semantic Decoupling Module for White-Box Story Pipeline

This module transforms raw storyboard text into a structured JSON representation
that cleanly separates:
- GLOBAL CHARACTER DESCRIPTIONS (immutable across all frames)
- PER-FRAME VARIABLE ELEMENTS (action, background, expression, camera, mood)

This decoupling is critical for the downstream White-Box SDXL generator to inject
character identity directly into UNet attention layers via MSA mechanism.

Architecture Reference:
- StoryDiffusion: Uses character embeddings for multi-frame consistency
- ConsiStory: Shares attention features across frames for training-free consistency
- Our innovation: Structured decoupling enables precise character injection

Author: White-Box Story Pipeline Team
Date: 2026-04-24
"""

import json
import re
import os
import torch
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# SYSTEM PROMPT FOR SEMANTIC DECOUPLING
# ============================================================================

SYSTEM_PROMPT_DECOUPLING = '''You are a professional storyboard artist converting raw scripts into precise visual specifications for AI image generation.

## YOUR CORE TASK
Transform story text into a STRICTLY STRUCTURED JSON format that separates:
1. CHARACTER identity (defined ONCE, used everywhere)
2. FRAME details (action, background, expression, camera, mood per panel)

## CRITICAL RULES

### For CHARACTERS (immutable, define once):
- Create ONE entry per named character (<Name> or [Name])
- MUST include: id, name, gender, appearance, outfit, signature_feature
- Global prompt: ≤30 words, include gender+hair+outfit+signature
- This global prompt is the ONLY identity anchor for ALL frames

### For FRAMES (per-frame mutable elements):
Each frame ONLY contains CHANGING aspects:
- action: What the character DOES (verb-focused, no appearance)
- background: Environment, location, props
- expression: Facial expression
- camera_angle: Shot type + angle (wide/medium/close_up + eye_level/low_angle/high_angle)
- mood: Emotional atmosphere

## FORBIDDEN PATTERNS
❌ NEVER describe character appearance in frame prompts (use global_prompt instead)
❌ NEVER use generic terms ("the girl", "a man") — always use character IDs
❌ NEVER add quality modifiers ("masterpiece", "8k", "best quality")
❌ NEVER repeat character appearance across frames

## OUTPUT FORMAT (STRICT JSON)
```json
{
  "story_id": "extract from filename or generate",
  "global_style": "overall art style description",
  "characters": [
    {
      "id": "CHAR_001",
      "name": "extracted name",
      "gender": "man/woman/person",
      "appearance": {"hair": "", "build": "", "skin_tone": ""},
      "outfit": "specific clothing description",
      "signature_feature": "most distinctive visual trait",
      "global_prompt": "≤30 words: gender + hair + outfit + signature"
    }
  ],
  "frames": [
    {
      "index": 1,
      "action": "verb-focused action description",
      "background": "environment and location",
      "expression": "facial expression",
      "camera_angle": "shot_type + camera_angle",
      "mood": "emotional atmosphere",
      "negative_prompt": "blurry, distorted, deformed, extra fingers, watermark, text, ugly, low quality"
    }
  ]
}
```

## EXAMPLE TRANSFORMATION

Input:
```
[SCENE-1] <Lily> makes breakfast in her kitchen. She looks happy.
[SCENE-2] <She> pours coffee and reads newspaper.
[SCENE-3] <Lily> smiles at camera.
```

Output:
```json
{
  "characters": [
    {
      "id": "Lily_001",
      "name": "Lily",
      "gender": "woman",
      "appearance": {"hair": "long brown hair", "build": "average, mid-20s", "skin_tone": "light"},
      "outfit": "blue apron over white blouse",
      "signature_feature": "blue apron",
      "global_prompt": "a woman with long brown hair wearing blue apron over white blouse"
    }
  ],
  "frames": [
    {
      "index": 1,
      "action": "standing at kitchen counter making breakfast",
      "background": "cozy kitchen with white cabinets and wooden countertop",
      "expression": "happy, soft smile",
      "camera_angle": "medium shot, low angle from counter height",
      "mood": "peaceful domestic morning"
    },
    {
      "index": 2,
      "action": "pouring coffee from pot into mug",
      "background": "same kitchen, morning sunlight through window",
      "expression": "relaxed, casual",
      "camera_angle": "close-up, eye level",
      "mood": "leisurely morning calm"
    },
    {
      "index": 3,
      "action": "smiling directly at camera",
      "background": "same kitchen, slightly blurred background",
      "expression": "bright smile, friendly",
      "camera_angle": "portrait, eye level",
      "mood": "cheerful connection"
    }
  ]
}
```

Now process the following story text and output ONLY the JSON:
'''


# ============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class CharacterAppearance:
    """Character appearance details - compact representation"""
    hair: str = ""
    build: str = ""
    skin_tone: str = ""


@dataclass
class Character:
    """Global character definition - immutable across all frames"""
    id: str
    name: str
    gender: str
    appearance: CharacterAppearance
    outfit: str
    signature_feature: str
    global_prompt: str  # ≤30 words, used for MSA injection

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "gender": self.gender,
            "appearance": asdict(self.appearance),
            "outfit": self.outfit,
            "signature_feature": self.signature_feature,
            "global_prompt": self.global_prompt
        }


@dataclass
class Frame:
    """Per-frame variable elements - action, background, expression, camera, mood"""
    index: int
    action: str
    background: str
    expression: str = ""
    camera_angle: str = ""
    mood: str = ""
    negative_prompt: str = "blurry, distorted, deformed, extra fingers, watermark, text, ugly, low quality"
    full_prompt: str = ""  # Auto-assembled

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "action": self.action,
            "background": self.background,
            "expression": self.expression,
            "camera_angle": self.camera_angle,
            "mood": self.mood,
            "negative_prompt": self.negative_prompt,
            "full_prompt": self.full_prompt
        }

    def assemble_full_prompt(self, char_global_prompt: str, global_style: str = "") -> str:
        """
        Assemble the complete prompt for SDXL generation.
        Format: [STYLE], [CHAR_GLOBAL], [ACTION], [BACKGROUND], [EXPRESSION], [CAMERA], [MOOD]
        """
        parts = []
        if global_style:
            parts.append(global_style)
        parts.append(char_global_prompt)
        if self.action:
            parts.append(self.action)
        if self.background:
            parts.append(self.background)
        if self.expression:
            parts.append(f"expression: {self.expression}")
        if self.camera_angle:
            parts.append(f"camera: {self.camera_angle}")
        if self.mood:
            parts.append(f"mood: {self.mood}")

        self.full_prompt = ", ".join(parts)
        return self.full_prompt


@dataclass
class StoryData:
    """Complete structured story representation"""
    story_id: str
    global_style: str = ""
    characters: List[Character] = None
    frames: List[Frame] = None

    def __post_init__(self):
        if self.characters is None:
            self.characters = []
        if self.frames is None:
            self.frames = []

    def to_dict(self) -> Dict:
        return {
            "story_id": self.story_id,
            "global_style": self.global_style,
            "characters": [c.to_dict() for c in self.characters],
            "frames": [f.to_dict() for f in self.frames]
        }

    def assemble_all_prompts(self):
        """Post-process: assemble full_prompt for each frame"""
        if not self.characters:
            return
        # Use first character's global prompt (primary character)
        primary_char_prompt = self.characters[0].global_prompt
        for frame in self.frames:
            frame.assemble_full_prompt(primary_char_prompt, self.global_style)


# ============================================================================
# LLM PROCESSOR CLASS
# ============================================================================

class LLMProcessor:
    """
    LLM-based semantic decoupling processor.

    Transforms raw storyboard text into structured JSON with:
    - Character global prompts (immutable identity anchors)
    - Per-frame variable elements (action, background, etc.)

    This structured output enables the downstream White-Box SDXL generator
    to inject character identity directly into UNet attention layers via MSA.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        use_flash_attention: bool = True,
        load_in_8bit: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize LLM Processor.

        Args:
            model_path: HuggingFace model path (supports HF-Mirror for China)
            device: Computation device
            use_flash_attention: Enable Flash Attention 2 for efficiency
            load_in_8bit: Use 8-bit quantization to save memory
            hf_token: HuggingFace token for gated models
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_flash_attention = use_flash_attention
        self.load_in_8bit = load_in_8bit
        self.hf_token = hf_token

        self._load_model()

    def _load_model(self):
        """Load Qwen2.5-7B model with optimizations"""
        print(f"[LLM] Loading model: {self.model_path}")
        print(f"[LLM] Device: {self.device}, FlashAttn: {self.use_flash_attention}, 8bit: {self.load_in_8bit}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                token=self.hf_token
            )
        except Exception as e:
            print(f"[LLM] Standard tokenizer load failed: {e}")
            print("[LLM] Trying with HF-Mirror fallback...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "http://hf-mirror.com/" + self.model_path,
                trust_remote_code=True
            )

        # Model loading kwargs
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        if self.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
        except Exception as e:
            print(f"[LLM] Model load failed: {e}")
            print("[LLM] Trying with HuggingFace-Mirror...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "http://hf-mirror.com/" + self.model_path,
                **load_kwargs
            )

        self.model.eval()
        print("[LLM] Model loaded successfully!")

    def parse_raw_input(self, text: str) -> str:
        """
        Preprocess raw story text before LLM parsing.

        Cleans and normalizes the input format:
        - Extracts scene blocks [SCENE-N]
        - Identifies character names <Name>
        - Normalizes separators [SEP]
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def extract_story_id(self, file_path: str) -> str:
        """Extract story ID from file path (e.g., '01' from 'TaskA/01.txt')"""
        basename = os.path.basename(file_path)
        match = re.match(r'(\d+)', basename)
        if match:
            return match.group(1)
        return "unknown"

    def run_inference(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
        """
        Run LLM inference with ChatML format.

        Args:
            prompt: Input prompt (already prefixed with system prompt)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DECOUPLING},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def parse_llm_output(self, raw_output: str) -> Optional[Dict]:
        """
        Parse LLM output into structured JSON.
        Includes fallback regex extraction if JSON parsing fails.
        """
        # Try direct JSON parsing first
        try:
            # Find JSON block (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\{[\s\S]*\}', raw_output)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = raw_output

            data = json.loads(json_str)

            # Validate required fields
            if "frames" not in data:
                print("[LLM] WARNING: No 'frames' field in output, attempting recovery...")
                return self._fallback_parse(raw_output)

            return data

        except json.JSONDecodeError as e:
            print(f"[LLM] JSON parsing failed: {e}")
            print("[LLM] Attempting fallback regex extraction...")
            return self._fallback_parse(raw_output)

    def _fallback_parse(self, raw_text: str) -> Optional[Dict]:
        """
        Fallback parser using regex when LLM output is malformed.
        Extracts what we can and fills in defaults.
        """
        print("[LLM] Using fallback parser - results may be incomplete")

        # Extract character names
        char_names = re.findall(r'[<\[]([A-Z][a-z]+)[>\]]', raw_text)
        char_names = list(dict.fromkeys(char_names))  # Remove duplicates, preserve order

        # Extract scene/action descriptions
        scene_blocks = re.split(r'\[SCENE-\d+\]|\[SEP\]', raw_text)

        frames = []
        for i, block in enumerate(scene_blocks):
            block = block.strip()
            if len(block) > 10:  # Filter out very short blocks
                frames.append({
                    "index": i + 1,
                    "action": block[:200],  # Truncate long descriptions
                    "background": "indoor scene",  # Default
                    "expression": "neutral",  # Default
                    "camera_angle": "medium shot, eye level",  # Default
                    "mood": "natural",  # Default
                    "negative_prompt": "blurry, distorted, deformed, extra fingers, watermark, text, ugly, low quality"
                })

        # Build characters from extracted names
        characters = []
        for i, name in enumerate(char_names[:3]):  # Max 3 characters
            characters.append({
                "id": f"{name.upper()}_{i+1:03d}",
                "name": name,
                "gender": "person",
                "appearance": {"hair": "short hair", "build": "average", "skin_tone": "medium"},
                "outfit": "casual clothing",
                "signature_feature": "distinctive appearance",
                "global_prompt": f"a person with short hair wearing casual clothing"
            })

        return {
            "story_id": "fallback",
            "global_style": "clean storyboard illustration",
            "characters": characters,
            "frames": frames if frames else [{"index": 1, "action": raw_text[:200], "background": "scene", "expression": "neutral", "camera_angle": "medium", "mood": "natural", "negative_prompt": "blurry, distorted"}]
        }

    def process(self, input_path: str) -> StoryData:
        """
        Main entry point: Process raw story text file into structured StoryData.

        Pipeline:
        1. Read raw text file
        2. Extract story ID from filename
        3. Parse and clean text
        4. Run LLM inference
        5. Parse JSON output
        6. Convert to StoryData and post-process

        Args:
            input_path: Path to raw story text file

        Returns:
            StoryData object with structured characters and frames
        """
        print(f"\n[LLM] Processing: {input_path}")

        # Read raw text
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Extract story ID
        story_id = self.extract_story_id(input_path)

        # Preprocess text
        clean_text = self.parse_raw_input(raw_text)

        # Run LLM inference
        print("[LLM] Running inference...")
        raw_output = self.run_inference(clean_text)

        print(f"[LLM] Raw output length: {len(raw_output)} chars")

        # Parse LLM output
        parsed_data = self.parse_llm_output(raw_output)

        if parsed_data is None:
            print("[LLM] FATAL: Could not parse LLM output")
            raise ValueError("LLM output parsing failed completely")

        # Convert to StoryData
        story = StoryData(
            story_id=story_id,
            global_style=parsed_data.get("global_style", "clean storyboard illustration, soft ink outlines"),
            characters=[Character(**c) if isinstance(c, dict) else c for c in parsed_data.get("characters", [])],
            frames=[Frame(**f) if isinstance(f, dict) else f for f in parsed_data.get("frames", [])]
        )

        # Post-process: assemble full prompts for each frame
        story.assemble_all_prompts()

        # Summary
        print(f"[LLM] Parsed: {len(story.characters)} character(s), {len(story.frames)} frame(s)")
        for char in story.characters:
            print(f"  - {char.id}: {char.global_prompt[:60]}...")
        for frame in story.frames[:3]:
            print(f"  - Frame {frame.index}: {frame.action[:50]}...")

        return story

    def process_text_direct(self, raw_text: str, story_id: str = "direct") -> StoryData:
        """
        Process text directly (without file I/O).
        Useful for testing or streaming scenarios.
        """
        clean_text = self.parse_raw_input(raw_text)
        raw_output = self.run_inference(clean_text)
        parsed_data = self.parse_llm_output(raw_output)

        if parsed_data is None:
            raise ValueError("LLM output parsing failed")

        story = StoryData(
            story_id=story_id,
            global_style=parsed_data.get("global_style", ""),
            characters=[Character(**c) for c in parsed_data.get("characters", [])],
            frames=[Frame(**f) for f in parsed_data.get("frames", [])]
        )

        story.assemble_all_prompts()
        return story


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Test the LLM processor with sample data"""
    import sys

    if len(sys.argv) < 2:
        # Test with default data
        test_text = """
        [SCENE-1] <Lily> makes breakfast in her kitchen. She cracks eggs into a bowl.
        [SCENE-2] <She> pours coffee into a mug and smiles.
        [SCENE-3] <Lily> sits at the table eating toast.
        """

        processor = LLMProcessor()
        story = processor.process_text_direct(test_text, "test_01")
        print("\n[RESULT]")
        print(json.dumps(story.to_dict(), indent=2, ensure_ascii=False))

    else:
        # Process file
        input_path = sys.argv[1]
        processor = LLMProcessor()
        story = processor.process(input_path)

        # Save output
        output_path = input_path.replace('.txt', '_parsed.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(story.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Parsed output to: {output_path}")


if __name__ == "__main__":
    main()
