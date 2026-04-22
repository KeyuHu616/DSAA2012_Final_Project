#!/usr/bin/env python3
"""
llm_processor.py
================
Refactored: Character recognition and consistency anchoring delegated entirely to LLM.
Core Strategy:
1. Keep all pronouns and tags (<Lily>, She/Her), let LLM resolve references
2. Force LLM to define character IDs ([Lily_001]) in first frame, strictly reuse in subsequent
3. For new characters, require LLM to assign new IDs explicitly
4. Output maintains path injection and JSON Schema
"""

import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chinese network mirror configuration for HuggingFace models
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ============================================================
# Section 1: Input Parsing (Pronoun replacement removed)
# ============================================================

def parse_raw_input(raw_text: str) -> dict:
    """
    Parse raw input, preserve all pronouns and tags, no text replacement.
    Extract first frame <Name> as default subject name (for prompt construction only).
    """
    blocks = re.split(r'\[SEP\]', raw_text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    scenes = []
    default_subject = None

    for i, block in enumerate(blocks):
        # Remove [SCENE-N] markers, preserve original text (with <Lily> and She/Her)
        text = re.sub(r'\[SCENE-\d+\]\s*', '', block).strip()
        
        # Only extract <Name> from first frame as default subject (for prompt context)
        if i == 0:
            name_match = re.search(r'<(\w+)>', text)
            if name_match:
                default_subject = name_match.group(1)
        
        scenes.append({"index": i + 1, "text": text})

    if default_subject is None:
        default_subject = "Subject"

    # Backward compatibility: return both keys
    return {
        "subject_name": default_subject,  # For pipeline_runner.py compatibility
        "default_subject": default_subject,  # New interface
        "scenes": scenes
    }

# ============================================================
# Section 2: LLM Prompt (Visual expansion + Character ID anchoring + Gender/Outfit consistency)
# ============================================================

SYSTEM_PROMPT = r"""You are a professional storyboard artist for animated films. Your task is to convert simple scene descriptions into detailed, consistent visual prompts.

# --------------------------
# SECTION 1: ROLE ID & CONSISTENCY RULES (STRICT)
# --------------------------

## PANEL 1: DEFINITION PHASE
- Extract ALL named characters from angle brackets (e.g., <Ryan>, <Lily>, <Jack>).
- For EACH character, assign a permanent UNIQUE ID: [Name_XXX] (e.g., [Ryan_001], [Sara_002]).
- The ID MUST appear at the VERY BEGINNING of the expanded_prompt.
- You MUST include a CLEAR physical description ONCE in Panel 1:
  * GENDER: Explicitly state "a man" or "a woman" (e.g., "[Ryan_001], a woman with...")
  * HAIR: length, color, style (e.g., "short dark hair")
  * OUTFIT: garment types and colors (e.g., "gray hoodie, black leggings, black shoes")
  * ACCESSORIES: bags, glasses, jewelry if mentioned
  * Do NOT skip appearance details.

## PANELS 2+: LOCKED PHASE
- You MUST reuse the EXACT same ID from Panel 1. Never change it.
- In EVERY panel, explicitly state the character's gender and outfit: "[Ryan_001], a woman wearing the same gray hoodie and black leggings, ..."
- NEVER repeat hair description after Panel 1. Assume the viewer remembers.
- Replace ALL pronouns (She/He/They/Her/His) with the correct ID mentally before writing.
- STRICTLY FORBIDDEN: Generic terms like "the girl", "the man", "a woman". Use IDs only.

## NEW CHARACTER HANDLING
- If the text introduces a new name (e.g., <Leo>), assign a new ID ([Leo_002]) and mark as new.
- If no new names appear, reuse existing IDs exclusively.

# --------------------------
# SECTION 2: PROMPT EXPANSION REQUIREMENTS
# --------------------------

## MANDATORY DETAILS (Panel 1)
- Expand the simple action into a vivid 1-2 sentence visual.
- Include: Location cues, Time of day if implied, Character pose, Gender, Outfit details.
- Example: Instead of "walks quickly toward a bus", write "[Ryan_001], a woman with short dark hair wearing a gray hoodie, black leggings, and black shoes, walks briskly down the sidewalk towards a red bus stop, early morning light casting shadows on the pavement."

## SUBSEQUENT PANELS (2+)
- In EACH panel, explicitly mention: ID, gender, and outfit (e.g., "[Ryan_001], a woman wearing the same gray hoodie and black leggings, pauses at the bus door...")
- Focus on ACTION and ENVIRONMENT changes only.
- Preserve the exact appearance from Panel 1 explicitly by mentioning "wearing the same [outfit description]".
- Example: "[Ryan_001], a woman wearing the same gray hoodie and black leggings, stops in front of the bus door and leans forward, looking intently at the approaching vehicle."

## STYLE CONSTRAINTS
- NO quality fluff: ban "masterpiece", "8k", "ultra-realistic", "trending on ArtStation".
- NO emoji or markdown symbols.
- Keep language concise: cinematic but simple.
- The system will prepend a global storyboard style later.

# --------------------------
# SECTION 3: OUTPUT SCHEMA (STRICT JSON)
# --------------------------

{
  "story_id": "02",
  "panels": [
    {
      "index": 1,
      "raw_text": "<Ryan> walks quickly toward a bus.",
      "expanded_prompt": "[Ryan_001], a woman with short dark hair wearing a gray hoodie, black leggings, and black shoes, walks briskly down the sidewalk towards a red bus stop, early morning light casting shadows on the pavement.",
      "negative_prompt": "blurry, distorted, multiple people, wrong outfit, gender swap",
      "reference_image": null
    },
    {
      "index": 2,
      "raw_text": "He pauses at the door and looks ahead.",
      "expanded_prompt": "[Ryan_001], a woman wearing the same gray hoodie and black leggings, stops in front of the bus door and leans forward, looking intently at the approaching vehicle.",
      "negative_prompt": "blurry, distorted, multiple people, wrong outfit, gender swap",
      "reference_image": "results/02/panel_1.png"
    }
  ]
}
"""

def build_user_prompt(default_subject: str, scenes: list) -> str:
    """Construct User Prompt, force LLM to expand visual details and anchor IDs with gender/outfit"""
    scene_blocks = []
    for s in scenes:
        # Preserve original text structure for LLM to parse references
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    return f"""
Input Story Sequence:
{scenes_text}

## Processing Instructions

1. CHARACTER INVENTORY: Identify every distinct character (including <{default_subject}> and any new names).
2. GENDER DETERMINATION: Determine the most likely gender for each character based on context and common knowledge. If uncertain, default to the gender that best fits the name and context.
3. ID ASSIGNMENT: In Panel 1, assign [Name_001] format IDs. For new characters later, increment the number (e.g., [New_002]).
4. VISUAL EXPANSION (Panel 1 ONLY): Expand the first scene into a detailed visual prompt including GENDER, HAIR, OUTFIT, ACCESSORIES, and LOCATION specifics.
5. ACTION FOCUS (Panels 2+): In EACH subsequent panel, explicitly state: ID, gender, and "wearing the same [outfit]" to maintain consistency. Focus on the changed action and camera angle.
6. PRONOUN REPLACEMENT: Mentally replace all "She/He/They/Her/His" with the correct ID and gender before writing.

Output must be pure JSON following the schema exactly. No commentary.
"""

# ============================================================
# Section 3: LLM Loading and Inference (China-friendly with HF-Mirror)
# ============================================================

def load_llm(llm_path: str):
    """
    Load LLM from local path or HuggingFace (via HF-Mirror).
    Mirrors ControlNet/SDXL loading pattern for consistency.
    """
    print(f"[LLM] Loading model (auto-cache via HF-Mirror): {llm_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        cache_dir="./models",
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="./models",
    )
    model.eval()
    print("[LLM] Model loaded successfully.")
    return tokenizer, model

def run_llm_inference(tokenizer, model, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 4096) -> str:
    """Qwen2.5 ChatML format inference"""
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ============================================================
# Section 4: JSON Parsing and Post-processing (Path injection unchanged)
# ============================================================

def parse_llm_output(raw_output: str) -> dict:
    """Extract JSON from LLM output"""
    # Remove possible Markdown code blocks
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()

    # Extract first complete JSON object
    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start != -1 and brace_end != -1:
        json_str = cleaned[brace_start:brace_end + 1]
    else:
        json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parsing failed: {e}")
        with open("llm_debug.txt", "w", encoding="utf-8") as f:
            f.write(f"Raw output:\n{raw_output}\n\nCleaned:\n{cleaned}")
        print("[DEBUG] Raw output saved to llm_debug.txt")
        return None

def derive_story_id(input_path: str) -> str:
    """Extract numeric ID from filename"""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    m = re.search(r'\d+', stem)
    return m.group(0) if m else stem

def post_process(data: dict, story_id: str, results_root: str, raw_scenes: list) -> dict:
    """
    Post-processing: Inject IDs, paths, and force alignment with original Raw Text.
    """
    data["story_id"] = story_id
    panels = data.get("panels", [])
    raw_text_map = {s["index"]: s["text"] for s in raw_scenes}

    for p in panels:
        idx = p["index"]
        # Force restore original Raw Text (maintain input authenticity)
        if idx in raw_text_map:
            p["raw_text"] = raw_text_map[idx]
        
        # Inject reference paths (Panel 1 has no reference, Panel N references N-1)
        if idx == 1:
            p["reference_image"] = None
        else:
            ref_path = os.path.join(results_root, story_id, f"panel_{idx-1}.png")
            p["reference_image"] = ref_path.replace("\\", "/")
            
    return data


# ============================================================
# Section 5: Main Process
# ============================================================

def process(input_path: str, output_path: str, llm_path: str, results_root: str = "results"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[INPUT] Parsing: {input_path}")
    parsed = parse_raw_input(raw_text)
    story_id = derive_story_id(input_path)
    
    # Use compatible key name for logging
    print(f"[INPUT] StoryID: {story_id}, Subject: {parsed['subject_name']}, Frames: {len(parsed['scenes'])}")
    for s in parsed["scenes"]:
        print(f"  [SCENE-{s['index']}] {s['text']}")

    # Load LLM
    tokenizer, model = load_llm(llm_path)
    
    # Build prompt
    user_prompt = build_user_prompt(parsed["default_subject"], parsed["scenes"])
    print("\n[LLM] Running inference (Character ID anchoring mode)...")
    
    raw_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT, user_prompt)
    data = parse_llm_output(raw_output)
    
    if data is None:
        print("[ERROR] LLM output parsing failed, aborting")
        return None

    # Post-processing injection
    data = post_process(data, story_id, results_root, parsed["scenes"])
    
    # Save JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OUTPUT] JSON saved: {output_path}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Refactored LLM Processor with Character ID Anchoring")
    parser.add_argument("--input", type=str, required=True, help="Input .txt file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model repo_id or local path")
    parser.add_argument("--results_root", type=str, default="results", help="Image output root directory")
    args = parser.parse_args()
    
    process(args.input, args.output, args.llm_path, args.results_root)

if __name__ == "__main__":
    main()
