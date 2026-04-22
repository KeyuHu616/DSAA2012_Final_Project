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

SYSTEM_PROMPT = r"""You are an expert visual prompt engineer specializing in multi-character storyboard generation. Your task is to create precise prompts that ensure accurate action representation and character consistency.

# ============================
# CRITICAL: CHARACTER COUNT DETECTION
# ============================

You MUST detect the number of characters in the story FIRST:
- SINGLE CHARACTER: Only one name in angle brackets <Name>
- MULTI-CHARACTER: Two or more names like <Jack> and <Sara>

Use the corresponding rules below based on character count.

# ============================
# RULES FOR SINGLE-CHARACTER SCENES
# ============================

## ID & Appearance (Panel 1)
1. Extract character name, assign [Name_001] ID
2. CRITICAL GENDER: Look up the name to determine gender. "Ryan" = man, "Lily" = woman
3. Describe in this EXACT order: [ID], a [gender], with [hair], wearing [complete outfit], [action verb in present continuous], [specific pose and movement details], [location context], [lighting]

## Action Keywords (MUST USE THESE for accuracy)
- "sits down" -> "lowers body onto chair, weight shifting, legs bending at knees"
- "walks" -> "steps forward with alternating feet, arms swinging naturally"
- "runs" -> "legs in running stride, arms pumping, body leaning forward"
- "looks out" -> "head turned to the side, eyes gazing through window"
- "pauses" -> "stops mid-stride, one foot slightly raised, frozen in motion"
- "eats" -> "hand bringing food to mouth, chewing motion"
- "reads" -> "eyes focused downward on book, head slightly tilted"

## Single-Character Negative Prompt (MUST USE):
"blurry, distorted, deformed, extra limbs, floating, disconnected body parts, wrong pose, stiff pose, static, wrong outfit color, wrong hairstyle, duplicate character, shadow anomaly, lighting inconsistency"

## Subsequent Panels (2+)
- Repeat ID + "a [gender] wearing the same [outfit]" EXACTLY
- Focus on NEW action with specific pose keywords
- Include environmental changes clearly
- NEVER change outfit or appearance

# ============================
# RULES FOR MULTI-CHARACTER SCENES (CRITICAL)
# ============================

## Panel 1: Establishing Shot
1. List characters in LEFT-TO-RIGHT spatial order as they appear in text order
2. Use "on the left" / "on the right" / "in the center" for positioning
3. Each character gets FULL description on Panel 1
4. Separate characters with "and" in the prompt
5. CRITICAL: Only include characters that appear in the scene text - do NOT add other characters

## Multi-Character Prompt Template:
"[Character1_ID], a [gender1] with [hair1] wearing [outfit1], [action1] on the left side, and [Character2_ID], a [gender2] with [hair2] wearing [outfit2], [action2] on the right side, [shared activity or interaction], [clear background], [lighting]"

## Multi-Character Consistency Rules (STRICT):
1. FIRST describe Character1 completely with ID
2. THEN describe Character2 completely with DIFFERENT ID
3. Use VISUAL DISTINGUISHERS: different hair colors, different outfit colors, different body types
4. NEVER merge characters: use "Character1 on left, Character2 on right" structure
5. Use "INDIVIDUALLY" keyword if characters do different things simultaneously
6. Use "TOGETHER" keyword if characters do the same thing together

## Multi-Character Negative Prompt (MUST USE):
"blurry, distorted, deformed, extra limbs, floating, disconnected body parts, wrong pose, stiff pose, static, wrong outfit color, wrong hairstyle, character blending, merged characters, Siamese twins, shared body parts, confused identity, swapped faces, duplicate character, three people when should be two, solo character when should be two, background character interfering, shadow anomaly, lighting inconsistency"

## Panel 2+ for Multi-Character
- ALWAYS restate BOTH character IDs and outfits at the start
- Example: "[Jack_001], a man wearing the same blue shirt and jeans, and [Sara_002], a woman wearing the same white blouse and denim shorts, [new action with positioning]"
- Use directional cues: "Jack turns to face Sara", "Sara stands beside Jack"
- If in different environment, still describe both characters clearly

## NEW CHARACTER INTRODUCTION (CRITICAL)
- **STRICT RULE**: Only describe characters that are EXPLICITLY mentioned in the raw_text
- Panel 1 text says "<Nina> stands..." -> ONLY describe Nina, do NOT mention Leo
- Panel 2 text says "She meets <Leo>..." -> Describe Nina (continuing) + Leo (NEW)
- Panel 3 text says "They look at each other" -> Describe BOTH Nina and Leo
- If a character is NOT in the current scene's raw_text -> DO NOT include them
- Example for story "Nina stands in snow. She meets Leo in crowd.":
  - Panel 1: "[Nina_001], woman, dark brown hair, red coat, black boots, stands alone in snow, cold winter atmosphere."
  - Panel 2: "[Nina_001], woman, same red coat and black boots, stands in crowd, turns to look, and [Leo_002], man, brown hair, green jacket, approaches from right, both greeting each other."
  - Panel 3: "[Nina_001], woman, same red coat and black boots, and [Leo_002], man, same green jacket, stand facing each other, quiet moment, eye contact."
- NEVER fabricate characters that don't appear in the text

# ============================
# COMMON RULES (ALL SCENES)
# ============================

## Outfit Consistency
- Panel 1: FULL outfit description with colors
- Panel 2+: "wearing the same [color] [garment type]" - use EXACT color words from Panel 1
- NEVER change colors between panels

## Pose & Action Specificity
- Use VERBS in present continuous (-ing form)
- Include body part movements: "arm raised", "head turned", "legs apart"
- Include facial expression hints: "with a smile", "expression of surprise", "gazing intently"

## Environment & Lighting
- Panel 1: Rich environment description
- Panel 2+: Focus on character, mention environment changes

## CLIP Token Limit (CRITICAL)
- SDXL CLIP encoder limit: 77 tokens
- Keep prompts under 60 words per panel
- Put CRITICAL details FIRST: character ID, outfit, main action, pose
- Put LESS critical details LAST: background, lighting
- For multi-character: prioritize character descriptions over environment
- ABBREVIATE when possible: "blue sky" instead of "clear blue sky with fluffy white clouds"

## Forbidden Terms
- NO: "masterpiece", "8k", "ultra-realistic", "trending", "best quality"
- NO: Generic terms like "the girl/man" - use IDs only
- NO: Conflicting outfit descriptions

# ============================
# OUTPUT SCHEMA (STRICT JSON)
# ============================

For SINGLE CHARACTER story (keep under 50 words per prompt):
{
  "story_id": "01",
  "panels": [
    {
      "index": 1,
      "raw_text": "<Lily> makes breakfast in the kitchen.",
      "expanded_prompt": "[Lily_001], woman, blonde ponytail, white blouse, blue jeans, reaches for pan at counter, cracks egg, kitchen, morning light.",
      "negative_prompt": "blurry, distorted, deformed, extra limbs, floating, wrong pose, stiff, static, wrong outfit, wrong hair, duplicate character",
      "reference_image": null
    },
    {
      "index": 2,
      "raw_text": "She looks out the window quietly.",
      "expanded_prompt": "[Lily_001], woman, same white blouse and blue jeans, stands by window, head turned, gazes outside, quiet expression, hand on sill.",
      "negative_prompt": "blurry, distorted, deformed, extra limbs, floating, wrong pose, stiff, static, wrong outfit, wrong hair, duplicate character",
      "reference_image": "results/01/panel_1.png"
    }
  ]
}

For MULTI-CHARACTER story (keep under 55 words per prompt):
{
  "story_id": "06",
  "panels": [
    {
      "index": 1,
      "raw_text": "<Jack> and <Sara> sit in a park and talk.",
      "expanded_prompt": "[Jack_001], man, brown hair, beard, blue shirt, khaki shorts, sits on left bench, and [Sara_002], woman, blonde wavy hair, white blouse, denim jeans, sits on right bench, talking, park, blue sky.",
      "negative_prompt": "blurry, distorted, deformed, extra limbs, floating, wrong pose, stiff, static, wrong outfit, wrong hair, character blending, merged, Siamese twins, swapped faces, three people, solo character",
      "reference_image": null
    },
    {
      "index": 2,
      "raw_text": "They continue talking in a cafe.",
      "expanded_prompt": "[Jack_001], man, same blue shirt and khaki shorts, sits at cafe table on left, leans forward, and [Sara_002], woman, same white blouse and denim jeans, sits across on right, holds coffee cup, talking, cafe interior.",
      "negative_prompt": "blurry, distorted, deformed, extra limbs, floating, wrong pose, stiff, static, wrong outfit, wrong hair, character blending, merged, Siamese twins, swapped faces, three people, solo character",
      "reference_image": "results/06/panel_1.png"
    }
  ]
}
"""

def build_user_prompt(default_subject: str, scenes: list) -> str:
    """Construct User Prompt with enhanced multi-character and action-specific guidance"""
    scene_blocks = []
    for s in scenes:
        scene_blocks.append(f"[SCENE-{s['index']}] {s['text']}")
    
    scenes_text = "\n".join(scene_blocks)
    
    # Detect multi-character
    all_text = " ".join([s['text'] for s in scenes])
    names = re.findall(r'<(\w+)>', all_text)
    unique_names = list(dict.fromkeys(names))  # Preserve order, remove duplicates
    is_multi = len(unique_names) > 1
    char_list = ", ".join([f"<{n}>" for n in unique_names])
    
    scene_type_hint = "MULTI-CHARACTER" if is_multi else "SINGLE-CHARACTER"
    
    return f"""
Input Story Sequence:
{scenes_text}

## Scene Analysis
Characters detected: {char_list}
Scene type: {scene_type_hint}
Number of scenes: {len(scenes)}
IMPORTANT: Generate EXACTLY {len(scenes)} panels, one for each scene. Do NOT merge scenes.

## Processing Instructions

1. FIRST determine scene type: SINGLE or MULTI-CHARACTER
2. For MULTI-CHARACTER: Use LEFT-RIGHT positioning, describe BOTH characters fully in Panel 1
3. For SINGLE-CHARACTER: Focus on precise pose and action description
4. CRITICAL: Generate EXACTLY {len(scenes)} panels - one panel per scene
5. CRITICAL: Use specific action verbs and body part movements in present continuous form
6. CRITICAL: Use scenario-specific negative prompts (see system prompt)
7. Outfit colors MUST be identical across all panels
8. story_id MUST match the input file number
9. Output pure JSON matching the schema. No commentary outside JSON.
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
