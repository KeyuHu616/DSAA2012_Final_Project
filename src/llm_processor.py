#!/usr/bin/env python3
"""
llm_processor.py
================
重构版：将多角色识别与一致性锚定完全交由LLM处理，废除代码级代词替换。
核心策略：
1. 保留输入中的所有代词与标签（<Lily>, She/Her），由LLM解析指代关系
2. 强制LLM在首帧定义角色ID（[Lily_001]），后续帧严格复用
3. 若出现新角色，要求LLM显式分配新ID并标记
4. 输出端保持路径注入与JSON Schema不变
"""

import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. 输入解析（废除代词替换）
# ============================================================

def parse_raw_input(raw_text: str) -> dict:
    """
    解析原始输入，保留所有代词与标签，不做任何文本替换。
    仅提取第一帧<Name>作为默认主体名（用于提示词构建，不用于修改文本）。
    """
    blocks = re.split(r'\[SEP\]', raw_text, flags=re.IGNORECASE)
    blocks = [b.strip() for b in blocks if b.strip()]

    scenes = []
    default_subject = None

    for i, block in enumerate(blocks):
        # 移除[SCENE-N]标记，保留原始文本（含<Lily>和She/Her）
        text = re.sub(r'\[SCENE-\d+\]\s*', '', block).strip()
        
        # 仅在第一帧尝试提取<Name>作为默认主体名（用于提示词上下文）
        if i == 0:
            name_match = re.search(r'<(\w+)>', text)
            if name_match:
                default_subject = name_match.group(1)
        
        scenes.append({"index": i + 1, "text": text})

    if default_subject is None:
        default_subject = "Subject"

    # 🔴 修复点1：保持向后兼容，同时返回两个键
    return {
        "subject_name": default_subject,  # 为pipeline_runner.py保持兼容
        "default_subject": default_subject,  # 新接口
        "scenes": scenes
    }

# ============================================================
# 2. 重构LLM Prompt（强制视觉展开 + 角色ID锚定 + 性别与着装一致性）
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
    """构造User Prompt，强制LLM展开视觉细节并锚定ID，明确性别与着装"""
    scene_blocks = []
    for s in scenes:
        # 保留原始文本结构，供LLM解析指代
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
# 3. LLM 加载与推理（保持不变）
# ============================================================

def load_llm(llm_path: str):
    print(f"📦 Loading LLM from {llm_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("✅ LLM loaded.")
    return tokenizer, model

def run_llm_inference(tokenizer, model, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 4096) -> str:
    """Qwen2.5 ChatML格式推理"""
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
# 4. JSON 解析与后处理（路径注入保持不变）
# ============================================================

def parse_llm_output(raw_output: str) -> dict:
    """从LLM输出中提取JSON"""
    # 移除可能的Markdown代码块
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()

    # 提取首个完整JSON对象
    brace_start = cleaned.find('{')
    brace_end = cleaned.rfind('}')
    if brace_start != -1 and brace_end != -1:
        json_str = cleaned[brace_start:brace_end + 1]
    else:
        json_str = cleaned

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败: {e}")
        with open("llm_debug.txt", "w", encoding="utf-8") as f:
            f.write(f"Raw output:\n{raw_output}\n\nCleaned:\n{cleaned}")
        print("原始输出已保存至 llm_debug.txt")
        return None

def derive_story_id(input_path: str) -> str:
    """从文件名提取数字ID"""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    m = re.search(r'\d+', stem)
    return m.group(0) if m else stem

def post_process(data: dict, story_id: str, results_root: str, raw_scenes: list) -> dict:
    """
    后处理：注入ID、路径，并强制对齐原始Raw Text（防LLM改写历史）
    """
    data["story_id"] = story_id
    panels = data.get("panels", [])
    raw_text_map = {s["index"]: s["text"] for s in raw_scenes}

    for p in panels:
        idx = p["index"]
        # 强制还原原始Raw Text（保持输入真实性）
        if idx in raw_text_map:
            p["raw_text"] = raw_text_map[idx]
        
        # 注入Reference路径（Panel 1无参考，Panel N参考N-1）
        if idx == 1:
            p["reference_image"] = None
        else:
            ref_path = os.path.join(results_root, story_id, f"panel_{idx-1}.png")
            p["reference_image"] = ref_path.replace("\\", "/")
            
    return data


# ============================================================
# 5. 主流程
# ============================================================

def process(input_path: str, output_path: str, llm_path: str, results_root: str = "results"):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"📖 解析输入: {input_path}")
    parsed = parse_raw_input(raw_text)
    story_id = derive_story_id(input_path)
    
    # 🔴 修复点2：输出日志时使用兼容的键名
    print(f"   StoryID: {story_id}, 主体: {parsed['subject_name']}, 共{len(parsed['scenes'])}帧")
    for s in parsed["scenes"]:
        print(f"     [SCENE-{s['index']}] {s['text']}")

    # 加载LLM
    tokenizer, model = load_llm(llm_path)
    
    # 构建提示词
    user_prompt = build_user_prompt(parsed["default_subject"], parsed["scenes"])
    print("\n🧠 Running LLM inference (角色ID锚定模式)...")
    
    raw_output = run_llm_inference(tokenizer, model, SYSTEM_PROMPT, user_prompt)
    data = parse_llm_output(raw_output)
    
    if data is None:
        print("❌ LLM输出解析失败，终止流程")
        return None

    # 后处理注入
    data = post_process(data, story_id, results_root, parsed["scenes"])
    
    # 保存JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n💾 已保存结构化JSON: {output_path}")
    return data

def main():
    parser = argparse.ArgumentParser(description="重构版LLM Processor：角色ID锚定")
    parser.add_argument("--input", type=str, required=True, help="输入.txt路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSON路径")
    parser.add_argument("--llm_path", type=str, default="./models/llm/Qwen2.5-7B-Instruct")
    parser.add_argument("--results_root", type=str, default="results", help="图片输出根目录")
    args = parser.parse_args()
    
    process(args.input, args.output, args.llm_path, args.results_root)

if __name__ == "__main__":
    main()