"""
LLM Script Parser - Narrative director using large language models
Parses story scripts into structured production boards for visual generation
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class Character:
    """Character information data class"""
    name: str
    visual_description: str
    token: str
    key_attributes: List[str] = field(default_factory=list)
    clothing: str = ""
    appearance_details: str = ""


@dataclass
class Panel:
    """Story panel/scene data class"""
    panel_id: int
    raw_prompt: str
    enhanced_prompt: str = ""
    shot_type: str = "medium"
    camera_movement: str = "static"
    lighting_mood: str = "natural"
    composition: str = ""
    key_actions: List[str] = field(default_factory=list)
    interactions: List[Dict[str, str]] = field(default_factory=list)
    setting: str = ""
    time_of_day: str = "day"
    weather: str = ""

    def __post_init__(self):
        if self.key_actions is None:
            self.key_actions = []
        if self.interactions is None:
            self.interactions = []


@dataclass
class ProductionBoard:
    """Complete story production blueprint"""
    story_id: str
    characters: Dict[str, Character]
    panels: List[Panel]
    global_style: str
    consistency_constraints: List[str] = field(default_factory=list)
    narrative_arc: str = "linear"


class LLMScriptParser:
    """
    LLM-based script parser for narrative planning

    This module uses large language models to analyze story scripts and generate
    structured production boards containing character descriptions, scene plans,
    and visual instructions for downstream image generation.
    """

    SYSTEM_PROMPT = """You are a professional film director and storyboard artist.
Your task is to parse story scripts into detailed 'production blueprints' for multi-image generation systems.

Core Capabilities:
1. Visual Language Translation: Convert text descriptions into specific visual elements
2. Character Consistency Planning: Ensure consistent appearance across scenes
3. Narrative Rhythm Control: Enhance storytelling through visual composition

Output Format:
Strict JSON format only, with the following structure. Do not include any explanations or markdown markers."""

    USER_PROMPT_TEMPLATE = """
Please deeply analyze the following story script and output a complete production blueprint JSON:

## Story Script:
```
{script_text}
```

## Analysis Requirements:

### 1. Character Analysis (characters)
Create detailed profiles for each character marked with <name>:
- `visual_description`: Detailed appearance description (100+ chars), including age range, hairstyle, facial features, build
- `token`: Format "sks {{name}}" as unique identifier
- `key_attributes`: 3-5 most distinctive features (e.g., "red scarf", "round glasses")
- `clothing`: Scene-inferred clothing, maintainable across scenes
- `appearance_details`: Specific details (hair color, eye color, skin tone)

### 2. Panel/Scene Planning (panels)
Create detailed storyboards for each [SCENE]:
- `enhanced_prompt`: Expand original text into 150-200 char English image prompt including:
  * Specific visual representation of main actions
  * Environment details (background objects, spatial layout)
  * Lighting effects (light source direction, color temperature, intensity)
  * Atmosphere keywords
  * Quality enhancement words (masterpiece, best quality, highly detailed)

- `shot_type`: Choose most suitable shot type:
  * "extreme_closeup" - Extreme close-up (emotion, details)
  * "closeup" - Close-up (facial expression, upper body)
  * "medium" - Medium shot (half body + partial environment)
  * "wide" - Wide shot (full body + complete environment)
  * "over_shoulder" - Over-shoulder (dialogue scenes)
  * "establishing" - Establishing shot (new scene introduction)

- `camera_movement`: Camera movement type
  * "static" / "slow_push_in" / "pull_back" / "pan_left_right" / "tracking"

- `lighting_mood`: Lighting atmosphere description
- `key_actions`: Decompose into 2-4 specific visualizable actions
- `interactions`: Record interactions with other characters/objects
- `setting`: Detailed scene description
- `time_of_day` and `weather`: Inferred time and weather

### 3. Global Style (global_style)
Define overall visual style:
- "warm_cinematic_lifestyle" - Warm cinematic drama
- "urban_drama" - Urban drama
- "whimsical_illustration" - Whimsical illustration style
- "photorealistic_documentary" - Photorealistic documentary style

### 4. Consistency Constraints (consistency_constraints)
List visual elements that must be strictly maintained across all frames.
"""

    def __init__(self, llm_backend: str = "local", model_name: str = None):
        """
        Initialize the parser

        Args:
            llm_backend: LLM backend type ("local" | "api_openai" | "api_claude")
            model_name: Model name (auto-selected if None)
        """
        self.llm_backend = llm_backend
        self.model_name = model_name or self._get_default_model()
        self.client = None
        self._initialize_client()

    def _get_default_model(self) -> str:
        """Get default model based on backend"""
        defaults = {
            "local": "llama3:70b",
            "api_openai": "gpt-4o",
            "api_claude": "claude-3-5-sonnet-20241022"
        }
        return defaults.get(self.llm_backend, "gpt-4o")

    def _initialize_client(self):
        """Initialize LLM client based on backend type"""
        if self.llm_backend == "local":
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                )
            except ImportError:
                print("[Parser] Warning: openai package not found, LLM calls will fail")
                self.client = None
        elif self.llm_backend == "api_openai":
            import openai
            self.client = openai.OpenAI()
        elif self.llm_backend == "api_claude":
            import anthropic
            self.client = anthropic.Anthropic()

    def parse_raw_script(self, script_text: str) -> Dict[str, List[str]]:
        """
        Parse raw script text to extract scenes and characters

        Args:
            script_text: Raw script content

        Returns:
            dict: {"scenes": [...], "characters": [...], "raw_text": ...}
        """
        # Extract scenes using regex
        scene_pattern = r'\[SCENE-(\d+)\]\s*(.*?)(?=\[SEP\]|\Z)'
        scenes_raw = re.findall(scene_pattern, script_text, re.DOTALL | re.IGNORECASE)

        scenes = []
        characters_found = set()

        for scene_id, content in scenes_raw:
            clean_content = content.strip()
            scenes.append({
                "id": int(scene_id),
                "content": clean_content
            })

            # Extract character names marked with <name>
            char_pattern = r'<([^>]+)>'
            chars_in_scene = re.findall(char_pattern, clean_content)
            characters_found.update(chars_in_scene)

        return {
            "scenes": scenes,
            "characters": list(characters_found),
            "raw_text": script_text
        }

    def call_llm_for_analysis(self, parsed_script: Dict) -> str:
        """
        Call LLM for deep script analysis

        Returns:
            JSON string with analysis results
        """
        if self.client is None:
            # Fallback to rule-based parsing if no LLM available
            print("[Parser] Warning: No LLM client available, using rule-based fallback")
            return self._rule_based_parse(parsed_script)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            script_text=parsed_script["raw_text"]
        )

        if self.llm_backend in ["local", "api_openai"]:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Parser] LLM call failed: {e}, falling back to rule-based parsing")
                return self._rule_based_parse(parsed_script)

        elif self.llm_backend == "api_claude":
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return message.content[0].text
            except Exception as e:
                print(f"[Parser] LLM call failed: {e}, falling back to rule-based parsing")
                return self._rule_based_parse(parsed_script)

    def _rule_based_parse(self, parsed_script: Dict) -> str:
        """
        Fallback rule-based parsing when LLM is unavailable

        This creates a basic production board using pattern matching
        """
        characters = {}
        for char_name in parsed_script["characters"]:
            characters[char_name] = {
                "visual_description": f"A person named {char_name}",
                "token": f"sks {char_name}",
                "key_attributes": [],
                "clothing": "casual clothing",
                "appearance_details": "generic appearance"
            }

        panels = []
        for i, scene in enumerate(parsed_script["scenes"], 1):
            panels.append({
                "panel_id": i,
                "raw_prompt": scene['content'],
                "enhanced_prompt": f"{scene['content']}, high quality, detailed",
                "shot_type": "medium",
                "camera_movement": "static",
                "lighting_mood": "natural",
                "key_actions": [],
                "interactions": [],
                "setting": scene['content'][:100],
                "time_of_day": "day"
            })

        result = {
            "characters": characters,
            "panels": panels,
            "global_style": "cinematic_realistic",
            "consistency_constraints": list(parsed_script["characters"]),
            "narrative_arc": "linear"
        }

        return json.dumps(result)

    def parse_llm_response(self, llm_output: str, raw_text: str = "", scenes: List = None) -> ProductionBoard:
        """
        Parse LLM output to build ProductionBoard object

        Args:
            llm_output: JSON string from LLM
            raw_text: Original raw text for fallback
            scenes: List of parsed scenes from initial parsing

        Returns:
            ProductionBoard: Structured production blueprint
        """
        if scenes is None:
            scenes = []
        
        # Clean potential markdown markers
        cleaned = llm_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'```$', '', cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[Parser] Warning: Failed to parse LLM JSON ({e}), using rule-based fallback")
            fallback_data = json.loads(self._rule_based_parse({"characters": [], "scenes": [], "raw_text": raw_text}))
            data = fallback_data

        # Build character dictionary
        # Handle both dict format and list format from LLM
        # Also use original parsed character names for proper matching
        characters = {}
        char_list = data.get("characters", {})
        
        # Get original character names from scenes for proper matching
        original_names = []
        if scenes:
            for scene in scenes:
                content = scene.get("content", "")
                import re
                found = re.findall(r'<([^>]+)>', content)
                original_names.extend(found)
            original_names = list(dict.fromkeys(original_names))  # Remove duplicates, preserve order
        
        if isinstance(char_list, list):
            # LLM returned characters as a list
            # Match LLM characters to original names by trying to find matching tokens
            used_original_names = []
            
            for idx, char_data in enumerate(char_list):
                if isinstance(char_data, dict):
                    name = char_data.get("name", "")
                    
                    if name:
                        # LLM provided a name - try to match case with original
                        name_lower = name.lower()
                        matched_orig = None
                        for orig_name in original_names:
                            if orig_name.lower() == name_lower:
                                matched_orig = orig_name
                                used_original_names.append(orig_name)
                                break
                        
                        # Use original name if found, otherwise keep LLM's name
                        if matched_orig:
                            name = matched_orig
                    else:
                        # No name provided, try to extract from token
                        token = char_data.get("token", "")
                        # Handle "sks Name" or "sks_name" format (case-insensitive)
                        if "sks" in token.lower():
                            import re
                            match = re.search(r'sks[\s_]*(.+)', token, re.IGNORECASE)
                            if match:
                                token_name = match.group(1).strip()
                                # Try to match with original names (case-insensitive)
                                for orig_name in original_names:
                                    if orig_name.lower() == token_name.lower() and orig_name not in used_original_names:
                                        name = orig_name
                                        used_original_names.append(orig_name)
                                        break
                                if not name:
                                    # Use token name capitalized
                                    name = token_name.title()
                    
                    # If still no name, try to use original names in order
                    if not name and idx < len(original_names):
                        for orig_name in original_names:
                            if orig_name not in used_original_names:
                                name = orig_name
                                used_original_names.append(orig_name)
                                break
                    
                    if name:
                        characters[name] = Character(
                            name=name,
                            visual_description=char_data.get("visual_description", ""),
                            token=char_data.get("token", f"sks {name}"),
                            key_attributes=char_data.get("key_attributes", []),
                            clothing=char_data.get("clothing", ""),
                            appearance_details=char_data.get("appearance_details", "")
                        )
        else:
            # LLM returned characters as a dictionary
            # Keys might be token names like "sks Jack" - need to normalize
            used_original_names = []
            
            for raw_name, char_data in char_list.items():
                if isinstance(char_data, dict):
                    name = raw_name
                    
                    # If the key is a token like "sks Jack", try to match with original names
                    if "sks" in raw_name.lower():
                        # Extract name from token
                        import re
                        match = re.search(r'sks[\s_]*(.+)', raw_name, re.IGNORECASE)
                        if match:
                            token_name = match.group(1).strip()
                            # Try to match with original names
                            for orig_name in original_names:
                                if orig_name.lower() == token_name.lower() and orig_name not in used_original_names:
                                    name = orig_name
                                    used_original_names.append(orig_name)
                                    break
                            if name == raw_name:  # No match found
                                name = token_name.title()
                    
                    characters[name] = Character(
                        name=name,
                        visual_description=char_data.get("visual_description", ""),
                        token=char_data.get("token", f"sks {name}"),
                        key_attributes=char_data.get("key_attributes", []),
                        clothing=char_data.get("clothing", ""),
                        appearance_details=char_data.get("appearance_details", "")
                    )

        # Build panel list - need to map LLM output to actual scene content
        panels = []
        for idx, panel_data in enumerate(data.get("panels", [])):
            if isinstance(panel_data, dict):
                # Get raw_prompt from actual scene content
                raw_prompt = f"Scene {idx + 1}"
                if idx < len(scenes):
                    raw_prompt = scenes[idx].get('content', raw_prompt)
                elif panel_data.get('key_actions'):
                    # Try to construct from key_actions
                    actions = panel_data.get('key_actions', [])
                    if actions:
                        raw_prompt = actions[0] if isinstance(actions[0], str) else str(actions[0])
                
                panel_data_with_defaults = {
                    'panel_id': panel_data.get('panel_id', idx + 1),
                    'raw_prompt': panel_data.get('raw_prompt', raw_prompt),
                    'enhanced_prompt': panel_data.get('enhanced_prompt', ''),
                    'shot_type': panel_data.get('shot_type', 'medium'),
                    'camera_movement': panel_data.get('camera_movement', 'static'),
                    'lighting_mood': panel_data.get('lighting_mood', 'natural'),
                    'key_actions': panel_data.get('key_actions', []),
                    'interactions': panel_data.get('interactions', []),
                    'setting': panel_data.get('setting', ''),
                    'time_of_day': panel_data.get('time_of_day', 'daytime'),
                    'weather': panel_data.get('weather', 'clear')
                }
                panels.append(Panel(**panel_data_with_defaults))

        # Build complete ProductionBoard
        board = ProductionBoard(
            story_id=f"story_{hash(data.get('global_style', '')) % 10000}",
            characters=characters,
            panels=panels,
            global_style=data.get("global_style", "cinematic_realistic"),
            consistency_constraints=data.get("consistency_constraints", []),
            narrative_arc=data.get("narrative_arc", "linear")
        )

        return board

    def process_script_file(self, file_path: str) -> ProductionBoard:
        """
        Main entry point for processing a single script file

        Args:
            file_path: Path to script file

        Returns:
            ProductionBoard: Complete production blueprint
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            script_text = f.read().strip()

        print(f"[Director] Parsing script: {Path(file_path).name}")

        # Initial parsing
        parsed = self.parse_raw_script(script_text)
        print(f"[Director] Found {len(parsed['scenes'])} scenes, characters: {parsed['characters']}")

        # LLM analysis
        print(f"[Director] Calling {self.model_name} for deep analysis...")
        llm_output = self.call_llm_for_analysis(parsed)

        # Build structured output
        production_board = self.parse_llm_response(llm_output, script_text, parsed.get('scenes', []))

        print(f"[Director] Parsing complete! Style: {production_board.global_style}")

        return production_board

    def save_production_board(self, board: ProductionBoard, output_path: str):
        """Save ProductionBoard to JSON file"""
        output_data = {
            "story_id": board.story_id,
            "characters": {k: asdict(v) for k, v in board.characters.items()},
            "panels": [asdict(p) for p in board.panels],
            "global_style": board.global_style,
            "consistency_constraints": board.consistency_constraints,
            "narrative_arc": board.narrative_arc
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test with sample script
    parser = LLMScriptParser(llm_backend="local", model_name="llama3:70b")
    test_file = "/home/KeyuHu/code/DSAA2012FinalNew/data/TaskA/06.txt"

    try:
        board = parser.process_script_file(test_file)
        print(f"Story ID: {board.story_id}")
        print(f"Characters: {list(board.characters.keys())}")
        print(f"Panels: {len(board.panels)}")
        print(f"Global Style: {board.global_style}")
    except FileNotFoundError:
        print(f"[Test] Sample file not found: {test_file}")
