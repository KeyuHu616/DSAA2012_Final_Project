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
    key_objects: str = ""  # Key objects like breakfast, book, etc.

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

## CRITICAL RULES - MUST FOLLOW:

### 1. STORY CONTEXT UNDERSTANDING (MOST IMPORTANT)
- pronouns like "He", "She", "They", "It" ALWAYS refer to characters/objects from PREVIOUS panels
- "He pauses at the door" → Understand WHICH door from context (door of current location)
- "She looks around" → She is in the SAME location as previous panel unless stated otherwise
- "It chases a ball" → "It" is the SAME animal/object from previous panels
- enhanced_prompt MUST include the setting/objects from previous panels when using pronouns
- DO NOT assume stories are about buses or trains unless explicitly stated

### 2. TIMELINE CONSISTENCY
- If the story takes place in the MORNING (breakfast, morning routine), ALL panels MUST have `time_of_day: "morning"`
- If the story takes place in the EVENING (dinner, evening routine), ALL panels MUST have `time_of_day: "evening"` or `"night"`
- NEVER change time_of_day between panels unless explicitly stated

### 3. SETTING CONSISTENCY
- If the story starts in a KITCHEN, subsequent panels should be KITCHEN or dining area
- If the story starts in a PARK, subsequent panels should remain in outdoor park settings
- Setting should flow naturally from the previous panel unless explicitly changed

### 4. VISUAL DESCRIPTION RULES
- visual_description MUST include SPECIFIC clothing (e.g., "blue button-up shirt", NOT "casual outfit")
- Include: hairstyle, hair color, eye color, build, skin tone, EXACT clothing description
- clothing field should match what's in visual_description

### 5. KEY OBJECTS CONSISTENCY
- Track key objects (book, ball, food, toys) across ALL panels
- If an object appears in Panel 1, it should appear/remain relevant in subsequent panels

### 6. CHARACTER COUNT
- Describe characters clearly, but allow flexibility for stories with multiple characters joining

Output Format:
Strict JSON format only, with the following structure. Do not include any explanations or markdown markers."""

    USER_PROMPT_TEMPLATE = """
Please deeply analyze the following story script and output a complete production blueprint JSON.

## Story Script:
```
{script_text}
```

## CRITICAL ANALYSIS:

### STEP 1: Identify Story Context
- What is the TIME OF DAY? (morning, afternoon, evening, night)
- Where does the story TAKE PLACE? (kitchen, office, street, etc.)
- What KEY OBJECTS appear? (breakfast food, book, coffee, toys, etc.)
- IMPORTANT: Track WHERE the story starts - this is the base setting for ALL panels!

### STEP 2: Pronoun Resolution (CRITICAL!)
When parsing scenes with pronouns (He/She/They/It):
- "He pauses at the door" → Understand context - what door? (current location's door)
- "She sits by the window" → Same room/location from previous panel
- "It chases a ball" → Same animal/object from previous panels
- enhanced_prompt MUST include the resolved context!

Example:
```
Script: [SCENE-1] <Ryan> walks in the park.
Script: [SCENE-2] He pauses at the door.
WRONG: "Ryan pauses at the door"
RIGHT: "Ryan pauses at the park entrance door"
```

### STEP 3: Character Analysis (characters)
Create detailed profiles for each character marked with <name>:
- `visual_description`: Detailed appearance (100+ chars), SPECIFIC clothing (e.g., "blue button-up shirt", NOT "casual outfit")
- `token`: Format "sks {{name}}" as unique identifier
- `key_attributes`: 3-5 most distinctive features (e.g., "short brown hair", "round glasses", "blue eyes")
- `clothing`: SPECIFIC clothing matching visual_description (e.g., "blue shirt and jeans")
- `appearance_details`: Specific details (hair color, eye color, skin tone)

### STEP 4: Panel/Scene Planning (panels)
CRITICAL: Setting and time MUST be consistent!

For each [SCENE]:
- `enhanced_prompt`: 150-200 char prompt including:
  * Character description (START with character name)
  * Main action (with resolved context for pronouns)
  * Setting (SAME as previous panel unless stated otherwise)
  * Key objects (SAME as previous panel)
  * Lighting matching time_of_day
  * "photorealistic, realistic photography, sharp focus, 8k detailed"
  
  **Example for morning routine story:**
  "Lily, a young woman with auburn hair and glasses, sits at kitchen table eating breakfast. Modern kitchen interior, morning sunlight through window. photorealistic..."

- `shot_type`: "extreme_closeup" / "closeup" / "medium" / "wide" / "over_shoulder" / "establishing"
- `camera_movement`: "static" / "slow_push_in" / "pull_back" / "pan_left_right" / "tracking"
- `lighting_mood`: MUST match time_of_day (morning = warm sunlight, evening = soft lamp light)
- `key_actions`: 2-4 specific visualizable actions
- `setting`: Detailed scene description (SAME as first panel unless changed!)
- `time_of_day`: SAME for ALL panels in same-time stories!
- `weather`: clear/rainy/snowy/cloudy
- `key_objects`: Track these across ALL panels (bus, food, book, etc.)

### STEP 5: Global Style (global_style)
Choose ONE style, MUST produce photorealistic images:
- "warm_cinematic_lifestyle" - Warm cinematic drama
- "urban_drama" - Urban drama
- "photorealistic_documentary" - Photorealistic documentary
- "cinematic_realistic" - Cinematic realistic

### STEP 6: Consistency Constraints (consistency_constraints)
List elements that must REMAIN CONSISTENT across ALL frames:
- Character appearance (hair color, clothing, features)
- Setting location (bus stop → bus interior, etc.)
- Time of day (ALL panels same time)
- Lighting style
- Key objects (food, book, toys, etc.)

## OUTPUT EXAMPLE:
```json
{{
  "characters": {{...}},
  "panels": [
    {{
      "enhanced_prompt": "Lily, young woman with auburn hair, sitting at kitchen table eating breakfast. Modern kitchen interior, morning sunlight through window. photorealistic...",
      "time_of_day": "morning",
      "setting": "Modern kitchen with breakfast table",
      "key_objects": "breakfast food, coffee cup"
    }}
  ],
  "consistency_constraints": [
    "All panels must be morning time with warm sunlight",
    "All panels must be in or near kitchen setting"
  ]
}}
```
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

    def _infer_gender_fallback(self, name: str) -> str:
        """Simple gender inference for fallback cases"""
        name_lower = name.lower()
        female_markers = {'girl', 'woman', 'female', 'lady', 'she', 'her', 'mom', 'nina', 'emma', 'sara', 'lily', 'olivia', 'rose'}
        male_markers = {'boy', 'man', 'male', 'he', 'his', 'dad', 'tom', 'jack', 'ben', 'leo', 'john', 'mike'}
        
        for marker in female_markers:
            if marker in name_lower:
                return "female"
        for marker in male_markers:
            if marker in name_lower:
                return "male"
        
        # Check name endings
        if name_lower.endswith(('a', 'e', 'i', 'y')):
            return "female"
        return "male"

    def _extract_char_from_raw_prompt(self, raw_prompt: str, char_names: List[str]) -> Optional[str]:
        """Extract character name from raw prompt like '<Lily> makes breakfast'"""
        import re
        # Match <Name> pattern
        match = re.search(r'<([A-Za-z]+)>', raw_prompt)
        if match:
            found_name = match.group(1)
            # Check if it's a known character
            for char_name in char_names:
                if char_name.lower() == found_name.lower():
                    return char_name
            return found_name
        return None

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
            used_original_lower = set()
            
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
                                used_original_lower.add(orig_name.lower())
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
                    # Use case-insensitive comparison to avoid duplicates
                    if not name and idx < len(original_names):
                        used_original_lower = {n.lower() for n in used_original_names}
                        for orig_name in original_names:
                            if orig_name.lower() not in used_original_lower:
                                name = orig_name
                                used_original_names.append(orig_name)
                                used_original_lower.add(orig_name.lower())
                                break
                    
                    if name:
                        characters[name] = Character(
                            name=name,
                            visual_description=char_data.get("visual_description", ""),
                            token=char_data.get("token", f"sks {name}").replace("_", " "),  # FIX: Replace underscore with space
                            key_attributes=char_data.get("key_attributes", []),
                            clothing=char_data.get("clothing", ""),
                            appearance_details=char_data.get("appearance_details", "")
                        )
        else:
            # LLM returned characters as a dictionary
            # Keys might be token names like "sks Jack" - need to normalize
            used_original_names = []
            used_original_lower = set()
            
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
                            # Try to match with original names (case-insensitive)
                            for orig_name in original_names:
                                if orig_name.lower() == token_name.lower() and orig_name.lower() not in used_original_lower:
                                    name = orig_name
                                    used_original_names.append(orig_name)
                                    used_original_lower.add(orig_name.lower())
                                    break
                            if name == raw_name:  # No match found
                                name = token_name.title()
                    
                    characters[name] = Character(
                        name=name,
                        visual_description=char_data.get("visual_description", ""),
                        token=char_data.get("token", f"sks {name}").replace("_", " "),  # FIX: Replace underscore with space
                        key_attributes=char_data.get("key_attributes", []),
                        clothing=char_data.get("clothing", ""),
                        appearance_details=char_data.get("appearance_details", "")
                    )
        
        # CRITICAL FIX: Check for MISSING characters that are in original_names but not in LLM output
        # LLM sometimes only returns some characters (e.g., Jack but not Sara in "<Jack> and <Sara>")
        # Use case-insensitive comparison to avoid duplicates like "milo" vs "Milo"
        existing_names_lower = {k.lower() for k in characters.keys()}
        missing_chars = [name for name in original_names if name.lower() not in existing_names_lower]
        
        if missing_chars:
            print(f"[Director] Warning: LLM missed characters: {missing_chars}. Creating placeholders.")
            for name in missing_chars:
                characters[name] = Character(
                    name=name,
                    visual_description="",  # Will be filled by the character enhancement logic below
                    token=f"sks {name.lower().replace(' ', '_')}",
                    key_attributes=[],
                    clothing="",
                    appearance_details=""
                )
        
        # Also handle case where LLM returned empty characters dict entirely
        if not characters and original_names:
            print(f"[Director] Warning: LLM returned empty characters. Creating from script: {original_names}")
            for name in original_names:
                characters[name] = Character(
                    name=name,
                    visual_description="",
                    token=f"sks {name.lower().replace(' ', '_')}",
                    key_attributes=[],
                    clothing="",
                    appearance_details=""
                )
        
        # CRITICAL FIX: Remove duplicate characters caused by case differences
        # If LLM returns both "milo" and "Milo", keep the one whose visual_description is actually USED in enhanced_prompts
        if len(characters) > len(set(k.lower() for k in characters.keys())):
            print(f"[Director] Warning: Found duplicate characters with different case. Deduplicating...")
            
            # Analyze panels to see which character's visual_description is actually USED in enhanced_prompts
            panels_data = data.get("panels", [])
            
            # Count based on visual_description matching in enhanced_prompts
            desc_usage = {}
            for key in characters.keys():
                char_info = characters[key]
                visual_desc = char_info.visual_description.lower() if char_info.visual_description else ""
                desc_usage[key] = 0
                
                for panel in panels_data:
                    ep = (panel.get("enhanced_prompt", "") or "").lower()
                    # Check if visual_description (first 50 chars to avoid full match issues) appears in enhanced_prompt
                    if visual_desc and len(visual_desc) > 20:
                        # Use first 40 chars of visual_description for matching
                        desc_prefix = visual_desc[:40]
                        if desc_prefix in ep:
                            desc_usage[key] += 1
            
            # Group keys by lowercase
            groups = {}
            for key in characters.keys():
                key_lower = key.lower()
                if key_lower not in groups:
                    groups[key_lower] = []
                groups[key_lower].append(key)
            
            # For each group, determine which to keep
            keys_to_remove = []
            for key_lower, keys in groups.items():
                if len(keys) > 1:
                    # Multiple versions of the same name
                    # PRIORITY: 
                    # 1. Use the one whose visual_description appears in enhanced_prompts
                    # 2. OR match original_names exactly
                    # 3. Or default to first
                    
                    keep_key = None
                    
                    # Find the one with most visual_description usage in enhanced_prompts
                    max_usage = -1
                    for key in keys:
                        usage = desc_usage.get(key, 0)
                        if usage > max_usage:
                            max_usage = usage
                            keep_key = key
                    
                    # If there's a clear winner by description usage, use it
                    if max_usage > 0:
                        # Found a clear winner by description usage
                        pass
                    else:
                        # No description usage - prefer original_names match
                        for key in keys:
                            if key in original_names:
                                keep_key = key
                                break
                    
                    if not keep_key:
                        keep_key = keys[0]  # Default to first
                    
                    print(f"  Keeping: {keep_key} (desc usage: {desc_usage.get(keep_key, 0)}, removing: {[k for k in keys if k != keep_key]})")
                    for key in keys:
                        if key != keep_key:
                            print(f"    Removing duplicate: {key}")
                            keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del characters[key]

        # Build panel list - need to map LLM output to actual scene content
        panels = []
        llm_panels = data.get("panels", [])
        num_scenes = len(scenes)
        num_llm_panels = len(llm_panels)
        
        # CRITICAL FIX: Ensure panel count matches scene count
        # If LLM output fewer panels than scenes, we need to fill in the gaps
        if num_llm_panels < num_scenes:
            print(f"[Director] Warning: LLM output {num_llm_panels} panels but script has {num_scenes} scenes. Filling gaps...")
        
        for idx in range(num_scenes):  # Always iterate over all scenes
            panel_data = llm_panels[idx] if idx < num_llm_panels else None
            
            # Get raw_prompt from actual scene content
            raw_prompt = scenes[idx].get('content', f"Scene {idx + 1}") if idx < len(scenes) else f"Scene {idx + 1}"
            
            # CRITICAL FIX: Ensure enhanced_prompt includes character description
            enhanced_prompt = ""
            shot_type = "medium"
            time_of_day = "daytime"
            setting = ""
            key_objects = ""
            
            if panel_data and isinstance(panel_data, dict):
                enhanced_prompt = panel_data.get('enhanced_prompt', '')
                shot_type = panel_data.get('shot_type', 'medium')
                time_of_day = panel_data.get('time_of_day', 'daytime')
                setting = panel_data.get('setting', '')
                key_objects = panel_data.get('key_objects', '')
                
                # CRITICAL FIX: If enhanced_prompt is too short or missing character, add character info
                if not enhanced_prompt or len(enhanced_prompt) < 30:
                    # Try to get from key_actions or construct
                    actions = panel_data.get('key_actions', [])
                    if actions:
                        enhanced_prompt = actions[0] if isinstance(actions[0], str) else str(actions[0])
                
                # If still empty, use raw_prompt
                if not enhanced_prompt:
                    enhanced_prompt = raw_prompt
                
                # CRITICAL FIX: Ensure character name appears in enhanced_prompt
                # First, try to extract character name from raw_prompt
                char_in_scene = self._extract_char_from_raw_prompt(raw_prompt, list(characters.keys()))
                
                # If no character in raw_prompt (environmental description like "city lights come on"),
                # use the first character from previous panels as the main character
                if not char_in_scene and characters and idx > 0:
                    # Get first character's name as default
                    first_char = list(characters.keys())[0]
                    char_in_scene = first_char
                
                if char_in_scene and char_in_scene not in enhanced_prompt[:50]:
                    # Prepend character name to enhanced_prompt
                    char_info = characters.get(char_in_scene)
                    if char_info and char_info.visual_description and "person" not in char_info.visual_description.lower():
                        enhanced_prompt = f"{char_in_scene}, {char_info.visual_description}, {enhanced_prompt}"
                    else:
                        enhanced_prompt = f"{char_in_scene}, {enhanced_prompt}"
            else:
                # Panel data missing - use raw_prompt
                char_in_scene = self._extract_char_from_raw_prompt(raw_prompt, list(characters.keys()))
                if char_in_scene:
                    char_info = characters.get(char_in_scene)
                    if char_info:
                        enhanced_prompt = f"{char_in_scene}, {char_info.visual_description}, {raw_prompt}"
                    else:
                        enhanced_prompt = f"{char_in_scene}, {raw_prompt}"
                else:
                    enhanced_prompt = raw_prompt
            
            # CRITICAL FIX: Infer better shot type based on panel content
            # - Environmental descriptions (lights, scenery) should NOT be close-ups
            # - Panels with pronouns and no named character might be reactions
            raw_lower = raw_prompt.lower()
            
            # Environmental/description-only panels should be wide or establishing
            env_keywords = ['lights', 'skyline', 'scenery', 'view', 'landscape', 'come on', 'begins', 'starts']
            is_env_panel = any(kw in raw_lower for kw in env_keywords) and not any(c.lower() in raw_lower for c in characters.keys())
            
            # If shot type is close-up/medium but panel is environmental, adjust to wide
            if is_env_panel and shot_type in ['closeup', 'medium']:
                shot_type = "wide"
            
            # If panel mentions "looks" or "smiles" or "reacts", use medium shot to show character
            reaction_keywords = ['looks', 'smile', 'laugh', 'wave', 'pause', 'stand', 'sit', 'walk']
            if any(kw in raw_lower for kw in reaction_keywords) and shot_type in ['wide', 'extreme_closeup']:
                shot_type = "medium"
            
            # CRITICAL FIX: If setting is empty but this is a continuation panel,
            # use the previous panel's setting for consistency
            if not setting and idx > 0 and panels:
                setting = panels[-1].setting if hasattr(panels[-1], 'setting') else ""
            
            panel_data_with_defaults = {
                'panel_id': idx + 1,
                'raw_prompt': raw_prompt,
                'enhanced_prompt': enhanced_prompt,
                'shot_type': shot_type,
                'camera_movement': panel_data.get('camera_movement', 'static') if panel_data else 'static',
                'lighting_mood': panel_data.get('lighting_mood', 'natural') if panel_data else 'natural',
                'key_actions': panel_data.get('key_actions', []) if panel_data else [],
                'interactions': panel_data.get('interactions', []) if panel_data else [],
                'setting': setting,
                'time_of_day': time_of_day,
                'weather': panel_data.get('weather', 'clear') if panel_data else 'clear',
                'key_objects': key_objects
            }
            panels.append(Panel(**panel_data_with_defaults))

        # CRITICAL FIX: Ensure all characters have complete descriptions
        # If any character has empty visual_description, key_attributes, etc., fill them in
        import random
        
        # Detect if this is an animal character
        animal_keywords = ['dog', 'cat', 'bird', 'rabbit', 'horse', 'lion', 'tiger', 'bear', 
                         'wolf', 'fox', 'deer', 'elephant', 'monkey', 'panda', 'koala',
                         'fish', 'owl', 'eagle', 'shark', 'duck', 'chicken', 'pig', 'cow',
                         'sheep', 'goat', 'snake', 'lizard', 'turtle', 'frog', 'puppy', 'kitten']
        
        for char_name, char_info in characters.items():
            # Consistent random for same name (moved inside loop)
            random.seed(hash(char_name) % 2**32)
            
            # Check if this is an animal character
            name_lower = char_name.lower()
            is_animal = any(animal in name_lower for animal in animal_keywords)
            
            # CRITICAL FIX: Handle animal characters differently from humans
            if is_animal:
                # Generate animal-specific description
                animal_type = char_name
                for animal_keyword in animal_keywords:
                    if animal_keyword in name_lower:
                        animal_type = animal_keyword
                        break
                
                animal_colors = ["golden brown", "black and white", "brown", "gray", "white", 
                              "orange and white", "black", "cream colored", "rust colored"]
                animal_features = ["fluffy fur", "smooth feathers", "shiny coat", "soft fur",
                                 "sleek body", "powerful build", "graceful posture"]
                animal_expressions = ["alert expression", "curious gaze", "friendly eyes",
                                    "happy demeanor", "calm presence", "playful stance"]
                
                color = random.choice(animal_colors)
                feature = random.choice(animal_features)
                expression = random.choice(animal_expressions)
                
                # Set animal-specific description (NO clothing for animals!)
                char_info.visual_description = f"A {color} {animal_type} with {feature}, {expression}, realistic animal proportions"
                char_info.key_attributes = [color, feature, expression, "realistic animal"]
                char_info.clothing = ""  # Animals don't wear clothes
                char_info.appearance_details = f"{animal_type} with {color} {feature}"
            else:
                # Human character - use existing logic
                # Infer gender and age from name
                gender = self._infer_gender_fallback(char_name)
                age = "young adult"
                
                hair_colors = ["black hair", "brown hair", "blonde hair", "dark brown hair", "auburn hair", "red hair"]
                hair_styles = ["short hair", "medium-length hair", "long hair", "wavy hair", "straight hair", "messy hair"]
                eye_colors = ["brown eyes", "blue eyes", "green eyes", "hazel eyes", "gray eyes"]
                skin_tones = ["fair skin", "medium skin tone", "olive skin", "tan skin"]
                builds = ["slim", "average", "athletic", "medium build"]
                
                # Check if visual_description is too generic or contradictory
                is_generic = "person" in char_info.visual_description.lower() if char_info.visual_description else True
                
                if not char_info.visual_description or is_generic:
                    # Generate specific visual description with actual features
                    hair = random.choice(hair_colors)
                    style = random.choice(hair_styles)
                    eyes = random.choice(eye_colors)
                    skin = random.choice(skin_tones)
                    build = random.choice(builds)
                    
                    if gender == "female":
                        clothing_options = [
                            "blue blouse and jeans",
                            "casual sweater and skirt",
                            "red dress",
                            "green top and white pants",
                            "yellow shirt and black jeans"
                        ]
                    else:
                        clothing_options = [
                            "blue shirt and jeans",
                            "casual t-shirt and pants",
                            "gray sweater and dark jeans",
                            "green jacket and khaki pants",
                            "white shirt and black pants"
                        ]
                    
                    clothing = random.choice(clothing_options)
                    
                    char_info.visual_description = (
                        f"A {age} {gender} with {hair}, {style}, {eyes}, {build} build, {skin}, wearing {clothing}"
                    )
                
                # CRITICAL FIX: Synchronize key_attributes with visual_description
                # If they contradict, trust visual_description as source of truth
                
                # First, fix visual_description if it's too vague
                vague_terms = ["casual outfit", "suitable for", "commute", "everyday", "typical", "simple"]
                is_vague = any(term in char_info.visual_description.lower() for term in vague_terms)
                
                if is_vague or not char_info.visual_description:
                    # Generate a more specific visual description
                    hair = random.choice(hair_colors)
                    style = random.choice(hair_styles)
                    eyes = random.choice(eye_colors)
                    skin = random.choice(skin_tones)
                    build = random.choice(builds)
                    
                    if gender == "female":
                        clothing = random.choice([
                            "blue blouse and dark jeans",
                            "casual white sweater and navy pants",
                            "red floral dress",
                            "green casual top and black skirt",
                            "yellow cardigan and jeans"
                        ])
                    else:
                        clothing = random.choice([
                            "blue button-up shirt and jeans",
                            "casual gray hoodie and dark pants",
                            "green casual jacket and khaki pants",
                            "white polo shirt and navy shorts",
                            "black t-shirt and jeans"
                        ])
                    
                    char_info.visual_description = (
                        f"A {age} {gender} with {hair}, {style}, {eyes}, {build} build, {skin}, wearing {clothing}"
                    )
                
                # Now extract key_attributes from the fixed visual_description
                # CRITICAL: Extract CLEAN feature phrases, not full sentences
                features = char_info.visual_description.split(",")
                
                hair_from_vd = None
                eyes_from_vd = None
                clothing_from_vd = None
                
                for f in features:
                    f_lower = f.lower().strip()
                    f_clean = f.strip()
                    
                    # Hair: look for specific hair descriptions
                    if 'hair' in f_lower and not hair_from_vd:
                        # Skip if it's the whole sentence (contains "is a" or starts with article)
                        if not any(v in f_lower for v in ['is a', 'with a', 'person', 'woman', 'man', 'boy', 'girl']):
                            hair_from_vd = f_clean
                        elif 'hair' in f_lower:
                            # Extract just the hair part
                            parts = f_clean.split()
                            hair_idx = None
                            for idx, w in enumerate(parts):
                                if 'hair' in w.lower():
                                    hair_idx = idx
                                    break
                            if hair_idx and hair_idx > 0:
                                # Get 2-3 words before hair
                                start = max(0, hair_idx - 2)
                                hair_from_vd = ' '.join(parts[start:hair_idx+1])
                    
                    # Eyes
                    if 'eyes' in f_lower and not eyes_from_vd:
                        if not any(v in f_lower for v in ['is a', 'with a']):
                            eyes_from_vd = f_clean
                    
                    # Clothing
                    if 'wearing' in f_lower and not clothing_from_vd:
                        clothing_from_vd = f_clean.replace('wearing ', '')
                
                # Build clean key_attributes
                new_attrs = []
                if hair_from_vd:
                    new_attrs.append(hair_from_vd)
                if eyes_from_vd:
                    new_attrs.append(eyes_from_vd)
                if clothing_from_vd:
                    new_attrs.append(clothing_from_vd)
                new_attrs.append("realistic proportions")
                
                char_info.key_attributes = new_attrs
                
                # Update clothing field
                if clothing_from_vd:
                    char_info.clothing = clothing_from_vd
                
                # If key_attributes is still empty or too few, generate specific attributes
                if not char_info.key_attributes or len(char_info.key_attributes) <= 2:
                    hair = hair_colors[random.randint(0, len(hair_colors)-1)]
                    style = hair_styles[random.randint(0, len(hair_styles)-1)]
                    eyes = eye_colors[random.randint(0, len(eye_colors)-1)]
                    char_info.key_attributes = [f"{hair}, {style}", eyes, "realistic proportions"]
                
                # If clothing is still empty or generic
                if not char_info.clothing or "clothing" in char_info.clothing.lower():
                    char_info.clothing = "casual comfortable clothing"
                
                # If appearance_details is empty, extract from visual_description
                if not char_info.appearance_details:
                    features = [part.strip() for part in char_info.visual_description.split(",")]
                    specific_features = [f for f in features if any(x in f.lower() for x in ["hair", "eyes", "skin", "build"])]
                    if specific_features:
                        char_info.appearance_details = ", ".join(specific_features[:3])
                    else:
                        char_info.appearance_details = f"{gender} appearance with natural features"

        # CRITICAL FIX: Enforce time_of_day consistency across all panels
        # Analyze all panels to determine the correct time setting
        time_keywords = {
            'night': ['night', 'darkness', 'moon', 'stars', 'evening', 'dusk', 'twilight', 'nighttime'],
            'morning': ['morning', 'sunrise', 'dawn', 'breakfast', 'early', 'a.m.'],
            'afternoon': ['afternoon', 'midday', 'noon', 'sunny day', 'daytime'],
            'evening': ['evening', 'sunset', 'dusk', 'golden hour', 'dinner']
        }
        
        # Count time_of_day occurrences
        time_counts = {}
        panel_times = []
        for panel in panels:
            tod = panel.time_of_day.lower() if panel.time_of_day else "daytime"
            panel_times.append(tod)
            time_counts[tod] = time_counts.get(tod, 0) + 1
        
        # Also analyze raw_prompts and enhanced_prompts for time hints
        for panel in panels:
            text = (panel.raw_prompt + " " + panel.enhanced_prompt).lower()
            for time_cat, keywords in time_keywords.items():
                if any(kw in text for kw in keywords):
                    # Boost count for this time category
                    time_counts[time_cat] = time_counts.get(time_cat, 0) + 0.5
        
        # Find the most common/indicated time
        if time_counts:
            dominant_time = max(time_counts, key=time_counts.get)
        else:
            dominant_time = "daytime"
        
        # Fix any panels that don't match the dominant time
        corrected_panels = []
        inconsistent_count = 0
        for panel in panels:
            panel_tod = panel.time_of_day.lower() if panel.time_of_day else "daytime"
            
            # Check if this panel's time is consistent with dominant time
            is_consistent = False
            
            # Direct match
            if panel_tod == dominant_time:
                is_consistent = True
            # Check if panel mentions dominant time keywords
            panel_text = (panel.raw_prompt + " " + panel.enhanced_prompt).lower()
            dominant_keywords = time_keywords.get(dominant_time, [])
            if any(kw in panel_text for kw in dominant_keywords):
                is_consistent = True
            # Check if panel time category matches dominant time category
            for time_cat, keywords in time_keywords.items():
                if panel_tod == time_cat and any(kw in dominant_time for kw in [time_cat]):
                    is_consistent = True
                    break
            
            if not is_consistent and dominant_time not in ['daytime', 'afternoon']:
                # Only correct obvious mismatches (e.g., night story shouldn't have daytime panels)
                if dominant_time in ['night', 'evening'] and panel_tod in ['daytime', 'afternoon', 'morning']:
                    panel.time_of_day = dominant_time
                    inconsistent_count += 1
                elif dominant_time == 'morning' and panel_tod in ['night', 'evening']:
                    panel.time_of_day = dominant_time
                    inconsistent_count += 1
        
        if inconsistent_count > 0:
            print(f"[Director] Fixed {inconsistent_count} panels with inconsistent time_of_day (set to '{dominant_time}')")

        # Build complete ProductionBoard
        # CRITICAL: Generate default consistency_constraints if empty
        consistency_constraints = data.get("consistency_constraints", [])
        
        # If consistency_constraints is empty or None, generate defaults from character data
        if not consistency_constraints:
            consistency_constraints = []
            for char_name, char_info in characters.items():
                if hasattr(char_info, 'appearance_details') and char_info.appearance_details:
                    consistency_constraints.append(f"{char_name}'s appearance details: {char_info.appearance_details}")
                if hasattr(char_info, 'clothing') and char_info.clothing:
                    consistency_constraints.append(f"{char_name}'s clothing: {char_info.clothing}")
        
        board = ProductionBoard(
            story_id=f"story_{hash(data.get('global_style', '')) % 10000}",
            characters=characters,
            panels=panels,
            global_style=data.get("global_style", "warm_cinematic_lifestyle"),  # Use defined style
            consistency_constraints=consistency_constraints,
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
