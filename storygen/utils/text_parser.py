"""
Text Parser Utility Functions
Helpers for parsing story scripts and extracting structured information
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParsedScene:
    """Structured scene data"""
    scene_id: int
    content: str
    characters: List[str]


def parse_script_scenes(script_text: str) -> List[ParsedScene]:
    """
    Parse raw script text to extract scenes

    Args:
        script_text: Raw script content with [SCENE-N] markers

    Returns:
        List of ParsedScene objects
    """
    scene_pattern = r'\[SCENE-(\d+)\]\s*(.*?)(?=\[SEP\]|\Z)'
    matches = re.findall(scene_pattern, script_text, re.DOTALL | re.IGNORECASE)

    scenes = []
    for scene_id_str, content in matches:
        scene_id = int(scene_id_str)
        clean_content = content.strip()
        char_pattern = r'<([^>]+)>'
        characters = re.findall(char_pattern, clean_content)
        scenes.append(ParsedScene(
            scene_id=scene_id,
            content=clean_content,
            characters=characters
        ))
    return scenes


def extract_characters(script_text: str) -> List[str]:
    """Extract all unique character names from script"""
    char_pattern = r'<([^>]+)>'
    characters = re.findall(char_pattern, script_text)
    seen = set()
    unique_chars = []
    for char in characters:
        if char not in seen:
            seen.add(char)
            unique_chars.append(char)
    return unique_chars


def extract_scene_metadata(scene_text: str) -> Dict[str, any]:
    """Extract metadata from a scene text"""
    metadata = {
        "time_of_day": "day",
        "setting": "",
        "actions": [],
        "emotions": []
    }

    time_indicators = {
        "morning": ["morning", "sunrise", "dawn", "early"],
        "afternoon": ["afternoon", "noon", "midday"],
        "evening": ["evening", "sunset", "dusk"],
        "night": ["night", "dark", "midnight", "stars"]
    }

    for time, indicators in time_indicators.items():
        if any(ind in scene_text.lower() for ind in indicators):
            metadata["time_of_day"] = time
            break

    action_verbs = [
        "walk", "run", "sit", "stand", "look", "see", "watch",
        "eat", "drink", "read", "write", "talk", "speak",
        "laugh", "smile", "cry", "think", "dream"
    ]

    for verb in action_verbs:
        if verb in scene_text.lower():
            metadata["actions"].append(verb)

    emotions = [
        "happy", "sad", "angry", "surprised", "scared",
        "excited", "calm", "nervous", "curious", "tired"
    ]

    for emotion in emotions:
        if emotion in scene_text.lower():
            metadata["emotions"].append(emotion)

    return metadata


def clean_script_text(text: str) -> str:
    """Clean and normalize script text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()
