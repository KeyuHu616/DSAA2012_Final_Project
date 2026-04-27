"""
Test script for the story generation pipeline
"""

import sys
sys.path.insert(0, '.')

from storygen.script_director.llm_parser import LLMScriptParser
from storygen.script_director.prompt_enhancer import PromptEnhancer
from storygen.utils.text_parser import parse_script_scenes, extract_characters

# Test with a sample script
parser = LLMScriptParser(llm_backend='local', model_name='llama3:70b')
enhancer = PromptEnhancer()

# Parse a sample script
test_file = 'data/TaskA/06.txt'

print("=" * 60)
print("Testing Narrative Weaver Pro Pipeline")
print("=" * 60)

try:
    board = parser.process_script_file(test_file)
    print(f"\nStory ID: {board.story_id}")
    print(f"Characters: {list(board.characters.keys())}")
    print(f"Panels: {len(board.panels)}")
    print(f"Global Style: {board.global_style}")

    # Test prompt enhancer
    prompts = enhancer.process_entire_story(board)
    print(f"\nGenerated {len(prompts)} enhanced prompts")

    for p in prompts[:2]:
        print(f"\n  Panel {p['panel_id']}:")
        print(f"    Shot: {p['shot_type']}")
        print(f"    Prompt: {p['prompt'][:100]}...")

    print("\n" + "=" * 60)
    print("Basic tests PASSED")
    print("=" * 60)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
