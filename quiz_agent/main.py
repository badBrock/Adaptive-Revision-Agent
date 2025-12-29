"""
Simple quiz agent entry point
"""

import sys
from graph import build_graph
from utils import validate_input_files

# Suppress NLTK messages
import logging
logging.getLogger('nltk').setLevel(logging.ERROR)

import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass


def main():
    print("=" * 60)
    print("🎓 ADAPTIVE QUIZ AGENT")
    print("=" * 60)
    
    print("\n📁 Enter file/directory path:")
    user_input = input(">>> ").strip()
    
    if not user_input:
        print("❌ No path provided")
        sys.exit(1)
    
    md_paths, img_paths = validate_input_files([user_input])
    
    if not md_paths and not img_paths:
        print("❌ No files found")
        sys.exit(1)
    
    print(f"✓ Loaded {len(md_paths)} markdown, {len(img_paths)} images")
    
    app = build_graph()
    
    initial_state = {
        "user_id": "user_001",
        "session_id": "",
        "system_prompt": "",
        "md_paths": md_paths,
        "image_paths": img_paths,
        "extracted_keywords": [],
        "raw_content": "",
        "image_descriptions": [],
        "topic_id": "",
        "topic_name": "",
        "current_mastery": 0.0,
        "question_text": "",
        "user_answer": None,
        "difficulty_level": ""
    }
    
    result = app.invoke(initial_state)
    
    print(f"\n💾 Session: {result['session_id']}")


if __name__ == "__main__":
    main()
