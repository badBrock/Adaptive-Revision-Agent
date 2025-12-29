"""
LangGraph node functions
"""

from datetime import datetime
from pathlib import Path
from models import QuizState
from config import KEYWORD_TOP_N, BASIC_THRESHOLD, INTERMEDIATE_THRESHOLD, SYSTEM_PROMPT

# Global instances (set by graph.py)
topic_kb = None
session_storage = None
keyword_extractor = None
image_descriptor = None
question_generator = None
answer_scorer = None


def initialize_session(state: QuizState) -> dict:
    """Create session"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "session_id": session_id,
        "system_prompt": state.get("system_prompt", SYSTEM_PROMPT)
    }


def extract_content(state: QuizState) -> dict:
    """Extract keywords and content"""
    md_paths = state.get("md_paths", [])
    img_paths = state.get("image_paths", [])
    
    all_content = []
    for md_path in md_paths:
        content = Path(md_path).read_text(encoding='utf-8')
        all_content.append(content)
    
    combined_content = "\n\n".join(all_content)
    
    keywords = keyword_extractor.extract(combined_content, top_n=KEYWORD_TOP_N) if combined_content else []
    
    image_descriptions = []
    for img_path in img_paths:
        desc = image_descriptor.describe_image(img_path)
        if desc:
            image_descriptions.append(desc)
            combined_content += f"\n\nImage: {desc}"
    
    return {
        "extracted_keywords": keywords,
        "raw_content": combined_content if combined_content else "Visual content",
        "image_descriptions": image_descriptions
    }


def identify_topic(state: QuizState) -> dict:
    """Identify or create topic"""
    keywords = state["extracted_keywords"]
    content = state["raw_content"]
    
    topic_id, topic_name, mastery = topic_kb.identify_topic(keywords, content)
    
    if mastery < BASIC_THRESHOLD:
        difficulty = "Basic"
    elif mastery < INTERMEDIATE_THRESHOLD:
        difficulty = "Intermediate"
    else:
        difficulty = "Advanced"
    
    print(f"\n📚 Topic: {topic_name}")
    print(f"⭐ Mastery: {mastery:.1f}/5.0 | Level: {difficulty}")
    
    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "current_mastery": mastery,
        "difficulty_level": difficulty
    }


def generate_question(state: QuizState) -> dict:
    """Generate question"""
    question_text = question_generator.generate(
        state["raw_content"],
        state["topic_name"],
        state["current_mastery"],
        state["system_prompt"]
    )
    
    return {"question_text": question_text}


def collect_answer(state: QuizState) -> dict:
    """Collect user answer"""
    print(f"\n{'='*60}")
    print(f"❓ {state['question_text']}")
    print(f"{'='*60}")
    
    if state.get("user_answer"):
        answer = state["user_answer"]
        print(f"💬 {answer}")
    else:
        answer = input("\n💬 Your answer: ").strip()
    
    return {"user_answer": answer}


def score_and_update(state: QuizState) -> dict:
    """Score answer and update topic"""
    score, feedback = answer_scorer.score(
        state["question_text"],
        state["user_answer"],
        state["raw_content"],
        state["system_prompt"]
    )
    
    topic_kb.update_topic_score(
        state["topic_id"],
        score,
        state["question_text"],
        state["user_answer"],
        feedback
    )
    
    session_data = {
        "session_id": state["session_id"],
        "topic_id": state["topic_id"],
        "topic_name": state["topic_name"],
        "question": state["question_text"],
        "answer": state["user_answer"],
        "score": score,
        "feedback": feedback,
        "mastery_before": state["current_mastery"],
        "difficulty": state["difficulty_level"],
        "timestamp": datetime.now().isoformat()
    }
    
    session_storage.save_session(state["session_id"], session_data)
    
    topic_data = topic_kb.get_topic(state["topic_id"])
    new_mastery = topic_data.get("mastery_score", 0.0)
    
    print(f"\n✅ Score: {score}/5")
    print(f"💡 {feedback}")
    print(f"📈 New Mastery: {new_mastery:.1f}/5.0")
    
    return {}
