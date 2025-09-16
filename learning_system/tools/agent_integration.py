# tools/agent_integration.py

from conversational_tutor_agent import start_conversational_tutoring
from typing import Dict, Any
from groq import Groq
import os
import re
import logging

logger = logging.getLogger(__name__)

def trigger_conversational_tutoring(user_input: str, context: Dict[str, Any] = None) -> bool:
    """
    Enhanced two-tier trigger for conversational tutoring:
    1. Explicit confusion signals ("I don't know") → immediate trigger
    2. LLM scoring: if answer scores < 20/100 → trigger
    """
    user_input = user_input.lower().strip()
    
    # TIER 1: Explicit confusion signals (immediate trigger)
    explicit_triggers = [
        "i don't know",
        "i dont know", 
        "idk",
        "no idea",
        "not sure",
        "i'm confused",
        "im confused",
        "i don't understand",
        "i dont understand",
        "help me",
        "i'm lost",
        "im lost"
    ]
    
    # Check for exact matches or phrases starting with triggers
    for trigger in explicit_triggers:
        if user_input == trigger or user_input.startswith(trigger + " "):
            logger.info(f"🎓 Explicit trigger detected: '{user_input}'")
            return True
    
    # Check very short confusion signals
    short_signals = {"?", "??", "...", "huh"}
    if user_input in short_signals:
        logger.info(f"🎓 Short confusion signal detected: '{user_input}'")
        return True
    
    # TIER 2: LLM-based answer quality scoring
    return _score_based_trigger(user_input, context)

def _score_based_trigger(user_answer: str, context: Dict[str, Any] = None, threshold: float = 20.0) -> bool:
    """
    Use LLM to score answer quality. Trigger if score < threshold.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("No GROQ_API_KEY found for scoring trigger")
        return False
    
    # Don't score very short answers (likely keywords)
    if len(user_answer.strip()) <= 2:
        return False
    
    try:
        client = Groq(api_key=api_key)
        
        # Get question context if available
        current_question = ""
        if context and hasattr(context, 'get'):
            current_question = context.get('current_question', '')
        
        scoring_prompt = f"""
Score this student's answer for correctness and relevance on a 0-100 scale.

Question: {current_question}
Student Answer: {user_answer}

Consider:
- Correctness of the answer
- Relevance to the question
- Demonstrates understanding

Respond with ONLY a number (0-100):
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=10,
            temperature=0
        )
        
        score_text = response.choices[0].message.content.strip()
        
        # Extract numeric score
        match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_text)
        if match:
            score = float(match.group(1))
            score = max(0, min(100, score))  # Clamp to 0-100
            
            logger.info(f"🎯 Answer scored: {score}/100 (threshold: {threshold})")
            
            if score < threshold:
                logger.info(f"🎓 Low score trigger: {score} < {threshold} - Starting conversational tutoring")
                return True
            else:
                logger.info(f"✅ Good answer: {score} ≥ {threshold} - Continue normal flow")
                return False
        else:
            logger.warning(f"Could not extract score from: '{score_text}'")
            return False
            
    except Exception as e:
        logger.error(f"❌ Scoring trigger failed: {str(e)}")
        return False

def integrate_with_quiz_agent(quiz_state, user_response: str) -> Dict[str, Any]:
    """Integration point for Quiz Agent"""
    if trigger_conversational_tutoring(user_response, quiz_state):
        # Transition to conversational tutoring
        tutoring_result = start_conversational_tutoring(
            user_id=quiz_state['user_id'],
            topic=quiz_state['topic_name'],
            entry_context={
                "from_agent": "quiz_agent",
                "failed_question": quiz_state.get('current_question', ''),
                "user_response": user_response,
                "messages": [
                    {"role": "user", "content": f"I need help with: {user_response}"}
                ]
            }
        )
        return {
            "action": "conversational_tutoring_completed",
            "tutoring_result": tutoring_result,
            "return_to_quiz": tutoring_result["session_outcome"] in ["mastery_achieved", "good_progress"]
        }
    
    return {"action": "continue_quiz"}

def integrate_with_teaching_agent(teaching_state, user_response: str) -> Dict[str, Any]:
    """Integration point for Teaching Agent"""
    if trigger_conversational_tutoring(user_response, teaching_state):
        tutoring_result = start_conversational_tutoring(
            user_id=teaching_state['user_id'],
            topic=teaching_state['topic_name'],
            entry_context={
                "from_agent": "teaching_agent",
                "context": "explanation_confusion",
                "user_response": user_response
            }
        )
        return {
            "action": "conversational_tutoring_completed",
            "tutoring_result": tutoring_result
        }
    
    return {"action": "continue_teaching"}

def detect_confusion_in_conversation(state: dict) -> list:
    """Extract confusion signals from conversation history"""
    messages = state.get("messages", [])
    recent_messages = messages[-3:] if messages else []
    confusion_indicators = []
    
    for msg in recent_messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if trigger_conversational_tutoring(content):
                confusion_indicators.append(content)
    
    return confusion_indicators
