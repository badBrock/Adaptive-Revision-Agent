# tools/agent_integration.py
from conversational_tutor_agent import start_conversational_tutoring
from typing import Dict, Any
from groq import Groq
import os
def trigger_conversational_tutoring(user_input: str, context: Dict[str, Any]) -> bool:
    """
    Detect when to trigger conversational tutoring based on user input.
    LLM-driven detection of confusion signals.
    """
    confusion_signals = [
        "i don't know", "i don't understand", "i'm confused", "what does that mean",
        "i have no idea", "this doesn't make sense", "i'm lost", "help me understand",
        "can you explain", "i don't get it"
    ]
    
    return any(signal in user_input.lower() for signal in confusion_signals)

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
def trigger_conversational_tutoring(user_input: str, context: dict = None) -> bool:
    """
    Enhanced confusion detection using both keywords and LLM analysis
    """
    # Immediate triggers (fast detection)
    confusion_signals = [
        "i don't know", "i dont know", "i don't understand", "i dont understand", 
        "i'm confused", "im confused", "what does that mean", "i have no idea",
        "this doesn't make sense", "i'm lost", "im lost", "help me understand",
        "can you explain", "i don't get it", "i dont get it", "i am not sure",
        "i'm not sure", "no clue", "no idea", "not sure", "confused",
        "help", "explain", "what", "huh", "?"
    ]
    
    # Quick keyword check
    if any(signal in user_input.lower().strip() for signal in confusion_signals):
        return True
    
    # Short/empty answers that indicate confusion
    if len(user_input.strip()) <= 2:
        return True
        
    # LLM-based confusion detection for edge cases
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        detection_prompt = f"""
        Analyze if this student response indicates confusion or lack of understanding:
        
        Student answer: "{user_input}"
        Context: Academic quiz question
        
        Does this response show confusion, uncertainty, or lack of knowledge?
        Respond with only: YES or NO
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": detection_prompt}],
            model="llama-3.3-70b-versatile",  # Fast model for detection
            max_tokens=5,
            temperature=0.1
        )
        
        llm_result = response.choices[0].message.content.strip().upper()
        return llm_result == "YES"
        
    except Exception:
        # Fallback to conservative detection
        return False

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