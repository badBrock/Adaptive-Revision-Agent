from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging
# ✅ Fixed import - removed unused function
from tools.agent_integration import trigger_conversational_tutoring
from conversational_tutor_agent import start_conversational_tutoring

# Import our modular tools
from tools.content_loader import ContentCache
from state_manager import StateManager
from groq import Groq
from tools.meta_learning import MetaLearningModule

logger = logging.getLogger(__name__)

class TeachingAgentState(TypedDict):
    user_id: str
    topic_name: str
    session_id: str
    content_cache: Dict[str, Any]
    teaching_prompts: List[Dict[str, Any]]
    current_prompt_index: int
    current_prompt: str
    # ✅ Fixed: Use consistent variable name
    user_explanation: str  # Changed from user_response to match usage
    feedback: str
    current_score: float
    qa_history: List[Dict[str, Any]]
    scores: List[float]
    average_score: float
    grade: str
    recommendation: str
    session_status: str
    
    # ✅ Added missing variable for teaching_decision function
    understanding_progress: List[float]
    
    # 🔧 Conversational tutoring fields
    conversational_tutoring_completed: bool
    tutoring_result: Dict[str, Any]
    understanding_improved: bool
    user_messages: List[Dict[str, Any]]
    
    # Meta-learning fields
    meta_adaptations: Dict[str, Any]
    meta_recommendations: Dict[str, Any]


# Node 1: Load content using ContentCache tool
def load_and_cache_content(state: TeachingAgentState) -> TeachingAgentState:
    """Load content using modular ContentCache tool"""
    logger.info(f"📁 Loading content for topic: {state['topic_name']}")
    
    try:
        # Load session config using StateManager
        state_manager = StateManager()
        session_file = state_manager.find_latest_session(state['user_id'], state['topic_name'])
        
        if not session_file:
            raise FileNotFoundError(f"No session file found for {state['user_id']} - {state['topic_name']}")
        
        session_config = state_manager.load_quiz_session(session_file)
        if not session_config:
            raise ValueError("Failed to load session configuration")
        
        # Get folder path from md_file_path
        md_file_path = session_config['md_file_path']
        folder_path = os.path.dirname(md_file_path)
        
        # Use ContentCache to load and cache all content ONCE
        content_cache = ContentCache.load_content_cache(folder_path)
        
        logger.info(f"✅ Content cached using ContentCache tool")
        
        return {
            **state,
            "content_cache": content_cache,
            "session_id": os.path.basename(session_file),
            "current_prompt_index": 0,
            "qa_history": [],
            "scores": [],
            "session_status": "content_loaded"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load content: {str(e)}")
        return {
            **state,
            "session_status": "error"
        }

# Node 2: Generate teaching prompts
def generate_teaching_prompts(state: TeachingAgentState) -> TeachingAgentState:
    """Generate Feynman-style teaching prompts from content"""
    logger.info("🧠 Generating teaching prompts...")
    
    try:
        content_cache = state['content_cache']
        combined_text_with_images = ContentCache.get_combined_text_with_images(content_cache, max_chars=1000)
        
        prompt_generation = f"""
Based on this content, create exactly 3 teaching prompts using the Feynman Technique.
Each prompt should ask the user to EXPLAIN a concept in simple terms.

Content: {combined_text_with_images}

Create prompts that test understanding, not memorization. Format as JSON array:
[
  {{"prompt": "Explain weight initialization like you're teaching a beginner", "concept": "weight_initialization"}},
  {{"prompt": "Why do gradients vanish? Explain in simple terms", "concept": "vanishing_gradients"}},
  {{"prompt": "Describe what happens with bad initialization", "concept": "initialization_problems"}}
]
"""
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # CHANGED: Text-only request with larger model
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_generation}],
            model="llama-3.3-70b-versatile",  # Larger text-only model
            max_tokens=400,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON
        if '[' in response_text and ']' in response_text:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            teaching_prompts = json.loads(response_text[json_start:json_end])
        else:
            raise ValueError("No valid JSON found")
        
        logger.info(f"📋 Generated {len(teaching_prompts)} teaching prompts")
        
        return {
            **state,
            "teaching_prompts": teaching_prompts,
            "session_status": "prompts_generated"
        }
        
    except Exception as e:
        logger.error(f"❌ Prompt generation failed: {str(e)}")
        # Fallback prompts
        fallback_prompts = [
            {"prompt": f"Explain the main concept of {state['topic_name']} in simple terms", "concept": "main_concept"},
            {"prompt": f"What problems does {state['topic_name']} solve?", "concept": "problem_solving"},
            {"prompt": f"How would you teach {state['topic_name']} to someone new?", "concept": "teaching_method"}
        ]
        
        return {
            **state,
            "teaching_prompts": fallback_prompts,
            "session_status": "prompts_generated"
        }

# Node 3: Present teaching prompt
def present_teaching_prompt(state: TeachingAgentState) -> TeachingAgentState:
    """Present the next teaching prompt to user"""
    current_index = state['current_prompt_index']
    teaching_prompts = state['teaching_prompts']
    
    if current_index >= len(teaching_prompts):
        return {
            **state,
            "session_status": "prompts_complete"
        }
    
    current_prompt_info = teaching_prompts[current_index]
    current_prompt = current_prompt_info['prompt']
    
    print(f"\n🎓 Teaching Challenge {current_index + 1}/{len(teaching_prompts)}")
    print(f"📝 {current_prompt}")
    print(f"💡 Explain clearly and in your own words")
    
    return {
        **state,
        "current_prompt": current_prompt,
        "session_status": "prompt_presented"
    }
def teaching_decision(state: TeachingAgentState) -> TeachingAgentState:
    """Process the teaching interaction and update progress"""
    
    # Increment prompt index
    current_index = state['current_prompt_index']
    updated_progress = state.get('understanding_progress', [])
    
    # Simple scoring based on response length and content
    user_response = state.get('user_response', '')
    understanding_score = min(10, max(1, len(user_response) // 10 + 5))  # Basic heuristic
    
    updated_progress.append(understanding_score)
    
    return {
        **state,
        "session_status": "progress_updated"
    }

def collect_user_explanation(state: TeachingAgentState) -> TeachingAgentState:
    """Enhanced to detect confusion and trigger conversational tutoring"""
    
    current_prompt = state['current_prompt']
    print(f"\n🎓 {current_prompt}")
    print(f"💭 Please explain your understanding or ask questions:")
    
    user_response = input("👤 Your response: ").strip()
    
    # 🔧 Check for confusion signals
    if trigger_conversational_tutoring(user_response, state):
        logger.info(f"🎓 Confusion detected in teaching: '{user_response}' - Starting conversational tutoring")
        
        # Trigger conversational tutoring
        tutoring_result = start_conversational_tutoring(
            user_id=state['user_id'],
            topic=state['topic_name'], 
            entry_context={
                "from_agent": "teaching_agent",
                "current_prompt": current_prompt,
                "user_response": user_response,
                "confusion_signal": user_response,
                "messages": [
                    {"role": "system", "content": f"Teaching prompt was: {current_prompt}"},
                    {"role": "user", "content": f"I'm confused: {user_response}"}
                ]
            }
        )
        
        # Update state with conversational tutoring results
        return {
            **state,
            "conversational_tutoring_completed": True,
            "tutoring_result": tutoring_result,
            "understanding_improved": tutoring_result.get("session_outcome") in ["mastery_achieved", "good_progress"],
            "user_explanation": user_response,  # ✅ Fixed variable name
            "session_status": "conversational_tutoring_complete"
        }
    
    # Normal teaching flow if no confusion
    logger.info(f"👤 User explained: '{user_response}'")
    
    # Add to conversation history
    updated_messages = state.get("user_messages", []) + [
        {"role": "assistant", "content": current_prompt},
        {"role": "user", "content": user_response, "timestamp": datetime.now().isoformat()}
    ]
    
    return {
        **state,
        "user_explanation": user_response,  # ✅ Fixed variable name
        "user_messages": updated_messages,
        "session_status": "explanation_collected"
    }


# Node 5: Provide feedback and score
def provide_feedback_and_score(state: TeachingAgentState) -> TeachingAgentState:
    """Provide feedback and score the teaching explanation"""
    current_prompt = state['current_prompt']
    user_explanation = state['user_explanation']  # ✅ Now this will work
    current_index = state['current_prompt_index']
    
    # Get reference content
    combined_text = ContentCache.get_combined_text(state['content_cache'], max_chars=800)
    
    scoring_prompt = f"""
Evaluate this user's explanation using the Feynman Technique criteria.

Teaching Prompt: {current_prompt}
User's Explanation: {user_explanation}

Reference Material: {combined_text[:400]}...

Score from 0-100 based on:
- Clarity and simplicity (30%)
- Accuracy of content (40%) 
- Completeness of explanation (30%)

Provide brief feedback in one sentence, then give numeric score.
Format: "Feedback: [one sentence] | Score: [0-100]"
"""
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # ROBUST SCORE EXTRACTION
        import re
        
        # Method 1: Look for "Score: XX" pattern
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
        if score_match:
            current_score = float(score_match.group(1))
            feedback = response_text.split(score_match.group(0))[0].replace("Feedback:", "").strip()
        else:
            # Method 2: Extract any valid number between 0-100
            numbers = re.findall(r'\b(\d{1,2}\.?\d*)\b', response_text)
            valid_scores = [float(num) for num in numbers if 0 <= float(num) <= 100]
            
            if valid_scores:
                current_score = valid_scores[0]
                feedback = response_text.replace(str(current_score), "").replace("Feedback:", "").strip()
            else:
                # Method 3: Fallback
                all_numbers = re.findall(r'\d+\.?\d*', response_text)
                if all_numbers:
                    raw_score = float(all_numbers[0])
                    current_score = max(0, min(100, raw_score))
                else:
                    current_score = 50.0
                feedback = response_text.replace("Feedback:", "").strip()
        
        # Ensure score is in valid range
        current_score = max(0, min(100, current_score))
            
    except Exception as e:
        logger.error(f"❌ Feedback generation failed: {str(e)}")
        feedback = f"Thank you for your explanation. Keep practicing!"
        current_score = 50.0
    
    print(f"\n💭 Feedback: {feedback}")
    print(f"🎯 Score: {current_score:.1f}/100")
    
    # Add to history
    qa_entry = {
        "prompt": current_prompt,
        "explanation": user_explanation,
        "feedback": feedback,
        "score": current_score,
        "timestamp": datetime.now().isoformat()
    }
    
    qa_history = state['qa_history'] + [qa_entry]
    scores = state['scores'] + [current_score]
    
    # ✅ Update understanding progress for teaching_decision function
    understanding_progress = state.get('understanding_progress', []) + [current_score / 10]  # Convert to 1-10 scale
    
    return {
        **state,
        "feedback": feedback,
        "current_score": current_score,
        "qa_history": qa_history,
        "scores": scores,
        "understanding_progress": understanding_progress,  # ✅ Add this
        "current_prompt_index": state['current_prompt_index'] + 1,
        "session_status": "explanation_scored"
    }


# Node 6: Session manager
def session_manager(state: TeachingAgentState) -> str:
    """Decide whether to continue or finalize"""
    current_index = state['current_prompt_index']
    total_prompts = len(state['teaching_prompts'])
    
    if current_index < total_prompts:
        return "present_teaching_prompt"
    else:
        return "finalize_session"
# In quiz_agent.py - check_tutoring_outcome function:
def check_tutoring_outcome(state: TeachingAgentState) -> TeachingAgentState:
    """Handle results from conversational tutoring session"""
    
    tutoring_result = state.get("tutoring_result", {})
    outcome = tutoring_result.get("session_outcome", "needs_practice")
    session_status = tutoring_result.get("session_status", "completed")
    
    # 🔧 NEW: Handle user-requested exit
    if session_status == "exited" or outcome == "early_exit":
        print("👋 Welcome back! Let's continue with your quiz.")
        return {
            **state,
            "user_answer": "",  # Clear previous answer
            "session_status": "returned_from_tutoring"
        }
    elif outcome in ["mastery_achieved", "good_progress"]:
        print("🎉 Great! You've improved your understanding. Let's try the question again!")
        return {
            **state,
            "user_answer": "",
            "session_status": "ready_for_retry"
        }
    else:
        print("💪 Let's continue with some feedback and move forward.")
        return {
            **state,
            "session_status": "continue_normal_flow"
        }

def teaching_session_manager(state: TeachingAgentState) -> str:
    """Decide whether to continue teaching, use more prompts, or end session"""
    current_index = state['current_prompt_index']
    total_prompts = len(state.get('teaching_prompts', []))
    understanding_progress = state.get('understanding_progress', [])
    
    # If we have good understanding progress, consider ending
    if len(understanding_progress) >= 3 and sum(understanding_progress[-3:]) / 3 >= 8:
        return "finalize_teaching_session"
    # If we haven't gone through all prompts, continue
    elif current_index < total_prompts:
        return "present_teaching_prompt"
    # Otherwise, finalize
    else:
        return "finalize_teaching_session"

# Node 7: Finalize session
def finalize_session(state: TeachingAgentState) -> TeachingAgentState:
    """Finalize teaching session with grades and results"""
    scores = state['scores']
    
    if not scores:
        average_score = 0
        grade = "🔴 No Explanations"
        recommendation = "Please attempt the teaching exercises."
    else:
        average_score = sum(scores) / len(scores)
        
        if average_score >= 60:
            grade = "🟢 Excellent Teacher"
            recommendation = "Outstanding! You have mastered this topic. Ready for advanced content."
        elif average_score >= 40:
            grade = "🟡 Good Understanding"
            recommendation = "Good explanations. Review weak areas and try teaching again."
        else:
            grade = "🔴 Needs Study"
            recommendation = "Explanations show gaps. Review the material thoroughly before teaching."
    
    # Save results using StateManager
    try:
        state_manager = StateManager()
        session_result = {
            "user_id": state['user_id'],
            "topic_name": state['topic_name'],
            "session_id": state['session_id'],
            "questions_answered": len(scores),
            "individual_scores": scores,
            "average_score": average_score,
            "grade": grade,
            "recommendation": recommendation,
            "qa_history": state['qa_history'],
            "completion_time": datetime.now().isoformat(),
            "session_type": "teaching_agent"  # Identify as teaching session
        }
        
        state_manager.save_quiz_results(state['user_id'], state['topic_name'], session_result)
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {str(e)}")
    
    print(f"\n" + "="*60)
    print(f"🎓 TEACHING SESSION COMPLETE!")
    print(f"="*60)
    print(f"📚 Topic: {state['topic_name']}")
    print(f"🎯 Prompts Completed: {len(scores)}")
    print(f"📊 Individual Scores: {[f'{s:.1f}' for s in scores]}")
    print(f"🏆 Average Score: {average_score:.2f}/100")
    print(f"🎯 Final Grade: {grade}")
    print(f"💡 Recommendation: {recommendation}")
    print(f"="*60)
    
    return {
        **state,
        "average_score": average_score,
        "grade": grade,
        "recommendation": recommendation,
        "session_status": "completed"
    }

# Build the workflow (ADD THIS)
def create_teaching_agent_workflow():
    """Enhanced Teaching Agent workflow with conversational tutoring integration"""
    
    workflow = StateGraph(TeachingAgentState)
    
    # Add all nodes
    workflow.add_node("load_and_cache_content", load_and_cache_content)
    workflow.add_node("generate_teaching_prompts", generate_teaching_prompts)
    workflow.add_node("present_teaching_prompt", present_teaching_prompt)
    workflow.add_node("collect_user_explanation", collect_user_explanation)
    workflow.add_node("provide_feedback_and_score", provide_feedback_and_score)
    workflow.add_node("teaching_decision", teaching_decision)
    workflow.add_node("finalize_session", finalize_session)
    workflow.add_node("check_tutoring_outcome_teaching", check_tutoring_outcome)
    
    # Set entry point
    workflow.set_entry_point("load_and_cache_content")
    
    # Linear flow
    workflow.add_edge("load_and_cache_content", "generate_teaching_prompts")
    workflow.add_edge("generate_teaching_prompts", "present_teaching_prompt")
    workflow.add_edge("present_teaching_prompt", "collect_user_explanation")
    
    # ✅ Conditional edge for conversational tutoring
    workflow.add_conditional_edges(
        "collect_user_explanation",
        lambda state: "tutoring_complete" if state.get("conversational_tutoring_completed") else "continue_teaching",
        {
            "tutoring_complete": "check_tutoring_outcome_teaching",
            "continue_teaching": "provide_feedback_and_score"
        }
    )
    
    # Continue normal flow
    workflow.add_edge("provide_feedback_and_score", "teaching_decision")
    
    # ✅ Session management routing
    workflow.add_conditional_edges(
        "teaching_decision",
        teaching_session_manager,
        {
            "present_teaching_prompt": "present_teaching_prompt",
            "finalize_session": "finalize_session"
        }
    )
    
    workflow.add_edge("finalize_session", END)
    
    # ✅ Handle tutoring outcome
    workflow.add_conditional_edges(
        "check_tutoring_outcome_teaching", 
        lambda state: "enhanced_teaching" if state.get("understanding_improved") else "continue_teaching",
        {
            "enhanced_teaching": "provide_feedback_and_score",
            "continue_teaching": "provide_feedback_and_score"
        }
    )
    
    return workflow.compile()

# Main function (ADD THIS)
def run_teaching_agent_session(user_id: str, topic_name: str):
    """Enhanced Teaching Agent with conversational tutoring support"""
    
    meta_module = MetaLearningModule()
    
    print(f"\n🎓 Enhanced Teaching Agent Session - Feynman Technique + Conversational Tutoring")
    print(f"👤 User: {user_id}")
    print(f"📚 Topic: {topic_name}")
    
    recommendations = meta_module.get_learning_recommendations(user_id, topic_name)
    adaptations = meta_module.get_agent_adaptations(user_id, topic_name, "teaching_agent")
    
    print(f"🧠 Meta-Learning Insights:")
    print(f"   Recommended difficulty: {adaptations['difficulty_adjustment']}")
    print(f"   Focus areas: {adaptations['focus_areas']}")
    print(f"   Learning velocity: {adaptations['learning_velocity']:.2f}")
    print("="*70)
    
    # ✅ Complete initial state
    initial_state: TeachingAgentState = {
        "user_id": user_id,
        "topic_name": topic_name,
        "session_id": f"teaching_{user_id}_{int(datetime.now().timestamp())}",
        "content_cache": {},
        "teaching_prompts": [],
        "current_prompt_index": 0,
        "current_prompt": "",
        "user_explanation": "",  # ✅ Fixed variable name
        "feedback": "",
        "current_score": 0.0,
        "qa_history": [],
        "scores": [],
        "average_score": 0.0,
        "grade": "",
        "recommendation": "",
        "understanding_progress": [],  # ✅ Added missing field
        "session_status": "starting",
        
        # Conversational tutoring fields
        "conversational_tutoring_completed": False,
        "tutoring_result": {},
        "understanding_improved": False,
        "user_messages": [],
        
        # Meta-learning data
        "meta_adaptations": adaptations,
        "meta_recommendations": recommendations
    }
    
    try:
        workflow = create_teaching_agent_workflow()
        result = workflow.invoke(
            initial_state,
            config={
                "recursion_limit": 40,
                "configurable": {"thread_id": f"teaching_session_{user_id}_{topic_name}"}
            }
        )
        
        # Check if conversational tutoring was used
        if result.get("conversational_tutoring_completed"):
            print(f"\n🎓 Conversational tutoring enhanced this teaching session!")
            tutoring_outcome = result.get("tutoring_result", {}).get("session_outcome", "unknown")
            print(f"📊 Tutoring outcome: {tutoring_outcome}")
        
        # Record session outcome for meta-learning
        session_outcome = {
            "understanding_progress": result.get('understanding_progress', []),
            "prompts_completed": result.get('current_prompt_index', 0),
            "engagement_level": calculate_engagement_from_responses(result.get('user_messages', [])),
            "time_taken": 0,
            "feedback_quality": 4,
            "conversational_tutoring_used": result.get("conversational_tutoring_completed", False),
            "tutoring_effectiveness": result.get("tutoring_result", {}).get("session_outcome", "not_used")
        }
        
        meta_module.record_session_outcome(user_id, topic_name, "teaching_agent", session_outcome)
        
        print(f"\n📊 Meta-Learning Updated with teaching session results")
        
        return {
            "status": "success",
            "understanding_progress": result.get('understanding_progress', []),
            "prompts_completed": result.get('current_prompt_index', 0),
            "final_understanding": sum(result.get('understanding_progress', [0])) / max(len(result.get('understanding_progress', [1])), 1),
            "user_messages": result.get('user_messages', []),
            "meta_insights": recommendations,
            "conversational_tutoring_used": result.get("conversational_tutoring_completed", False),
            "tutoring_result": result.get("tutoring_result", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced Teaching Agent workflow failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def calculate_engagement_from_responses(messages: List[Dict]) -> int:
    """Calculate engagement score from user responses (1-10)"""
    if not messages:
        return 5
    
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return 5
    
    # Simple heuristic: longer responses = higher engagement
    avg_length = sum(len(msg.get("content", "")) for msg in user_messages) / len(user_messages)
    
    if avg_length > 100:
        return 9
    elif avg_length > 50:
        return 7
    elif avg_length > 20:
        return 5
    else:
        return 3


def extract_explanation_issues(qa_history: List[Dict]) -> List[Dict]:
    """Extract explanation quality issues"""
    issues = []
    for qa in qa_history:
        if qa.get('score', 100) < 60:
            issues.append({
                "prompt": qa.get('prompt', ''),
                "explanation": qa.get('explanation', ''),
                "type": "poor_explanation",
                "score": qa.get('score', 0)
            })
    return issues