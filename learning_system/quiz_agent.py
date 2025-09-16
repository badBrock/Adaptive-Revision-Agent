from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging
from tools.agent_integration import trigger_conversational_tutoring, integrate_with_quiz_agent
from conversational_tutor_agent import start_conversational_tutoring

# Import our modular tools
from tools.content_loader import ContentCache
from tools.question_planner import QuestionPlanner
from tools.scorer import AnswerScorer
from state_manager import StateManager
from tools.meta_learning import MetaLearningModule

logger = logging.getLogger(__name__)

class QuizAgentState(TypedDict):
    user_id: str
    topic_name: str
    session_id: str
    content_cache: Dict[str, Any]
    question_plan: List[Dict[str, Any]]
    total_questions: int
    current_question_index: int
    current_question: str
    user_answer: str
    feedback: str
    current_score: float
    qa_history: List[Dict[str, Any]]
    scores: List[float]
    average_score: float
    grade: str
    recommendation: str
    session_status: str
    # 🔧 NEW: Conversational tutoring fields
    conversational_tutoring_completed: bool
    tutoring_result: Dict[str, Any]
    understanding_improved: bool
    # Meta-learning fields
    meta_adaptations: Dict[str, Any]
    meta_recommendations: Dict[str, Any]

# Node 1: Load content using ContentCache tool
def load_and_cache_content(state: QuizAgentState) -> QuizAgentState:
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
            "current_question_index": 0,
            "qa_history": [],
            "scores": [],
            "session_status": "content_loaded"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load content: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "recommendation": f"Failed to load content: {str(e)}"
        }

# Node 2: Plan questions using QuestionPlanner tool
def plan_questions(state: QuizAgentState) -> QuizAgentState:
    """Plan all questions using modular QuestionPlanner tool"""
    logger.info("🧠 Planning questions using QuestionPlanner tool...")
    
    try:
        # Use QuestionPlanner tool instead of inline planning
        planner = QuestionPlanner(api_key=os.getenv("GROQ_API_KEY"))
        question_plan = planner.plan_questions(
            content_cache=state['content_cache'],
            difficulty="medium",
            num_questions=3
        )
        
        total_questions = len(question_plan)
        logger.info(f"📋 QuestionPlanner generated {total_questions} questions")
        
        return {
            **state,
            "question_plan": question_plan,
            "total_questions": total_questions,
            "session_status": "questions_planned"
        }
        
    except Exception as e:
        logger.error(f"❌ Question planning failed: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "recommendation": f"Question planning failed: {str(e)}"
        }

# Node 3: Generate question (from plan, no API calls)
def generate_question(state: QuizAgentState) -> QuizAgentState:
    """Get next question from plan (no content re-sending)"""
    current_index = state['current_question_index']
    question_plan = state['question_plan']
    
    if current_index >= len(question_plan):
        return {
            **state,
            "current_question": "No more questions available.",
            "session_status": "questions_complete"
        }
    
    question_info = question_plan[current_index]
    current_question = question_info['question']
    
    print(f"\n🎯 Question {current_index + 1}/{state['total_questions']} ({question_info.get('difficulty', 'medium')})")
    print(f"📝 {current_question}")
    

    
    return {
        **state,
        "current_question": current_question,
        "session_status": "question_presented"
    }

# Node 4: Collect one-word answer (same as before)
def collect_one_word_answer(state: QuizAgentState) -> QuizAgentState:
    """Enhanced to detect confusion and trigger conversational tutoring"""
    
    
    user_answer = input("👤 Your answer: ").strip()
    
    # 🔧 NEW: Check for confusion signals
    if trigger_conversational_tutoring(user_answer, state):
        logger.info(f"🎓 Confusion detected: '{user_answer}' - Starting conversational tutoring")
        
        # Trigger conversational tutoring
        tutoring_result = start_conversational_tutoring(
            user_id=state['user_id'],
            topic=state['topic_name'], 
            entry_context={
                "from_agent": "quiz_agent",
                "failed_question": state['current_question'],
                "user_response": user_answer,
                "confusion_signal": user_answer,
                "messages": [
                    {"role": "system", "content": f"User struggled with: {state['current_question']}"},
                    {"role": "user", "content": f"I need help: {user_answer}"}
                ]
            }
        )
        
        # Update state with conversational tutoring results
        return {
            **state,
            "conversational_tutoring_completed": True,
            "tutoring_result": tutoring_result,
            "understanding_improved": tutoring_result.get("session_outcome") in ["mastery_achieved", "good_progress"],
            "session_status": "conversational_tutoring_complete"
        }
    
    # Normal quiz flow if no confusion
    logger.info(f"📝 User answered: '{user_answer}'")
    return {
        **state,
        "user_answer": user_answer,
        "session_status": "answer_collected"
    }
# Node 5: Provide feedback using cached content
def provide_feedback(state: QuizAgentState) -> QuizAgentState:
    """Provide feedback using cached content (no re-sending)"""
    current_question = state['current_question']
    user_answer = state['user_answer']
    current_index = state['current_question_index']
    
    # Use cached content for context
    combined_text_with_images = ContentCache.get_combined_text_with_images(state['content_cache'], max_chars=800)
    feedback_prompt = f"""
Provide brief feedback for this answer in ONE SENTENCE only:

Question: {current_question}
User's Answer: {user_answer}
Topic: {state['topic_name']}
Context: {combined_text_with_images[:400]}...

Give constructive feedback in exactly one sentence.
"""

    
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # CHANGED: Text-only request with larger model
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": feedback_prompt}],
            model="llama-3.3-70b-versatile",  # Larger text-only model
            max_tokens=200,
            temperature=0.5
        )
        feedback = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ Feedback generation failed: {str(e)}")
        feedback = f"Thank you for your answer '{user_answer}'. Let's continue!"
    
    print(f"\n💭 Feedback: {feedback}")
    
    return {
        **state,
        "feedback": feedback,
        "session_status": "feedback_provided"
    }
# In quiz_agent.py - check_tutoring_outcome function:
def check_tutoring_outcome(state: QuizAgentState) -> QuizAgentState:
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

# Node 6: Score answer using AnswerScorer tool
def score_answer(state: QuizAgentState) -> QuizAgentState:
    """Score answer using modular AnswerScorer tool"""
    current_question = state['current_question']
    user_answer = state['user_answer']
    feedback = state['feedback']
    current_index = state['current_question_index']
    question_info = state['question_plan'][current_index] if current_index < len(state['question_plan']) else None
    
    try:
        # Use AnswerScorer tool instead of inline scoring
        scorer = AnswerScorer(api_key=os.getenv("GROQ_API_KEY"))
        scoring_result = scorer.score_answer(
            question=current_question,
            user_answer=user_answer,
            content_cache=state['content_cache'],
            feedback=feedback,
            question_info=question_info
        )
        
        current_score = scoring_result["score"]
        
    except Exception as e:
        logger.error(f"❌ AnswerScorer failed: {str(e)}")
        current_score = 50.0  # Fallback score
    
    # Add to history and scores
    qa_entry = {
        "question": current_question,
        "answer": user_answer,
        "feedback": feedback,
        "score": current_score,
        "timestamp": datetime.now().isoformat()
    }
    
    qa_history = state['qa_history'] + [qa_entry]
    scores = state['scores'] + [current_score]
    
    print(f"\n🎯 Score: {current_score:.1f}/100")
    
    return {
        **state,
        "current_score": current_score,
        "qa_history": qa_history,
        "scores": scores,
        "current_question_index": state['current_question_index'] + 1,
        "session_status": "answer_scored"
    }

# Node 7: Session manager (same logic)
def session_manager(state: QuizAgentState) -> str:
    """Decide whether to continue with more questions or finalize"""
    current_index = state['current_question_index']
    total_questions = state['total_questions']
    
    if current_index < total_questions:
        return "generate_question"
    else:
        return "finalize_session"
def continue_or_end(state: QuizAgentState) -> str:
    """Decide whether to continue with more questions or finalize session"""
    current_index = state['current_question_index']
    total_questions = state['total_questions']
    
    logger.info(f"🔄 Routing decision: {current_index}/{total_questions} questions completed")
    
    if current_index < total_questions:
        return "generate_question"
    else:
        return "finalize_session"


# Node 8: Finalize session using AnswerScorer for grading
def finalize_session(state: QuizAgentState) -> QuizAgentState:
    """Finalize session using AnswerScorer for final grading"""
    scores = state['scores']
    
    try:
        # Use AnswerScorer tool for session grading
        scorer = AnswerScorer(api_key=os.getenv("GROQ_API_KEY"))
        session_grade = scorer.calculate_session_grade(scores)
        
        average_score = session_grade["average_score"]
        grade = session_grade["grade"]
        recommendation = session_grade["recommendation"]
        
    except Exception as e:
        logger.error(f"❌ Session grading failed: {str(e)}")
        average_score = sum(scores) / len(scores) if scores else 0
        grade = "🔴 Error in grading"
        recommendation = "Grading system error occurred."
    
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
            "completion_time": datetime.now().isoformat()
        }
        
        state_manager.save_quiz_results(state['user_id'], state['topic_name'], session_result)
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {str(e)}")
    
    print(f"\n" + "="*60)
    print(f"🎉 QUIZ SESSION COMPLETE!")
    print(f"="*60)
    print(f"📚 Topic: {state['topic_name']}")
    print(f"❓ Questions Answered: {len(scores)}")
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

# Build Quiz Agent workflow (same structure)
def create_quiz_agent_workflow():
    """Enhanced workflow with conversational tutoring integration"""
    
    workflow = StateGraph(QuizAgentState)
    
    # Add all nodes
    workflow.add_node("load_and_cache_content", load_and_cache_content)
    workflow.add_node("plan_questions", plan_questions)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("collect_one_word_answer", collect_one_word_answer)
    workflow.add_node("provide_feedback", provide_feedback)
    workflow.add_node("score_answer", score_answer)
    workflow.add_node("finalize_session", finalize_session)
    workflow.add_node("check_tutoring_outcome", check_tutoring_outcome)
    
    # Set entry point
    workflow.set_entry_point("load_and_cache_content")
    
    # Linear flow
    workflow.add_edge("load_and_cache_content", "plan_questions")
    workflow.add_edge("plan_questions", "generate_question")
    workflow.add_edge("generate_question", "collect_one_word_answer")
    
    # Conditional edge for conversational tutoring
    workflow.add_conditional_edges(
        "collect_one_word_answer",
        lambda state: "tutoring_complete" if state.get("conversational_tutoring_completed") else "continue_quiz",
        {
            "tutoring_complete": "check_tutoring_outcome",
            "continue_quiz": "provide_feedback"
        }
    )
    
    # Continue normal flow
    workflow.add_edge("provide_feedback", "score_answer")
    
    # 🔧 THIS IS WHERE continue_or_end IS USED
    workflow.add_conditional_edges(
        "score_answer", 
        continue_or_end,  # This function was missing!
        {
            "generate_question": "generate_question",
            "finalize_session": "finalize_session"
        }
    )
    
    workflow.add_edge("finalize_session", END)
    
    # Handle tutoring outcome
    workflow.add_conditional_edges(
        "check_tutoring_outcome", 
        lambda state: "retry_question" if state.get("understanding_improved") else "continue_quiz",
        {
            "retry_question": "generate_question",
            "continue_quiz": "provide_feedback"
        }
    )
    
    return workflow.compile()


def run_quiz_agent_session(user_id: str, topic_name: str):
    """Enhanced Quiz Agent with conversational tutoring support"""
    
    # Initialize meta-learning
    meta_module = MetaLearningModule()
    
    print(f"\n🎯 Enhanced Quiz Agent Session - Meta-Learning Enabled")
    print(f"👤 User: {user_id}")
    print(f"📚 Topic: {topic_name}")
    
    # Get learning recommendations
    recommendations = meta_module.get_learning_recommendations(user_id, topic_name)
    adaptations = meta_module.get_agent_adaptations(user_id, topic_name, "quiz_agent")
    
    print(f"🧠 Meta-Learning Insights:")
    print(f"   Recommended difficulty: {adaptations['difficulty_adjustment']}")
    print(f"   Focus areas: {adaptations['focus_areas']}")
    print(f"   Learning velocity: {adaptations['learning_velocity']:.2f}")
    print("="*50)
    
    # Your existing initial_state setup...
    initial_state: QuizAgentState = {
        "user_id": user_id,
        "topic_name": topic_name,
        "session_id": "",
        "content_cache": {},
        "question_plan": [],
        "total_questions": 0,
        "current_question_index": 0,
        "current_question": "",
        "user_answer": "",
        "feedback": "",
        "current_score": 0.0,
        "qa_history": [],
        "scores": [],
        "average_score": 0.0,
        "grade": "",
        "recommendation": "",
        "session_status": "starting",
        # Add meta-learning data
        "meta_adaptations": adaptations,
        "meta_recommendations": recommendations,
        # 🔧 NEW: Add conversational tutoring tracking
        "conversational_tutoring_completed": False,
        "tutoring_result": {},
        "understanding_improved": False
    }
    
    try:
        workflow = create_quiz_agent_workflow()
        result = workflow.invoke(
            initial_state,
            config={
                "recursion_limit": 30,  # 🔧 Increased for conversational tutoring
                "configurable": {"thread_id": f"quiz_session_{user_id}_{topic_name}"}  # 🔧 Memory persistence
            }
        )
        
        # 🔧 NEW: Check if conversational tutoring was used
        if result.get("conversational_tutoring_completed"):
            print(f"\n🎓 Conversational tutoring was activated during this session!")
            tutoring_outcome = result.get("tutoring_result", {}).get("session_outcome", "unknown")
            print(f"📊 Tutoring outcome: {tutoring_outcome}")
        
        # Record session outcome for meta-learning
        session_outcome = {
            "average_score": result.get('average_score', 0),
            "questions_answered": len(result.get('scores', [])),
            "mistakes": extract_mistakes_from_qa_history(result.get('qa_history', [])),
            "time_taken": 0,
            "feedback_rating": 4,
            # 🔧 NEW: Track conversational tutoring usage
            "conversational_tutoring_used": result.get("conversational_tutoring_completed", False),
            "tutoring_effectiveness": result.get("tutoring_result", {}).get("session_outcome", "not_used")
        }
        
        meta_module.record_session_outcome(user_id, topic_name, "quiz_agent", session_outcome)
        
        print(f"\n📊 Meta-Learning Updated with session results")
        
        return {
            "status": "success",
            "average_score": result.get('average_score', 0),
            "grade": result.get('grade', 'N/A'),
            "recommendation": result.get('recommendation', 'No recommendation'),
            "questions_answered": len(result.get('scores', [])),
            "qa_history": result.get('qa_history', []),
            "meta_insights": recommendations,
            # 🔧 NEW: Return conversational tutoring info
            "conversational_tutoring_used": result.get("conversational_tutoring_completed", False),
            "tutoring_result": result.get("tutoring_result", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced Quiz Agent workflow failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def extract_mistakes_from_qa_history(qa_history: List[Dict]) -> List[Dict]:
    """Extract mistake patterns from Q&A history"""
    mistakes = []
    for qa in qa_history:
        if qa.get('score', 100) < 70:  # Consider <70% as mistake
            mistakes.append({
                "question": qa.get('question', ''),
                "user_answer": qa.get('answer', ''),
                "correct_answer": qa.get('expected_answer', ''),
                "type": "incorrect_answer",  # You can categorize better
                "score": qa.get('score', 0)
            })
    return mistakes