from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging

# Import our modular tools
from tools.content_loader import ContentCache
from tools.question_planner import QuestionPlanner
from tools.scorer import AnswerScorer
from state_manager import StateManager

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
    print(f"💡 Provide your answer")

    
    return {
        **state,
        "current_question": current_question,
        "session_status": "question_presented"
    }

# Node 4: Collect one-word answer (same as before)
def collect_answer(state: QuizAgentState) -> QuizAgentState:
    """Collect free-form answer from user"""
    user_answer = input("\n👤 Your answer: ").strip()
    
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
    combined_text = ContentCache.get_combined_text(state['content_cache'], max_chars=800)
    base64_images = ContentCache.get_base64_images(state['content_cache'], max_images=1)
    
    feedback_prompt = f"""
Provide brief feedback for this answer in ONE SENTENCE only:

Question: {current_question}
User's Answer: {user_answer}
Topic: {state['topic_name']}
Context: {combined_text[:400]}...

Give constructive feedback in exactly one sentence.
"""

    
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        content = [{"type": "text", "text": feedback_prompt}]
        
        # Add one cached image for context
        if base64_images:
            content.append({
                "type": "image_url", 
                "image_url": {"url": base64_images[0]}
            })
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
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
    """Create Quiz Agent LangGraph workflow using modular tools"""
    workflow = StateGraph(QuizAgentState)
    
    workflow.add_node("load_and_cache_content", load_and_cache_content)
    workflow.add_node("plan_questions", plan_questions)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("collect_one_word_answer", collect_answer)
    workflow.add_node("provide_feedback", provide_feedback)
    workflow.add_node("score_answer", score_answer)
    workflow.add_node("finalize_session", finalize_session)
    
    workflow.set_entry_point("load_and_cache_content")
    workflow.add_edge("load_and_cache_content", "plan_questions")
    workflow.add_edge("plan_questions", "generate_question")
    workflow.add_edge("generate_question", "collect_one_word_answer")
    workflow.add_edge("collect_one_word_answer", "provide_feedback")
    workflow.add_edge("provide_feedback", "score_answer")
    
    workflow.add_conditional_edges(
        "score_answer",
        session_manager,
        {
            "generate_question": "generate_question",
            "finalize_session": "finalize_session"
        }
    )
    
    workflow.add_edge("finalize_session", END)
    
    return workflow.compile()

# Run function (same interface)
def run_quiz_agent_session(user_id: str, topic_name: str):
    """Run Quiz Agent session using modular tools"""
    print(f"\n🎯 Quiz Agent Session - Using Modular Tools")
    print(f"👤 User: {user_id}")
    print(f"📚 Topic: {topic_name}")
    print("="*50)
    
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
        "session_status": "starting"
    }
    
    try:
        workflow = create_quiz_agent_workflow()
        result = workflow.invoke(initial_state,config={"recursion_limit": 50})
        
        return {
            "status": "success",
            "average_score": result['average_score'],
            "grade": result['grade'],
            "recommendation": result['recommendation'],
            "questions_answered": len(result['scores']),
            "qa_history": result['qa_history']
        }
        
    except Exception as e:
        logger.error(f"❌ Quiz Agent workflow failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
