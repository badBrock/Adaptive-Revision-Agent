from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging

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
    user_explanation: str
    feedback: str
    current_score: float
    qa_history: List[Dict[str, Any]]
    scores: List[float]
    average_score: float
    grade: str
    recommendation: str
    session_status: str

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

# Node 4: Collect user explanation
def collect_user_explanation(state: TeachingAgentState) -> TeachingAgentState:
    """Collect user's teaching explanation"""
    user_explanation = input("\n👤 Your explanation: ").strip()
    
    logger.info(f"📝 User explained: '{user_explanation[:50]}...'")
    
    return {
        **state,
        "user_explanation": user_explanation,
        "session_status": "explanation_collected"
    }

# Node 5: Provide feedback and score
def provide_feedback_and_score(state: TeachingAgentState) -> TeachingAgentState:
    """Provide feedback and score the teaching explanation"""
    current_prompt = state['current_prompt']
    user_explanation = state['user_explanation']
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
        
        # ROBUST SCORE EXTRACTION (REPLACE THE OLD PARSING)
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
                current_score = valid_scores[0]  # Take first valid score
                feedback = response_text.replace(str(current_score), "").replace("Feedback:", "").strip()
            else:
                # Method 3: Fallback - extract any number and clamp it
                all_numbers = re.findall(r'\d+\.?\d*', response_text)
                if all_numbers:
                    raw_score = float(all_numbers[0])
                    current_score = max(0, min(100, raw_score))  # Clamp between 0-100
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
    
    return {
        **state,
        "feedback": feedback,
        "current_score": current_score,
        "qa_history": qa_history,
        "scores": scores,
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
    """Create the Teaching Agent LangGraph workflow"""
    workflow = StateGraph(TeachingAgentState)
    
    # Add nodes
    workflow.add_node("load_and_cache_content", load_and_cache_content)
    workflow.add_node("generate_teaching_prompts", generate_teaching_prompts)
    workflow.add_node("present_teaching_prompt", present_teaching_prompt)
    workflow.add_node("collect_user_explanation", collect_user_explanation)
    workflow.add_node("provide_feedback_and_score", provide_feedback_and_score)
    workflow.add_node("finalize_session", finalize_session)
    
    # Define edges
    workflow.set_entry_point("load_and_cache_content")
    workflow.add_edge("load_and_cache_content", "generate_teaching_prompts")
    workflow.add_edge("generate_teaching_prompts", "present_teaching_prompt")
    workflow.add_edge("present_teaching_prompt", "collect_user_explanation")
    workflow.add_edge("collect_user_explanation", "provide_feedback_and_score")
    
    # Conditional edge for session management
    workflow.add_conditional_edges(
        "provide_feedback_and_score",
        session_manager,
        {
            "present_teaching_prompt": "present_teaching_prompt",
            "finalize_session": "finalize_session"
        }
    )
    
    workflow.add_edge("finalize_session", END)
    
    return workflow.compile()

# Main function (ADD THIS)
def run_teaching_agent_session(user_id: str, topic_name: str):
    """Enhanced Teaching Agent with meta-learning"""
    
    # Initialize meta-learning
    meta_module = MetaLearningModule()
    
    print(f"\n🎓 Enhanced Teaching Agent Session - Meta-Learning Enabled")
    print(f"👤 User: {user_id}")
    print(f"📚 Topic: {topic_name}")
    
    # Get learning recommendations  
    recommendations = meta_module.get_learning_recommendations(user_id, topic_name)
    adaptations = meta_module.get_agent_adaptations(user_id, topic_name, "teaching_agent")
    
    print(f"🧠 Meta-Learning Insights:")
    print(f"   Explanation style: {adaptations.get('explanation_style', 'standard')}")
    print(f"   Prompt complexity: {adaptations.get('prompt_complexity', 'medium')}")
    print(f"   Focus areas: {adaptations['focus_areas']}")
    print("="*50)
    
    # Your existing teaching logic, adapted based on meta-learning
    initial_state: TeachingAgentState = {
        "user_id": user_id,
        "topic_name": topic_name,
        "session_id": "",
        "content_cache": {},
        "teaching_prompts": [],
        "current_prompt_index": 0,
        "current_prompt": "",
        "user_explanation": "",
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
        "meta_recommendations": recommendations
    }
    
    try:
        workflow = create_teaching_agent_workflow()
        result = workflow.invoke(
            initial_state,
            config={"recursion_limit": 25}
        )
        
        # Record session outcome for meta-learning
        session_outcome = {
            "average_score": result['average_score'],
            "questions_answered": result.get('questions_answered', 0),
            "mistakes": extract_explanation_issues(result.get('qa_history', [])),
            "time_taken": 0,
            "feedback_rating": 4
        }
        
        meta_module.record_session_outcome(user_id, topic_name, "teaching_agent", session_outcome)
        
        return {
            "status": "success",
            "average_score": result['average_score'],
            "grade": result['grade'], 
            "recommendation": result['recommendation'],
            "questions_answered": len(result['scores']),
            "qa_history": result['qa_history'],
            "meta_insights": recommendations
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced Teaching Agent workflow failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

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