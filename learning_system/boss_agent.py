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

class BossAgentState(TypedDict):
    user_id: str
    content_folder_path: str
    content_cache: Dict[str, Any]
    user_state: Dict[str, Any]
    reasoning_context: Dict[str, Any]
    decision: Dict[str, Any]
    action_result: Dict[str, Any]
    quiz_session_file: str
    quiz_results: Optional[Dict[str, Any]]
    session_status: str
    error_message: str

# Node 1: Initialize Boss Agent using StateManager
def initialize_boss_agent(state: BossAgentState) -> BossAgentState:
    """Initialize Boss Agent using modular StateManager"""
    logger.info(f"🤖 Initializing Boss Agent for user: {state['user_id']}")
    
    try:
        # Use StateManager instead of inline state management
        state_manager = StateManager()
        user_state = state_manager.load_user_state(state['user_id'])
        
        return {
            **state,
            "user_state": user_state,
            "session_status": "initialized"
        }
        
    except Exception as e:
        logger.error(f"❌ Boss Agent initialization failed: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "error_message": f"Initialization failed: {str(e)}"
        }

# Node 2: Discover content using ContentCache tool
def discover_content(state: BossAgentState) -> BossAgentState:
    """Discover content using modular ContentCache tool"""
    logger.info(f"🔍 Discovering content using ContentCache...")
    
    try:
        # Use ContentCache tool instead of inline content discovery
        content_cache = ContentCache.load_content_cache(state['content_folder_path'])
        
        # Get summary for logging
        summary = ContentCache.get_content_summary(content_cache)
        logger.info(f"📁 Content discovered: {summary}")
        
        return {
            **state,
            "content_cache": content_cache,
            "session_status": "content_discovered"
        }
        
    except Exception as e:
        logger.error(f"❌ Content discovery failed: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "error_message": f"Content discovery failed: {str(e)}"
        }

# Add to the top of your boss_agent.py file

def react_reasoning(state: BossAgentState) -> BossAgentState:
    """Enhanced ReAct reasoning with meta-learning capabilities"""
    logger.info("🧠 Starting Enhanced ReAct reasoning with Meta-Learning...")
    
    user_state = state['user_state']
    content_cache = state['content_cache']
    
    # Prepare reasoning context
    reasoning_context = {
        "user_id": state['user_id'],
        "total_sessions": user_state.get("total_sessions", 0),
        "topics_studied": list(user_state.get("topics", {}).keys()),
        "topic_scores": user_state.get("topics", {}),
        "available_topics": [doc["topic_name"] for doc in content_cache["documents"]],
        "overall_progress": user_state.get("overall_progress", 0.0),
        "content_summary": ContentCache.get_content_summary(content_cache)
    }

    # Get combined text and images from cache
    combined_text_with_images = ContentCache.get_combined_text_with_images(content_cache, max_chars=1500)
    # META-LEARNING ENHANCEMENT: Get recommendations for each topic
    meta_module = MetaLearningModule()
    user_id = reasoning_context["user_id"]
    topic_recommendations = {}
    
    logger.info("🧠 Gathering meta-learning insights...")
    for topic in reasoning_context["available_topics"]:
        try:
            recommendations = meta_module.get_learning_recommendations(user_id, topic)
            topic_recommendations[topic] = recommendations
        except Exception as e:
            logger.warning(f"Failed to get meta-learning recommendations for {topic}: {e}")
            topic_recommendations[topic] = {
                "recommended_agent": "quiz_agent",
                "difficulty": "medium",
                "reasoning": "Default recommendation due to meta-learning failure"
            }

    # Enhanced reasoning prompt with meta-learning insights
    reasoning_prompt = f"""
You are an intelligent learning coordinator (Boss Agent) enhanced with meta-learning capabilities.

USER LEARNING CONTEXT:
{json.dumps(reasoning_context, indent=2)}

META-LEARNING INSIGHTS:
{json.dumps(topic_recommendations, indent=2)}

AVAILABLE CONTENT:
{combined_text_with_images[:500]}...

ENHANCED ROUTING RULES:
- Use meta-learning recommendations as primary guidance
- Score < 40%: Quiz Agent (basics/fundamentals)
- Score 40-59%: Quiz Agent (review/standard)
- Score 60-79%: Teaching Agent (Feynman technique)
- Score 80%+: Teaching Agent (mastery confirmation)

Consider the meta-learning insights about agent effectiveness and user patterns when making decisions.

Respond with ONLY valid JSON:
{{
    "reasoning": "Your step-by-step thinking incorporating meta-learning insights",
    "recommended_topic": "specific_topic_name_from_available_topics",
    "recommended_agent": "quiz_agent|teaching_agent",
    "learning_strategy": "review|sequential|advanced|basics|teaching",
    "confidence": 0.95,
    "expected_difficulty": "easy|medium|hard",
    "session_recommendation": "quick|standard|extended",
    "meta_learning_applied": true
}}
"""

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # CHANGED: Use text-only content, larger model
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": reasoning_prompt}],
            model="llama-3.3-70b-versatile",  # Use larger text-only model
            max_tokens=600,
            temperature=0.3
        )
        
        reasoning_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 Raw LLM response: {reasoning_text[:200]}...")
        
        # Parse JSON response
        if '{' in reasoning_text and '}' in reasoning_text:
            json_start = reasoning_text.find('{')
            json_end = reasoning_text.rfind('}') + 1
            decision = json.loads(reasoning_text[json_start:json_end])
        else:
            raise ValueError("No valid JSON found in LLM response")
        
        # Validate and correct routing decision using meta-learning + score rules
        decision = validate_and_correct_routing(decision, reasoning_context["topic_scores"], topic_recommendations)
        
        # Validate recommended topic exists
        available_topics = [doc["topic_name"] for doc in content_cache["documents"]]
        if decision.get("recommended_topic") not in available_topics:
            decision["recommended_topic"] = available_topics[0] if available_topics else None
            logger.warning(f"🔧 Corrected topic to: {decision['recommended_topic']}")
        
        logger.info(f"💡 Final Decision: Study '{decision.get('recommended_topic')}' using '{decision.get('recommended_agent')}' with strategy '{decision.get('learning_strategy')}'")
        
        return {
            **state,
            "reasoning_context": reasoning_context,
            "decision": decision,
            "meta_recommendations": topic_recommendations,
            "session_status": "reasoning_complete"
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced ReAct reasoning failed: {str(e)}")
        
        # Enhanced fallback using meta-learning if available
        available_topics = [doc["topic_name"] for doc in content_cache["documents"]]
        fallback_topic = available_topics[0] if available_topics else None
        fallback_agent = "quiz_agent"
        
        # Try to use meta-learning for fallback
        if fallback_topic and fallback_topic in topic_recommendations:
            meta_rec = topic_recommendations[fallback_topic]
            fallback_agent = meta_rec.get("recommended_agent", "quiz_agent")
        
        fallback_decision = {
            "reasoning": f"Fallback applied due to error: {str(e)}. Using meta-learning backup if available.",
            "recommended_topic": fallback_topic,
            "recommended_agent": fallback_agent,
            "learning_strategy": "basics",
            "confidence": 0.5,
            "expected_difficulty": "medium",
            "session_recommendation": "standard",
            "meta_learning_applied": bool(topic_recommendations)
        }
        
        return {
            **state,
            "reasoning_context": reasoning_context,
            "decision": fallback_decision,
            "meta_recommendations": topic_recommendations,
            "session_status": "reasoning_complete_fallback"
        }

def validate_and_correct_routing(decision: Dict[str, Any], topic_scores: Dict, meta_recommendations: Dict) -> Dict[str, Any]:
    """Validate LLM routing decision against score-based rules and meta-learning insights"""
    
    recommended_topic = decision.get("recommended_topic")
    if not recommended_topic or recommended_topic not in topic_scores:
        return decision  # Can't validate, trust LLM decision
    
    topic_data = topic_scores[recommended_topic]
    score = topic_data.get("score", 0)
    attempts = topic_data.get("attempts", 1)
    llm_agent = decision.get("recommended_agent")
    
    # Get meta-learning recommendation for this topic
    meta_rec = meta_recommendations.get(recommended_topic, {})
    meta_agent = meta_rec.get("recommended_agent", llm_agent)
    
    # Score-based validation rules
    if score < 40:
        correct_agent = "quiz_agent"
        reason = "Low score requires quiz review"
    elif score < 60:
        correct_agent = "quiz_agent" 
        reason = "Moderate score needs quiz reinforcement"
    elif score < 80:
        correct_agent = "teaching_agent"
        reason = "Good score ready for teaching practice"
    else:
        # High score - consider meta-learning and attempts
        if attempts > 3 and meta_agent == "teaching_agent":
            correct_agent = "teaching_agent"
            reason = "High score with multiple attempts - continue teaching"
        else:
            correct_agent = "teaching_agent"
            reason = "High score ready for advanced teaching"
    
    # Check if correction needed
    if llm_agent != correct_agent:
        logger.warning(f"🔧 Routing correction: LLM suggested {llm_agent}, but score {score}% suggests {correct_agent}")
        decision["recommended_agent"] = correct_agent
        decision["reasoning"] = f"Meta-corrected: {decision.get('reasoning', '')} | Adjusted {llm_agent}→{correct_agent} based on {score}% score and meta-insights. {reason}"
        decision["meta_correction_applied"] = True
    
    # Apply meta-learning difficulty adjustment
    if meta_rec.get("difficulty"):
        original_difficulty = decision.get("expected_difficulty", "medium")
        meta_difficulty = meta_rec["difficulty"] 
        if original_difficulty != meta_difficulty:
            logger.info(f"🎯 Meta-learning difficulty adjustment: {original_difficulty} → {meta_difficulty}")
            decision["expected_difficulty"] = meta_difficulty
    
    return decision


# Node 4: Execute action using StateManager
def execute_action(state: BossAgentState) -> BossAgentState:
    """Execute action using StateManager for session creation"""
    logger.info("⚡ Executing action...")
    
    decision = state['decision']
    content_cache = state['content_cache']
    recommended_topic = decision.get("recommended_topic")
    
    if not recommended_topic:
        return {
            **state,
            "session_status": "error",
            "error_message": "No valid topic recommendation"
        }

    # Find topic info from content cache
    topic_info = None
    for doc in content_cache["documents"]:
        if doc["topic_name"] == recommended_topic:
            topic_info = doc
            break
    
    if not topic_info:
        return {
            **state,
            "session_status": "error",
            "error_message": f"Topic '{recommended_topic}' not found in content"
        }

    # Prepare session data for Quiz Agent
    quiz_session_config = {
        "md_file_path": topic_info["filepath"],
        "image_paths": [img["path"] for img in content_cache["images"]],
        "learning_strategy": decision.get("learning_strategy"),
        "expected_difficulty": decision.get("expected_difficulty"),
        "session_recommendation": decision.get("session_recommendation"),
        "word_count": topic_info["word_count"],
        "boss_agent_reasoning": decision.get("reasoning")
    }
    
    # Use StateManager to create quiz session
    try:
        state_manager = StateManager()
        session_file_path = state_manager.create_quiz_session(
            state['user_id'], 
            recommended_topic, 
            quiz_session_config
        )
        
        action_result = {
            "action": "route_to_quiz_agent",
            "topic": recommended_topic,
            "session_file": session_file_path,
            "status": "success"
        }
        
        return {
            **state,
            "action_result": action_result,
            "quiz_session_file": session_file_path,
            "session_status": "action_executed"
        }
        
    except Exception as e:
        logger.error(f"❌ Action execution failed: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "error_message": f"Failed to create quiz session: {str(e)}"
        }

# Node 5: Update progress using StateManager
def update_learning_progress(state: BossAgentState) -> BossAgentState:
    """Update learning progress using StateManager"""
    logger.info("📊 Updating learning progress...")
    
    quiz_results = state.get('quiz_results')
    decision = state['decision']
    topic_name = decision.get("recommended_topic")
    
    if not quiz_results or not topic_name:
        logger.warning("⚠️ No quiz results to update progress")
        return {**state, "session_status": "progress_update_skipped"}
    
    try:
        # Use StateManager to update progress
        state_manager = StateManager()
        success = state_manager.update_topic_progress(
            state['user_id'], 
            topic_name, 
            quiz_results
        )
        
        if success:
            # Reload updated user state
            updated_user_state = state_manager.load_user_state(state['user_id'])
            
            logger.info(f"📊 Updated progress: Topic '{topic_name}' scored {quiz_results.get('average_score', 0):.1f}")
            
            return {
                **state,
                "user_state": updated_user_state,
                "session_status": "progress_updated"
            }
        else:
            return {
                **state,
                "session_status": "error",
                "error_message": "Failed to update learning progress"
            }
            
    except Exception as e:
        logger.error(f"❌ Progress update failed: {str(e)}")
        return {
            **state,
            "session_status": "error",
            "error_message": f"Progress update failed: {str(e)}"
        }

# Build the workflow (same structure)
def create_boss_agent_workflow():
    """Create the Boss Agent LangGraph workflow"""
    workflow = StateGraph(BossAgentState)
    
    # Add nodes
    workflow.add_node("initialize_boss_agent", initialize_boss_agent)
    workflow.add_node("discover_content", discover_content)
    workflow.add_node("react_reasoning", react_reasoning)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("update_learning_progress", update_learning_progress)
    
    # Define edges
    workflow.set_entry_point("initialize_boss_agent")
    workflow.add_edge("initialize_boss_agent", "discover_content")
    workflow.add_edge("discover_content", "react_reasoning")
    workflow.add_edge("react_reasoning", "execute_action")
    workflow.add_edge("execute_action", "update_learning_progress")
    workflow.add_edge("update_learning_progress", END)
    
    return workflow.compile()

# Main function (same interface)
def run_boss_agent_session(user_id: str, content_folder_path: str, quiz_results: Optional[Dict[str, Any]] = None):
    """Run Boss Agent session using modular tools"""
    print(f"\n🤖 Boss Agent Session - Using Modular Tools")
    print(f"👤 User: {user_id}")
    print(f"📁 Content: {content_folder_path}")
    print("=" * 60)
    
    # Initialize state
    initial_state: BossAgentState = {
        "user_id": user_id,
        "content_folder_path": content_folder_path,
        "content_cache": {},
        "user_state": {},
        "reasoning_context": {},
        "decision": {},
        "action_result": {},
        "quiz_session_file": "",
        "quiz_results": quiz_results,
        "session_status": "starting",
        "error_message": ""
    }
    
    # Create and run workflow
    try:
        workflow = create_boss_agent_workflow()
        result = workflow.invoke(initial_state)
        
        print(f"\n✅ Boss Agent Session Complete!")
        print(f"📊 Status: {result['session_status']}")
        
        if result['session_status'] in ["action_executed", "progress_updated"]:
            decision = result.get('decision', {})
            
            print(f"🎯 Recommended Topic: {decision.get('recommended_topic')}")
            print(f"📚 Learning Strategy: {decision.get('learning_strategy')}")
            print(f"📄 Quiz Session File: {result.get('quiz_session_file')}")
            print(f"🏆 Overall Progress: {result['user_state'].get('overall_progress', 0):.1f}%")
        
        elif result.get('error_message'):
            print(f"❌ Error: {result['error_message']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Boss Agent workflow failed: {str(e)}")
        return {
            "session_status": "workflow_error",
            "error_message": str(e)
        }
