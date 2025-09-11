# import os
import asyncio
import json
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# LangSmith imports
from langsmith import traceable, Client as LangSmithClient
from langsmith.run_helpers import get_current_run_tree

# Load environment variables from .env file
load_dotenv()

# Configure LangSmith
os.environ["LANGSMITH_TRACING"] = "true" 
os.environ["LANGSMITH_PROJECT"] = "adaptive-tutoring-system"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import meta-learning module
from tools.meta_learning import MetaLearningModule

# Initialize LangSmith client
langsmith_client = LangSmithClient()

class SessionTracker:
    """Track session metrics for LangSmith visualization"""
    
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.total_tokens = 0
        self.total_cost = 0.0
        self.agent_calls = []
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens from text (rough approximation)"""
        return len(str(text).split()) * 1.3
    
    def estimate_cost(self, tokens: int, model: str = "groq-llama") -> float:
        """Estimate cost based on token usage"""
        cost_per_1k = {
            "groq-llama": 0.0002,  # Groq Llama pricing
            "gpt-4o": 0.015,
            "gpt-4o-mini": 0.0001
        }
        return (tokens / 1000) * cost_per_1k.get(model, 0.0002)
    
    def track_agent_call(self, agent_name: str, result: Dict, duration: float):
        """Track individual agent performance"""
        tokens = self.estimate_tokens(json.dumps(result))
        cost = self.estimate_cost(tokens)
        
        call_data = {
            "agent": agent_name,
            "tokens": tokens,
            "cost": cost,
            "duration": duration,
            "score": result.get('average_score', 0),
            "status": result.get('status', 'unknown')
        }
        
        self.agent_calls.append(call_data)
        self.total_tokens += tokens
        self.total_cost += cost
        
        # Log to current LangSmith run
        current_run = get_current_run_tree()
        if current_run:
            current_run.add_metadata({
                f"{agent_name}_metrics": call_data
            })
        
        logger.info(f"📊 {agent_name}: {tokens} tokens, ${cost:.4f}, {duration:.2f}s")

# Initialize session tracker
tracker = SessionTracker()

@traceable(
    name="complete_tutoring_workflow",
    run_type="chain", 
    tags=["multi-agent", "adaptive-learning", "meta-learning", "async-mcp"],
    metadata={"system": "adaptive-tutoring", "version": "3.0-async"}
)
async def run_integrated_learning_system(user_id: str, content_folder_path: str) -> Dict[str, Any]:
    """
    Complete ASYNC integrated learning system workflow with MCP + meta-learning + LangSmith tracing:
    Boss Agent → Quiz Agent OR Teaching Agent → Meta-Learning Update → Boss Agent (with results)
    """
    print(f"\n🤖 ASYNC INTEGRATED LEARNING SYSTEM WITH MCP + META-LEARNING + LANGSMITH")
    print(f"="*75)
    print(f"👤 User: {user_id}")
    print(f"📁 Content: {content_folder_path}")
    print(f"🔍 Session ID: {tracker.session_id}")
    print(f"⚡ Mode: ASYNC + MCP")
    print(f"="*75)
    
    # Initialize meta-learning module
    meta_module = MetaLearningModule()
    
    # Add session info to LangSmith trace
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_metadata({
            "session_id": tracker.session_id,
            "user_id": user_id,
            "content_folder": content_folder_path,
            "start_time": datetime.now().isoformat(),
            "mode": "async_mcp"
        })
    
    try:
        # Step 1: Boss Agent Decision Making (ASYNC)
        boss_result = await execute_boss_agent_step(user_id, content_folder_path)
        
        if boss_result.get('status') != 'success':
            return handle_boss_agent_failure(boss_result)

        # Extract recommendation  
        decision = boss_result.get('decision', {})
        recommended_topic = decision.get('recommended_topic')
        recommended_agent = decision.get('recommended_agent', 'quiz_agent')
        
        if not recommended_topic:
            return handle_no_topic_error(boss_result)
        
        print(f"✅ Boss Agent recommends studying: '{recommended_topic}'")
        print(f"📚 Learning Strategy: {decision.get('learning_strategy', 'unknown')}")
        print(f"🎯 Recommended Agent: {recommended_agent}")
        
        # Show pre-session meta-learning insights
        show_pre_session_insights(meta_module, user_id, recommended_topic)
        
        # Step 2: Execute Selected Agent (ASYNC)
        agent_result, agent_type, session_duration = await execute_selected_agent(
            recommended_agent, user_id, recommended_topic
        )
        
        if agent_result.get('status') != 'success':
            return handle_agent_failure(agent_result, agent_type, boss_result)
        
        print(f"✅ {agent_type} completed!")
        print(f"🏆 Final Score: {agent_result['average_score']:.2f}/100")
        print(f"🎯 Grade: {agent_result['grade']}")
        
        # Step 3: Record Meta-Learning Data
        record_meta_learning_data(
            meta_module, user_id, recommended_topic, recommended_agent, 
            agent_result, session_duration, decision
        )
        
        # Step 4: Show Post-Session Insights  
        post_insights, agent_adaptations = show_post_session_insights(
            meta_module, user_id, recommended_topic, recommended_agent
        )
        
        # Step 5: Update Progress (ASYNC)
        final_boss_result, overall_progress = await update_learning_progress(
            user_id, content_folder_path, agent_result, agent_type
        )
        
        # Final Summary
        display_final_summary(
            recommended_topic, agent_type, agent_result, overall_progress, 
            session_duration, post_insights
        )
        
        # Log final metrics to LangSmith
        log_session_metrics_to_langsmith(
            user_id, recommended_topic, agent_type, agent_result, 
            session_duration, decision
        )
        
        return build_success_response(
            user_id, recommended_topic, agent_type, decision, agent_result,
            final_boss_result, session_duration, post_insights, agent_adaptations
        )
        
    except ImportError as e:
        return handle_import_error(e)
    except Exception as e:
        return handle_unexpected_error(e)

@traceable(
    name="boss_agent_routing_decision",
    run_type="llm",
    tags=["routing", "decision-making", "boss-agent", "async"]
)
async def execute_boss_agent_step(user_id: str, content_folder_path: str) -> Dict[str, Any]:
    """Execute Boss Agent routing decision with tracing (ASYNC)"""
    print(f"\n🧠 STEP 1: Boss Agent Decision Making (ASYNC)...")
    
    start_time = time.time()
    
    # Import boss agent with async support
    from boss_agent import run_boss_agent_session
    
    # 🔧 KEY CHANGE: Check if boss agent is async, if not run in executor
    try:
        # Try async first
        boss_result = await run_boss_agent_session(user_id, content_folder_path, quiz_results=None)
    except TypeError:
        # Fallback to sync version in executor
        boss_result = await asyncio.to_thread(
            run_boss_agent_session, user_id, content_folder_path, None
        )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Track metrics
    tracker.track_agent_call("boss_agent", boss_result, duration)
    
    # Add to trace
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_metadata({
            "routing_decision": boss_result.get('decision', {}),
            "session_status": boss_result.get('session_status'),
            "duration": duration,
            "async_mode": True
        })
    
    if boss_result.get('session_status') in ['error', 'workflow_error']:
        return {"status": "boss_agent_failed", "boss_result": boss_result}
    
    return {"status": "success", "decision": boss_result.get('decision', {})}

@traceable(
    name="learning_agent_execution", 
    run_type="chain",
    tags=["quiz-agent", "teaching-agent", "learning", "async-mcp"]
)
async def execute_selected_agent(recommended_agent: str, user_id: str, recommended_topic: str):
    """Execute the selected learning agent with tracing (ASYNC + MCP)"""
    
    session_start_time = time.time()
    
    if recommended_agent == 'teaching_agent':
        print(f"\n🎓 STEP 2: Teaching Agent Session (ASYNC + MCP)...")
        from teaching_agent import run_teaching_agent_session
        
        try:
            # Try async MCP-enabled version first
            agent_result = await run_teaching_agent_session(user_id, recommended_topic)
        except TypeError:
            # Fallback to sync version in executor
            agent_result = await asyncio.to_thread(
                run_teaching_agent_session, user_id, recommended_topic
            )
        
        agent_type = "Teaching Agent"
    else:
        print(f"\n🎯 STEP 2: Quiz Agent Session (ASYNC + MCP)...")  
        from quiz_agent import run_quiz_agent_session
        
        try:
            # Try async MCP-enabled version first  
            agent_result = await run_quiz_agent_session(user_id, recommended_topic)
        except TypeError:
            # Fallback to sync version in executor
            agent_result = await asyncio.to_thread(
                run_quiz_agent_session, user_id, recommended_topic
            )
        
        agent_type = "Quiz Agent"
    
    session_end_time = time.time()
    session_duration = session_end_time - session_start_time
    
    # Track agent performance
    tracker.track_agent_call(recommended_agent, agent_result, session_duration)
    
    # Add detailed metrics to trace
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_metadata({
            "agent_type": agent_type,
            "questions_answered": agent_result.get('questions_answered', 0),
            "average_score": agent_result.get('average_score', 0),
            "grade": agent_result.get('grade', ''),
            "session_duration": session_duration,
            "async_mode": True,
            "mcp_enabled": True
        })
    
    return agent_result, agent_type, session_duration

@traceable(
    name="meta_learning_update",
    run_type="tool", 
    tags=["meta-learning", "adaptation", "personalization"]
)
def record_meta_learning_data(meta_module, user_id, recommended_topic, recommended_agent, 
                            agent_result, session_duration, decision):
    """Record session data for meta-learning with tracing"""
    print(f"\n📊 STEP 3: Recording Meta-Learning Data...")
    
    try:
        # Extract mistakes from QA history
        mistakes = []
        qa_history = agent_result.get('qa_history', [])
        for qa in qa_history:
            if qa.get('score', 100) < 70:
                mistakes.append({
                    "question": qa.get('question', ''),
                    "user_answer": qa.get('answer', ''),
                    "expected_answer": qa.get('expected_answer', ''),
                    "type": classify_mistake_type(qa),
                    "score": qa.get('score', 0)
                })
        
        # Prepare session data
        session_outcome = {
            "average_score": agent_result['average_score'],
            "questions_answered": agent_result.get('questions_answered', 0),
            "mistakes": mistakes,
            "time_taken": session_duration,
            "feedback_rating": 4,
            "agent_used": recommended_agent,
            "difficulty_level": decision.get('expected_difficulty', 'medium'),
            "learning_strategy": decision.get('learning_strategy', 'unknown'),
            "async_mode": True,
            "mcp_enabled": True
        }
        
        # Record to meta-learning system
        meta_module.record_session_outcome(
            user_id, recommended_topic, recommended_agent, session_outcome
        )
        
        # Add to LangSmith trace
        current_run = get_current_run_tree()
        if current_run:
            current_run.add_metadata({
                "meta_learning_data": {
                    "mistakes_count": len(mistakes),
                    "mistake_types": list(set([m["type"] for m in mistakes])),
                    "session_outcome": session_outcome,
                    "async_mode": True
                }
            })
        
        print(f"✅ Meta-learning data recorded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to record meta-learning data: {e}")
        print(f"⚠️ Warning: Meta-learning data recording failed: {e}")

def show_pre_session_insights(meta_module, user_id, recommended_topic):
    """Display pre-session meta-learning insights"""
    try:
        pre_insights = meta_module.get_learning_recommendations(user_id, recommended_topic)
        print(f"\n🧠 PRE-SESSION META-LEARNING INSIGHTS:")
        print(f"📈 Learning Velocity: {pre_insights.get('learning_velocity', 'N/A')}")
        print(f"🎯 Recommended Agent: {pre_insights.get('recommended_agent', 'N/A')}")
        print(f"🔍 Focus Areas: {', '.join(pre_insights.get('focus_areas', []))}")
        print(f"💡 Reasoning: {pre_insights.get('reasoning', 'Starting new topic')}")
    except Exception as e:
        logger.warning(f"Failed to get pre-session meta-learning insights: {e}")

def show_post_session_insights(meta_module, user_id, recommended_topic, recommended_agent):
    """Display post-session meta-learning insights and adaptations"""
    try:
        post_insights = meta_module.get_learning_recommendations(user_id, recommended_topic)
        agent_adaptations = meta_module.get_agent_adaptations(user_id, recommended_topic, recommended_agent)
        
        print(f"\n🧠 POST-SESSION META-LEARNING INSIGHTS:")
        print(f"📈 Updated Learning Velocity: {post_insights.get('learning_velocity', 'N/A'):.3f}")
        
        effectiveness = post_insights.get('agent_effectiveness', {})
        quiz_eff = effectiveness.get('quiz', 0)
        teaching_eff = effectiveness.get('teaching', 0)
        print(f"🎯 Agent Effectiveness: Quiz={quiz_eff:.2f}, Teaching={teaching_eff:.2f}")
        
        focus_areas = post_insights.get('focus_areas', [])
        if focus_areas:
            print(f"🔍 Focus Areas: {', '.join(focus_areas[:3])}")
        else:
            print(f"🔍 Focus Areas: No specific patterns detected yet")
        
        print(f"💡 Next Session Reasoning: {post_insights.get('reasoning', 'Continue learning')}")
        
        # Show adaptations
        if agent_adaptations:
            print(f"\n🔧 ADAPTIVE RECOMMENDATIONS:")
            print(f"   Optimal Difficulty: {agent_adaptations.get('difficulty_adjustment', 'medium')}")
            if recommended_agent == 'quiz_agent':
                print(f"   Question Pacing: {agent_adaptations.get('pacing', 'standard')}")
                print(f"   Question Types: {', '.join(agent_adaptations.get('question_types', ['standard']))}")
            elif recommended_agent == 'teaching_agent':
                print(f"   Explanation Style: {agent_adaptations.get('explanation_style', 'standard')}")
                print(f"   Prompt Complexity: {agent_adaptations.get('prompt_complexity', 'medium')}")
        
        return post_insights, agent_adaptations
        
    except Exception as e:
        logger.error(f"Failed to get post-session meta-learning insights: {e}")
        print(f"⚠️ Warning: Could not retrieve updated meta-learning insights: {e}")
        return None, None

@traceable(
    name="progress_update",
    run_type="tool",
    tags=["state-management", "progress-tracking", "async"]  
)
async def update_learning_progress(user_id, content_folder_path, agent_result, agent_type):
    """Update learning progress with tracing (ASYNC)"""
    print(f"\n📊 STEP 5: Updating Boss Agent with {agent_type} Results (ASYNC)...")
    
    start_time = time.time()
    
    from boss_agent import run_boss_agent_session
    
    try:
        # Try async version first
        final_boss_result = await run_boss_agent_session(
            user_id, content_folder_path, quiz_results=agent_result
        )
    except TypeError:
        # Fallback to sync version in executor
        final_boss_result = await asyncio.to_thread(
            run_boss_agent_session, user_id, content_folder_path, agent_result
        )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Track progress update
    tracker.track_agent_call("progress_update", final_boss_result, duration)
    
    if final_boss_result.get('session_status') not in ['progress_updated', 'completed_successfully']:
        print(f"⚠️ Warning: Progress update may have failed: {final_boss_result.get('error_message')}")
    else:
        print(f"✅ Learning progress updated successfully!")
    
    updated_user_state = final_boss_result.get('user_state', {})
    overall_progress = updated_user_state.get('overall_progress', 0.0)
    
    return final_boss_result, overall_progress

def log_session_metrics_to_langsmith(user_id, recommended_topic, agent_type, 
                                   agent_result, session_duration, decision):
    """Log comprehensive session metrics to LangSmith"""
    
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_metadata({
            "session_summary": {
                "user_id": user_id,
                "topic_studied": recommended_topic,
                "agent_used": agent_type,
                "final_score": agent_result['average_score'],
                "grade": agent_result['grade'],
                "questions_answered": agent_result['questions_answered'],
                "session_duration": session_duration,
                "learning_strategy": decision.get('learning_strategy'),
                "async_mode": True,
                "mcp_enabled": True
            },
            "performance_metrics": {
                "total_tokens": tracker.total_tokens,
                "total_cost": tracker.total_cost,
                "agent_calls": len(tracker.agent_calls),
                "tokens_per_second": tracker.total_tokens / session_duration if session_duration > 0 else 0,
                "cost_per_minute": (tracker.total_cost / session_duration) * 60 if session_duration > 0 else 0
            },
            "agent_breakdown": tracker.agent_calls
        })
    
    print(f"\n📊 LANGSMITH METRICS LOGGED:")
    print(f"🪙 Total Tokens: {tracker.total_tokens:,}")
    print(f"💰 Estimated Cost: ${tracker.total_cost:.4f}")
    print(f"⏱️  Session Duration: {session_duration:.1f}s")
    print(f"⚡ Async Mode: Enabled")
    print(f"🔗 MCP Integration: Active")
    print(f"🔗 View traces at: https://smith.langchain.com")

def display_final_summary(recommended_topic, agent_type, agent_result, 
                        overall_progress, session_duration, post_insights):
    """Display final session summary"""
    print(f"\n🎉 ASYNC + MCP LEARNING SESSION COMPLETE!")
    print(f"="*75)
    print(f"📚 Topic Studied: {recommended_topic}")
    print(f"🤖 Agent Used: {agent_type}")
    print(f"❓ Questions/Prompts Answered: {agent_result['questions_answered']}")
    print(f"🏆 Session Score: {agent_result['average_score']:.2f}/100")
    print(f"🎯 Grade: {agent_result['grade']}")
    print(f"📈 Overall Progress: {overall_progress:.1f}%")
    print(f"💡 Recommendation: {agent_result['recommendation']}")
    print(f"🧠 Meta-Learning: {'Active' if post_insights else 'Initializing'}")
    print(f"⏱️ Session Duration: {session_duration:.1f} seconds")
    print(f"⚡ Async Mode: Enabled")
    print(f"🔗 MCP Integration: Active")
    print(f"🔍 LangSmith Session: {tracker.session_id}")
    print(f"="*75)

# Helper functions for error handling (unchanged)
def handle_boss_agent_failure(boss_result):
    print(f"❌ Boss Agent failed: {boss_result.get('error_message', 'Unknown error')}")
    return {
        "status": "boss_agent_failed",
        "error": boss_result.get('error_message', 'Boss Agent did not complete successfully'),
        "boss_result": boss_result
    }

def handle_no_topic_error(boss_result):
    print(f"❌ Boss Agent did not recommend a topic")
    return {
        "status": "no_topic_recommended", 
        "error": "Boss Agent failed to recommend a topic",
        "boss_result": boss_result
    }

def handle_agent_failure(agent_result, agent_type, boss_result):
    print(f"❌ {agent_type} failed: {agent_result.get('error', 'Unknown error')}")
    return {
        "status": "agent_failed",
        "error": agent_result.get('error', f'{agent_type} failed'),
        "boss_result": boss_result,
        "agent_result": agent_result,
        "agent_type": agent_type
    }

def handle_import_error(e):
    error_msg = f"Missing module: {str(e)}"
    print(f"❌ Import Error: {error_msg}")
    return {"status": "import_error", "error": error_msg}

def handle_unexpected_error(e):
    error_msg = f"Unexpected error: {str(e)}"
    print(f"❌ Unexpected Error: {error_msg}")
    logger.exception("Integration workflow failed")
    return {"status": "unexpected_error", "error": error_msg}

def build_success_response(user_id, recommended_topic, agent_type, decision, agent_result,
                          final_boss_result, session_duration, post_insights, agent_adaptations):
    """Build successful response with all data"""
    updated_user_state = final_boss_result.get('user_state', {})
    overall_progress = updated_user_state.get('overall_progress', 0.0)
    
    return {
        "status": "success",
        "user_id": user_id,
        "session_summary": {
            "topic_studied": recommended_topic,
            "agent_used": agent_type,
            "learning_strategy": decision.get('learning_strategy'),
            "questions_answered": agent_result['questions_answered'],
            "session_score": agent_result['average_score'],
            "grade": agent_result['grade'],
            "overall_progress": overall_progress,
            "session_duration": session_duration,
            "meta_learning_active": True,
            "langsmith_session_id": tracker.session_id,
            "total_tokens": tracker.total_tokens,
            "estimated_cost": tracker.total_cost,
            "async_mode": True,
            "mcp_enabled": True
        },
        "boss_decision": decision,
        "agent_results": agent_result,
        "updated_user_state": updated_user_state,
        "meta_learning_insights": post_insights,
        "meta_adaptations": agent_adaptations,
        "langsmith_metrics": {
            "session_id": tracker.session_id,
            "total_tokens": tracker.total_tokens,
            "total_cost": tracker.total_cost,
            "agent_calls": tracker.agent_calls
        }
    }

def classify_mistake_type(qa_entry: Dict[str, Any]) -> str:
    """Classify the type of mistake for meta-learning analysis"""
    question = qa_entry.get('question', '').lower()
    user_answer = qa_entry.get('answer', '').lower()
    
    if 'what is' in question or 'define' in question:
        return 'definition_error'
    elif 'how' in question or 'explain' in question:
        return 'explanation_error'
    elif 'why' in question:
        return 'reasoning_error'
    elif len(user_answer) < 10:
        return 'incomplete_answer'
    else:
        return 'general_error'

async def main():
    """ASYNC Main function with user input"""
    print("🎓 ASYNC MCP-Enhanced Integrated Learning System")
    print("Boss Agent + Quiz/Teaching Agents + Meta-Learning + LangSmith + MCP")
    print("="*100)
    
    # Get user input (these remain synchronous)
    try:
        user_id = input("Enter User ID: ").strip() or "student_001"
        content_folder = input("Enter content folder path: ").strip() or "./content"
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Validate content folder exists
    if not os.path.exists(content_folder):
        print(f"❌ Content folder does not exist: {content_folder}")
        print("Please create the folder and add some .md files for learning content.")
        return
    
    # 🔧 KEY CHANGE: AWAIT the async integrated system
    result = await run_integrated_learning_system(user_id, content_folder)
    
    # Handle different outcomes
    if result["status"] == "success":
        print(f"\n✅ ASYNC MCP-enhanced integration completed successfully!")
        
        # Show LangSmith metrics
        langsmith_metrics = result.get('langsmith_metrics', {})
        if langsmith_metrics:
            print(f"\n📊 LANGSMITH ANALYTICS:")
            print(f"🔍 Session ID: {langsmith_metrics.get('session_id')}")
            print(f"🪙 Total Tokens Used: {langsmith_metrics.get('total_tokens', 0):,}")
            print(f"💰 Estimated Cost: ${langsmith_metrics.get('total_cost', 0):.4f}")
            print(f"📞 API Calls Made: {len(langsmith_metrics.get('agent_calls', []))}")
            print(f"⚡ Async Mode: Enabled")
            print(f"🔗 MCP Integration: Active")
            print(f"🔗 View detailed traces at: https://smith.langchain.com")
        
        # Show meta-learning summary
        meta_insights = result.get('meta_learning_insights')
        if meta_insights:
            print(f"\n🧠 META-LEARNING SUMMARY:")
            print(f"Learning velocity: {meta_insights.get('learning_velocity', 'N/A')}")
            effectiveness = meta_insights.get('agent_effectiveness', {})
            if effectiveness:
                print(f"Agent effectiveness: Quiz={effectiveness.get('quiz', 0):.2f}, Teaching={effectiveness.get('teaching', 0):.2f}")
            
            focus_areas = meta_insights.get('focus_areas', [])
            if focus_areas:
                print(f"Focus areas: {', '.join(focus_areas[:2])}")
        
        # Ask if user wants to run another session
        try:
            another = input("\nWould you like to run another learning session? (y/n): ").strip().lower()
            if another == 'y' or another == 'yes':
                # Reset tracker for new session
                global tracker
                tracker = SessionTracker()
                await main()  # 🔧 KEY CHANGE: AWAIT recursive call
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print(f"\n❌ Integration failed with status: {result['status']}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Debugging info
        if result.get('boss_result'):
            print(f"Boss Agent Status: {result['boss_result'].get('session_status')}")
        if result.get('agent_result'):
            print(f"Agent Status: {result['agent_result'].get('status')}")
            if result.get('agent_type'):
                print(f"Agent Type: {result['agent_type']}")

if __name__ == "__main__":
    # 🔧 KEY CHANGE: Use asyncio.run() for async main
    asyncio.run(main())
