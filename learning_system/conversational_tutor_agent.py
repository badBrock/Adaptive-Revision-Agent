# conversational_tutor_agent.py
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List, TypedDict
import json
import logging
from datetime import datetime
from tools.conversational_tutoring_tools import CHATBOT_TOOLS
from groq import Groq
import os

logger = logging.getLogger(__name__)

class ConversationalTutorState(TypedDict):
    # Session identifiers
    user_id: str
    session_id: str
    original_topic: str
    entry_agent: str
    
    # Conversation state
    messages: List[Dict[str, Any]]
    current_concept: str
    understanding_level: int  # 1-10 scale
    mastery_counter: int  # Correct answers toward goal of 5
    tool_iterations: List[Dict[str, Any]]
    
    # Long-term memory (persisted across sessions)
    learning_history: Dict[str, Any]
    struggle_patterns: Dict[str, str]
    effective_methods: List[str]
    mastered_concepts: List[str]
    
    # Session control
    session_status: str
    max_iterations: int
    current_iteration: int
    current_decision: Dict[str, Any]
    last_user_input: str
    last_evaluation: Dict[str, Any]
    exit_requested: bool
class ConversationalTutorAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.tools = {tool["name"]: tool["func"] for tool in CHATBOT_TOOLS}
        self.tool_descriptions = {tool["name"]: tool["description"] for tool in CHATBOT_TOOLS}

    def initialize_session(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Initialize conversational tutoring session"""
        logger.info(f"🎓 Starting conversational tutoring for {state['user_id']} on {state['current_concept']}")
        
        # Welcome message
        print(f"\n🎓 CONVERSATIONAL TUTORING SESSION STARTED")
        print(f"📚 Topic: {state['current_concept']}")
        print(f"🎯 Goal: Master this concept through interactive learning")
        print(f"✨ I'll help you understand step-by-step until you're confident!")
        
        return {
            **state,
            "understanding_level": 2,  # Start assuming low understanding
            "mastery_counter": 0,
            "tool_iterations": [],
            "current_iteration": 0,
            "session_status": "active",
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": f"Let's work together to understand {state['current_concept']}. I'm here to help!"}
            ]
        }

    def assess_and_decide(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """LLM judges current understanding and decides next action"""
        recent_messages = state["messages"][-5:] if state["messages"] else []
        
        judge_prompt = f"""
        You are evaluating a tutoring conversation to decide the next step.
        
        Student: {state['user_id']}
        Topic: {state['current_concept']} 
        Understanding level: {state['understanding_level']}/10
        Mastery progress: {state['mastery_counter']}/5 correct applications
        Current iteration: {state['current_iteration']}/{state['max_iterations']}
        
        Recent conversation:
        {json.dumps(recent_messages, indent=2)}
        
        Available tools:
        - refine_explanation: {self.tool_descriptions['refine_explanation']}
        - micro_lesson_generator: {self.tool_descriptions['micro_lesson_generator']}  
        - generate_leading_question: {self.tool_descriptions['generate_leading_question']}
        
        Based on the conversation, decide what to do next.
        
        Respond with JSON:
        {{
            "action": "use_tool" or "test_mastery" or "session_complete",
            "tool_name": "tool_name_if_using_tool", 
            "reasoning": "why this choice",
            "tool_input": {{"concept": "{state['current_concept']}", "context": "relevant_context"}},
            "confidence": 0.8
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": judge_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=300,
                temperature=0.3
            )
            
            decision_text = response.choices[0].message.content.strip()
            decision = self._parse_json_decision(decision_text)
            
            logger.info(f"🤖 LLM Decision: {decision['action']} - {decision['reasoning']}")
            
            return {
                **state,
                "current_decision": decision,
                "session_status": "decision_made"
            }
            
        except Exception as e:
            logger.error(f"❌ Assessment failed: {e}")
            # Fallback decision
            decision = {
                "action": "use_tool",
                "tool_name": "refine_explanation",
                "reasoning": "Fallback: Continue with explanation refinement",
                "tool_input": {"current_explanation": "Let me explain this differently", "concept": state['current_concept']},
                "confidence": 0.5
            }
            
            return {
                **state,
                "current_decision": decision,
                "session_status": "fallback_decision"
            }

    def execute_tool_action(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Execute the chosen tool based on LLM decision"""
        decision = state.get("current_decision", {})
        action = decision.get("action")
        
        if action != "use_tool":
            return {**state, "session_status": "no_tool_needed"}
        
        tool_name = decision.get("tool_name")
        tool_input = decision.get("tool_input", {})
        
        if tool_name not in self.tools:
            logger.error(f"❌ Unknown tool: {tool_name}")
            return {**state, "session_status": "tool_error"}
        
        # Add conversation context to tool input
        enhanced_input = {
            **tool_input,
            "conversation_history": state["messages"],
            "topic": state["current_concept"],
            "learner_level": state["understanding_level"],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logger.info(f"🔧 Using tool: {tool_name}")
            tool_output = self.tools[tool_name](enhanced_input)
            
            # Extract tool response content
            tool_response = (
                tool_output.get("refined_explanation") or 
                tool_output.get("micro_lesson") or 
                tool_output.get("leading_question") or 
                "Tool completed successfully"
            )
            
            # Display tool output to user
            print(f"\n🎓 Tutor: {tool_response}")
            
            # Add tool interaction to conversation
            tool_interaction = {
                "timestamp": datetime.now().isoformat(),
                "tool_used": tool_name,
                "tool_input": enhanced_input,
                "tool_output": tool_output,
                "reasoning": decision.get("reasoning", "")
            }
            
            updated_iterations = state["tool_iterations"] + [tool_interaction]
            updated_messages = state["messages"] + [
                {"role": "system", "content": f"Used {tool_name}: {decision.get('reasoning', '')}"},
                {"role": "assistant", "content": tool_response}
            ]
            
            return {
                **state,
                "tool_iterations": updated_iterations,
                "messages": updated_messages,
                "current_iteration": state["current_iteration"] + 1,
                "session_status": "tool_executed"
            }
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            return {
                **state,
                "session_status": "tool_execution_error",
                "error_message": str(e)
            }

    def collect_user_response(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Get user input and update conversation state"""
        
        user_input = input("👤 Your response: ").strip()
        
        # 🔧 NEW: Check for exit commands
        exit_phrases = ['done', "i'm done", 'exit', 'quit', 'back', 'return', 'stop']
        if user_input.lower() in exit_phrases:
            print("🚪 Thanks for learning with me! Returning to your main session...")
            logger.info(f"👤 User requested exit: '{user_input}'")
            
            return {
                **state,
                "exit_requested": True,
                "last_user_input": user_input,
                "session_status": "user_exit_requested",
                "messages": state["messages"] + [
                    {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()}
                ]
            }
        
        # Normal processing for other inputs
        updated_messages = state["messages"] + [
            {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()}
        ]
        
        logger.info(f"👤 User responded: '{user_input}'")
        
        return {
            **state,
            "messages": updated_messages,
            "last_user_input": user_input,
            "session_status": "user_responded",
            "exit_requested": False  # Ensure it's set to False for normal responses
        }


    def check_mastery_progress(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Judge if user has progressed toward mastery"""
        last_user_input = state.get("last_user_input", "")
        
        mastery_prompt = f"""
        Evaluate this student response for understanding of {state['current_concept']}:
        
        Student response: "{last_user_input}"
        Context: Tutoring session on {state['current_concept']}
        Current understanding level: {state['understanding_level']}/10
        Previous mastery count: {state['mastery_counter']}/5
        
        Rate the response:
        1. Shows understanding (0-10)
        2. Demonstrates confidence (0-10) 
        3. Ready for challenge question (yes/no)
        
        Respond with JSON:
        {{
            "understanding_score": 7,
            "confidence_score": 6,
            "ready_for_challenge": true,
            "understanding_change": 1,
            "mastery_increment": 1,
            "feedback": "brief encouraging feedback"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": mastery_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=200,
                temperature=0.3
            )
            
            evaluation = self._parse_json_decision(response.choices[0].message.content.strip())
            
            # 🔧 FIXED: Explicit type conversion with fallbacks
            try:
                understanding_change = int(evaluation.get("understanding_change", 0))
            except (ValueError, TypeError):
                understanding_change = 0
                logger.warning("Could not parse understanding_change, defaulting to 0")
            
            try:
                mastery_increment = int(evaluation.get("mastery_increment", 0))
            except (ValueError, TypeError):
                mastery_increment = 0
                logger.warning("Could not parse mastery_increment, defaulting to 0")
            
            # Safe arithmetic with guaranteed integers
            new_understanding = min(10, max(1, int(state["understanding_level"]) + understanding_change))
            new_mastery_counter = min(5, int(state["mastery_counter"]) + mastery_increment)
            
            if mastery_increment > 0:
                print(f"🎉 Great progress! Mastery: {new_mastery_counter}/5")
            
            return {
                **state,
                "understanding_level": new_understanding,
                "mastery_counter": new_mastery_counter,
                "last_evaluation": evaluation,
                "session_status": "mastery_evaluated"
            }
            
        except Exception as e:
            logger.error(f"❌ Mastery evaluation failed: {e}")
            # Safe fallback with no changes
            return {
                **state, 
                "session_status": "evaluation_error",
                "last_evaluation": {"feedback": "Evaluation failed, continuing session"}
            }

    def session_manager(self, state: ConversationalTutorState) -> str:
        """Decide whether to continue, test mastery, or end session with strict conditions"""
        # 🔧 NEW: Priority check for user exit request
        if state.get("exit_requested", False):
            logger.info("🚪 User requested exit - ending tutoring session")
            return "exit_session"
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 20)
        mastery_counter = state.get("mastery_counter", 0)
        understanding_level = state.get("understanding_level", 0)
        
        # 🔧 FIXED: Multiple explicit stopping conditions
        
        # Force stop if max iterations reached
        if current_iteration >= max_iterations:
            logger.info(f"🛑 Stopping: Max iterations ({max_iterations}) reached")
            return "end_session"
        
        # Stop if mastery achieved
        if mastery_counter >= 5:
            logger.info(f"🎉 Stopping: Mastery achieved ({mastery_counter}/5)")
            return "end_session"
        
        # Stop if understanding is high and we have some mastery
        if understanding_level >= 8 and mastery_counter >= 3:
            logger.info(f"🎯 Stopping: High understanding ({understanding_level}/10) with good mastery ({mastery_counter}/5)")
            return "end_session"
        
        # Force stop if stuck (same understanding for too long)
        if current_iteration > 10 and understanding_level <= 3:
            logger.info(f"🛑 Stopping: Stuck at low understanding after {current_iteration} iterations")
            return "end_session"
        
        # Emergency stop if too many iterations
        if current_iteration >= 25:
            logger.warning(f"🚨 Emergency stop at iteration {current_iteration}")
            return "end_session"
        
        # Continue tutoring
        logger.info(f"🔄 Continuing: Iteration {current_iteration}/{max_iterations}, Understanding {understanding_level}/10, Mastery {mastery_counter}/5")
        return "continue_tutoring"

    def end_session(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Finalize session and update long-term memory"""
        
        # Update learning history
        updated_learning_history = state.get("learning_history", {})
        updated_learning_history[state["current_concept"]] = {
            "final_understanding": state["understanding_level"],
            "mastery_achieved": state["mastery_counter"] >= 5,
            "session_duration": state["current_iteration"],
            "effective_tools": [t["tool_used"] for t in state["tool_iterations"] if t.get("tool_output", {}).get("success", False)],
            "last_session": datetime.now().isoformat()
        }
        
        # Determine session outcome
        if state["mastery_counter"] >= 5:
            outcome = "mastery_achieved"
            message = f"🎉 Excellent! You've mastered {state['current_concept']}!"
        elif state["understanding_level"] >= 6:
            outcome = "good_progress"
            message = f"👍 Good progress on {state['current_concept']}. Let's continue next time!"
        else:
            outcome = "needs_more_practice"
            message = f"💪 Keep practicing {state['current_concept']}. We're making progress!"
        
        print(f"\n🎓 TUTORING SESSION COMPLETE")
        print(f"{message}")
        print(f"📊 Final understanding level: {state['understanding_level']}/10")
        print(f"🎯 Mastery progress: {state['mastery_counter']}/5")
        print(f"🔧 Tools used: {len(state['tool_iterations'])} times")
        
        return {
            **state,
            "session_status": "completed",
            "session_outcome": outcome,
            "learning_history": updated_learning_history,
            "completion_message": message
        }
    def exit_session(self, state: ConversationalTutorState) -> ConversationalTutorState:
        """Handle user-requested early exit from tutoring"""
        
        logger.info("🚪 Processing user exit from conversational tutoring")
        
        # Update learning history with partial progress
        updated_learning_history = state.get("learning_history", {})
        updated_learning_history[state["current_concept"]] = {
            "final_understanding": state["understanding_level"],
            "mastery_achieved": False,  # User exited early
            "session_duration": state["current_iteration"],
            "effective_tools": [t["tool_used"] for t in state["tool_iterations"] if t.get("tool_output", {}).get("success", False)],
            "last_session": datetime.now().isoformat(),
            "exit_reason": "user_requested"
        }
        
        # Determine appropriate exit message
        if state["understanding_level"] >= 6:
            outcome = "partial_progress"
            message = f"👍 Good progress on {state['current_concept']}! Feel free to come back anytime."
        else:
            outcome = "early_exit"
            message = f"💪 We made some progress on {state['current_concept']}. Practice makes perfect!"
        
        print(f"\n🚪 TUTORING SESSION EXITED")
        print(f"{message}")
        print(f"📊 Understanding level reached: {state['understanding_level']}/10")
        print(f"🎯 Mastery progress: {state['mastery_counter']}/5")
        print(f"🔧 Tools used: {len(state['tool_iterations'])} times")
        print(f"🔄 Returning to your main session...")
        
        return {
            **state,
            "session_status": "exited",
            "session_outcome": outcome,
            "learning_history": updated_learning_history,
            "completion_message": message
        }

    def _parse_json_decision(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"action": "use_tool", "tool_name": "refine_explanation", "reasoning": "JSON parse failed"}

def create_conversational_tutor_workflow():
    """Create LangGraph workflow for conversational tutoring"""
    
    agent = ConversationalTutorAgent()
    workflow = StateGraph(ConversationalTutorState)
    
    # Add nodes
    workflow.add_node("initialize_session", agent.initialize_session)
    workflow.add_node("assess_and_decide", agent.assess_and_decide)
    workflow.add_node("execute_tool_action", agent.execute_tool_action)
    workflow.add_node("collect_user_response", agent.collect_user_response)
    workflow.add_node("check_mastery_progress", agent.check_mastery_progress)
    workflow.add_node("end_session", agent.end_session)
    workflow.add_node("exit_session", agent.exit_session)
    # Define workflow edges
    workflow.set_entry_point("initialize_session")
    workflow.add_edge("initialize_session", "assess_and_decide")
    workflow.add_edge("assess_and_decide", "execute_tool_action")
    workflow.add_edge("execute_tool_action", "collect_user_response")
    workflow.add_edge("collect_user_response", "check_mastery_progress")
    
    # Conditional routing based on session manager
    workflow.add_conditional_edges(
        "check_mastery_progress",
        agent.session_manager,
        {
            "continue_tutoring": "assess_and_decide",
            "test_final_mastery": "assess_and_decide", 
            "end_session": "end_session",
            "exit_session": "exit_session"
        }
    )
    
    workflow.add_edge("end_session", END)
    workflow.add_edge("exit_session", END)
    return workflow.compile()

def start_conversational_tutoring(user_id: str, topic: str, entry_context: Dict = None) -> Dict[str, Any]:
    """
    Start conversational tutoring session with iterative tools and memory persistence.
    
    Args:
        user_id: Unique identifier for the learner
        topic: The concept/topic to teach
        entry_context: Optional context from calling agent (Quiz/Teaching/Boss)
        
    Returns:
        Dict containing session results, mastery metrics, and learning progress
    """
    
    logger.info(f"🎓 Starting conversational tutoring: {user_id} learning {topic}")
    
    # Initialize session state
    initial_state: ConversationalTutorState = {
        "user_id": user_id,
        "session_id": f"conv_{user_id}_{int(datetime.now().timestamp())}",
        "original_topic": topic,
        "current_concept": topic,
        "entry_agent": entry_context.get("from_agent", "direct") if entry_context else "direct",
        "messages": entry_context.get("messages", []) if entry_context else [],
        "learning_history": {},
        "struggle_patterns": {},
        "effective_methods": [],
        "mastered_concepts": [],
        "tool_iterations": [],
        "understanding_level": 3,
        "mastery_counter": 0,
        "session_status": "starting",
        "max_iterations": 20,
        "current_iteration": 0,
        "current_decision": {},
        "last_user_input": "",
        "last_evaluation": {},
        "exit_requested": False
    }
    
    try:
        # 🔧 FIXED: Simple workflow creation without checkpointing issues
        workflow = create_conversational_tutor_workflow()
        
        # 🔧 FIXED: Execute with higher recursion limit
        result = workflow.invoke(
            initial_state,
            config={"recursion_limit": 100}  # Increased from 30
        )
        
        logger.info(f"✅ Conversational tutoring session completed with outcome: {result.get('session_outcome', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Conversational tutoring session failed: {e}")
        return {
            "session_status": "failed",
            "session_outcome": "error",
            "error_message": str(e),
            "user_id": user_id,
            "topic": topic
        }

# For testing/debugging
if __name__ == "__main__":
    # Test the conversational tutoring system
    result = start_conversational_tutoring(
        user_id="test_user",
        topic="calculus_derivatives",
        entry_context={
            "from_agent": "quiz_agent",
            "messages": [{"role": "user", "content": "I don't understand derivatives at all"}]
        }
    )
    print(f"Session completed: {result}")
