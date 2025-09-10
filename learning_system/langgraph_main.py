#!/usr/bin/env python3
"""
LangGraph Studio Entry Point for Adaptive Tutoring System
Exposes your multi-agent workflow for visualization and debugging
"""

import os
from typing import Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load your existing modules
from integration import run_integrated_learning_system
from boss_agent import run_boss_agent_session
from quiz_agent import run_quiz_agent_session  
from teaching_agent import run_teaching_agent_session

load_dotenv()

# Define LangGraph State for Studio visualization
class TutoringState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    content_folder: str
    current_step: str
    boss_decision: Dict[str, Any]
    agent_result: Dict[str, Any]
    session_summary: Dict[str, Any]
    
def create_tutoring_workflow():
    """Create LangGraph workflow for Studio visualization"""
    
    def initialize_session(state: TutoringState) -> TutoringState:
        """Initialize tutoring session"""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Extract user request from message
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Parse user_id and content_folder from input
            # Format: "user:brock folder:./data/..."
            parts = user_input.split()
            user_id = "student_001"  # default
            content_folder = "./content"  # default
            
            for part in parts:
                if part.startswith("user:"):
                    user_id = part.split(":")[1]
                elif part.startswith("folder:"):
                    content_folder = part.split(":")[1]
        
        return {
            **state,
            "user_id": user_id,
            "content_folder": content_folder,
            "current_step": "initialized",
            "messages": messages + [AIMessage(content=f"Initializing session for {user_id}")]
        }
    
    def boss_agent_node(state: TutoringState) -> TutoringState:
        """Boss Agent routing decision"""
        try:
            # Run your existing boss agent
            boss_result = run_boss_agent_session(
                state["user_id"], 
                state["content_folder"], 
                quiz_results=None
            )
            
            decision = boss_result.get('decision', {})
            recommended_agent = decision.get('recommended_agent', 'quiz_agent')
            recommended_topic = decision.get('recommended_topic', 'unknown')
            
            message = f"🧠 Boss Agent Decision: Study '{recommended_topic}' using {recommended_agent}"
            
            return {
                **state,
                "boss_decision": decision,
                "current_step": "boss_completed",
                "messages": state["messages"] + [AIMessage(content=message)]
            }
            
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [AIMessage(content=f"Boss Agent failed: {e}")]
            }
    
    def quiz_agent_node(state: TutoringState) -> TutoringState:
        """Quiz Agent execution"""
        try:
            decision = state["boss_decision"]
            recommended_topic = decision.get("recommended_topic", "default")
            
            # Run quiz agent
            agent_result = run_quiz_agent_session(state["user_id"], recommended_topic)
            
            score = agent_result.get('average_score', 0)
            message = f"🎯 Quiz Agent completed: {score:.1f}% score, {agent_result.get('grade', 'N/A')} grade"
            
            return {
                **state,
                "agent_result": agent_result,
                "current_step": "quiz_completed", 
                "messages": state["messages"] + [AIMessage(content=message)]
            }
            
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "messages": state["messages"] + [AIMessage(content=f"Quiz Agent failed: {e}")]
            }
    
    def teaching_agent_node(state: TutoringState) -> TutoringState:
        """Teaching Agent execution"""
        try:
            decision = state["boss_decision"]
            recommended_topic = decision.get("recommended_topic", "default")
            
            # Run teaching agent
            agent_result = run_teaching_agent_session(state["user_id"], recommended_topic)
            
            score = agent_result.get('average_score', 0)
            message = f"🎓 Teaching Agent completed: {score:.1f}% score, {agent_result.get('grade', 'N/A')} grade"
            
            return {
                **state,
                "agent_result": agent_result,
                "current_step": "teaching_completed",
                "messages": state["messages"] + [AIMessage(content=message)]
            }
            
        except Exception as e:
            return {
                **state,
                "current_step": "error", 
                "messages": state["messages"] + [AIMessage(content=f"Teaching Agent failed: {e}")]
            }
    
    def finalize_session(state: TutoringState) -> TutoringState:
        """Finalize learning session"""
        agent_result = state.get("agent_result", {})
        
        summary = {
            "user_id": state["user_id"],
            "topic_studied": state["boss_decision"].get("recommended_topic", "unknown"),
            "agent_used": state["boss_decision"].get("recommended_agent", "unknown"),
            "final_score": agent_result.get("average_score", 0),
            "grade": agent_result.get("grade", "N/A")
        }
        
        message = f"✅ Session Complete! Topic: {summary['topic_studied']}, Score: {summary['final_score']:.1f}%, Grade: {summary['grade']}"
        
        return {
            **state,
            "session_summary": summary,
            "current_step": "completed",
            "messages": state["messages"] + [AIMessage(content=message)]
        }
    
    def route_to_agent(state: TutoringState) -> str:
        """Route to appropriate learning agent"""
        decision = state.get("boss_decision", {})
        recommended_agent = decision.get("recommended_agent", "quiz_agent")
        
        if recommended_agent == "teaching_agent":
            return "teaching_agent"
        else:
            return "quiz_agent"
    
    # Build the workflow graph
    workflow = StateGraph(TutoringState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_session)
    workflow.add_node("boss_agent", boss_agent_node)
    workflow.add_node("quiz_agent", quiz_agent_node)
    workflow.add_node("teaching_agent", teaching_agent_node)
    workflow.add_node("finalize", finalize_session)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "boss_agent")
    
    # Conditional routing based on boss decision
    workflow.add_conditional_edges(
        "boss_agent",
        route_to_agent,
        {
            "quiz_agent": "quiz_agent",
            "teaching_agent": "teaching_agent"
        }
    )
    
    workflow.add_edge("quiz_agent", "finalize")
    workflow.add_edge("teaching_agent", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Create the compiled graph for LangGraph Studio
graph = create_tutoring_workflow()

# Export for LangGraph Studio
if __name__ == "__main__":
    # Test the workflow
    initial_state = {
        "messages": [HumanMessage(content="user:brock folder:./data/rag_8_Weight_Initilization_20250906_032237")],
        "user_id": "",
        "content_folder": "",
        "current_step": "start",
        "boss_decision": {},
        "agent_result": {},
        "session_summary": {}
    }
    
    result = graph.invoke(initial_state)
    print("Workflow completed:", result["current_step"])
