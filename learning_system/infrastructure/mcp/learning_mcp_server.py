import asyncio
import os
from fastmcp import FastMCP
from typing import Dict, Any
import sys
import os
import logging
import sys

# Configure logging to stderr at the top of your server file
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}", file=sys.stderr) # Debug line
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))
print(f"GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

# Import your existing tools
from tools.question_planner import QuestionPlanner
from tools.scorer import AnswerScorer
from tools.content_loader import ContentCache
from tools.meta_learning import MetaLearningModule
from tools.agent_integration import trigger_conversational_tutoring

# Create FastMCP server
mcp = FastMCP("Learning-System-Server")

# Initialize your existing tool instances (lazy loading)
_question_planner = None
_answer_scorer = None
_meta_learning = None

def get_question_planner():
    global _question_planner
    if _question_planner is None:
        _question_planner = QuestionPlanner(api_key=os.getenv("GROQ_API_KEY"))
    return _question_planner

def get_answer_scorer():
    global _answer_scorer
    if _answer_scorer is None:
        _answer_scorer = AnswerScorer(api_key=os.getenv("GROQ_API_KEY"))
    return _answer_scorer

def get_meta_learning():
    global _meta_learning
    if _meta_learning is None:
        _meta_learning = MetaLearningModule()
    return _meta_learning

@mcp.tool()
def plan_questions(content_cache: Dict[str, Any], difficulty: str = "medium", num_questions: int = 3) -> Dict[str, Any]:
    """Generate questions from content using question planner"""
    try:
        planner = get_question_planner()
        questions = planner.plan_questions(
            content_cache=content_cache,
            difficulty=difficulty,
            num_questions=num_questions
        )
        return {"questions": questions, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def score_answer(question: str, user_answer: str, content_cache: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
    """Score a user's answer"""
    try:
        scorer = get_answer_scorer()
        result = scorer.score_answer(
            question=question,
            user_answer=user_answer,
            content_cache=content_cache,
            feedback=feedback
        )
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed", "score": 50.0}

@mcp.tool()
def calculate_session_grade(scores: list) -> Dict[str, Any]:
    """Calculate final session grade from scores"""
    try:
        scorer = get_answer_scorer()
        result = scorer.calculate_session_grade(scores)
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def load_content(folder_path: str) -> Dict[str, Any]:
    """Load content from folder using ContentCache"""
    try:
        content_cache = ContentCache.load_content_cache(folder_path)
        return {"content_cache": content_cache, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def get_learning_recommendations(user_id: str, topic: str) -> Dict[str, Any]:
    """Get meta-learning recommendations"""
    try:
        meta = get_meta_learning()
        recommendations = meta.get_learning_recommendations(user_id, topic)
        return recommendations
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def get_agent_adaptations(user_id: str, topic: str, agent_type: str) -> Dict[str, Any]:
    """Get agent-specific adaptations"""
    try:
        meta = get_meta_learning()
        adaptations = meta.get_agent_adaptations(user_id, topic, agent_type)
        return adaptations
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def analyze_confusion(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze if user is confused"""
    try:
        is_confused = trigger_conversational_tutoring(text, context or {})
        return {"is_confused": is_confused, "confidence": 0.8, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@mcp.tool()
def start_conversational_tutoring(user_id: str, topic: str, trigger_context: str) -> Dict[str, Any]:
    """Start conversational tutoring session"""
    try:
        from conversational_tutor_agent import start_conversational_tutoring
        result = start_conversational_tutoring(user_id, topic, trigger_context)
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    print(" Starting Learning System FastMCP Server...",file=sys.stderr)
    mcp.run(transport="stdio")
