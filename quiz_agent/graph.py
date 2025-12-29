"""
LangGraph construction
"""

from langgraph.graph import StateGraph, END, START
from models import QuizState
from storage import TopicKnowledgeBase, SessionStorage
from extractors import KeywordExtractor, ImageDescriptor
from agents import AdaptiveQuestionGenerator, AnswerScorer
import nodes
from config import (
    GROQ_API_KEY, TOPICS_FILE, SESSIONS_DIR,
    VISION_MODEL, VISION_TEMPERATURE, VISION_MAX_TOKENS, IMAGE_DESCRIPTION_PROMPT,
    QUESTION_MODEL, QUESTION_TEMPERATURE, QUESTION_MAX_TOKENS,
    SCORER_MODEL, SCORER_TEMPERATURE, SCORER_MAX_TOKENS
)


def initialize_components():
    """Initialize all components"""
    nodes.topic_kb = TopicKnowledgeBase(TOPICS_FILE)
    nodes.session_storage = SessionStorage(SESSIONS_DIR)
    nodes.keyword_extractor = KeywordExtractor()
    
    nodes.image_descriptor = ImageDescriptor(
        api_key=GROQ_API_KEY,
        model=VISION_MODEL,
        temperature=VISION_TEMPERATURE,
        max_tokens=VISION_MAX_TOKENS,
        prompt=IMAGE_DESCRIPTION_PROMPT
    )
    
    nodes.question_generator = AdaptiveQuestionGenerator(
        api_key=GROQ_API_KEY,
        model=QUESTION_MODEL,
        temperature=QUESTION_TEMPERATURE,
        max_tokens=QUESTION_MAX_TOKENS
    )
    
    nodes.answer_scorer = AnswerScorer(
        api_key=GROQ_API_KEY,
        model=SCORER_MODEL,
        temperature=SCORER_TEMPERATURE,
        max_tokens=SCORER_MAX_TOKENS
    )


def build_graph() -> StateGraph:
    """Build the LangGraph"""
    initialize_components()
    
    graph = StateGraph(QuizState)
    
    graph.add_node("initialize_session", nodes.initialize_session)
    graph.add_node("extract_content", nodes.extract_content)
    graph.add_node("identify_topic", nodes.identify_topic)
    graph.add_node("generate_question", nodes.generate_question)
    graph.add_node("collect_answer", nodes.collect_answer)
    graph.add_node("score_and_update", nodes.score_and_update)
    
    graph.add_edge(START, "initialize_session")
    graph.add_edge("initialize_session", "extract_content")
    graph.add_edge("extract_content", "identify_topic")
    graph.add_edge("identify_topic", "generate_question")
    graph.add_edge("generate_question", "collect_answer")
    graph.add_edge("collect_answer", "score_and_update")
    graph.add_edge("score_and_update", END)
    
    return graph.compile()
