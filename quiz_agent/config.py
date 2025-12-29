"""
Configuration - All prompts, models, and parameters
"""

from pathlib import Path
import os
from dotenv import load_dotenv
# ============================================================================
# API & STORAGE
# ============================================================================
load_dotenv()  # loads .env into environment variables [web:60]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Put it in .env or set it in the environment.")
TOPICS_FILE = Path("topics_knowledge.json")
SESSIONS_DIR = Path("quiz_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODELS & PARAMETERS
# ============================================================================

# Image Description
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_TEMPERATURE = 0.3
VISION_MAX_TOKENS = 1024

# Question Generation
QUESTION_MODEL = "llama-3.3-70b-versatile"
QUESTION_TEMPERATURE = 0.75
QUESTION_MAX_TOKENS = 200

# Answer Scoring
SCORER_MODEL = "llama-3.3-70b-versatile"
SCORER_TEMPERATURE = 0.3
SCORER_MAX_TOKENS = 300

# ============================================================================
# THRESHOLDS
# ============================================================================

BASIC_THRESHOLD = 2.0
INTERMEDIATE_THRESHOLD = 3.5
KEYWORD_TOP_N = 15
KEYWORD_MIN_LENGTH = 3
TOPIC_NAME_MAX_WORDS = 4

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """Expert educational AI. Generate clear questions and constructive feedback."""

IMAGE_DESCRIPTION_PROMPT = "Describe this educational image focusing on key concepts."

DIFFICULTY_INSTRUCTIONS = {
    "basic": "Ask a BASIC question about fundamental concepts.",
    "intermediate": "Ask an INTERMEDIATE question requiring understanding of connections.",
    "advanced": "Ask an ADVANCED question requiring deep analysis."
}

SCORING_PROMPT_TEMPLATE = """Score this answer (1-5).

Question: {question}
Reference: {reference}
Answer: {answer}

Format:
SCORE: [1-5]
FEEDBACK: [brief feedback]"""

# ============================================================================
# FILE EXTENSIONS
# ============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}
MARKDOWN_EXTENSIONS = {'.md', '.markdown'}
