"""
Data models for quiz agent
"""

from dataclasses import dataclass
from typing import TypedDict, Optional

@dataclass
class Topic:
    id: str
    name: str
    keywords: list[str]
    mastery_score: float
    question_count: int
    total_score: int
    last_updated: str
    content_samples: list[str]

@dataclass
class QuestionRecord:
    question_text: str
    user_answer: str
    score: int
    feedback: str

class QuizState(TypedDict):
    user_id: str
    session_id: str
    system_prompt: str
    md_paths: list[str]
    image_paths: list[str]
    extracted_keywords: list[str]
    raw_content: str
    image_descriptions: list[str]
    topic_id: str
    topic_name: str
    current_mastery: float
    question_text: str
    user_answer: Optional[str]
    difficulty_level: str
