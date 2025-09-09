from typing import List, Dict, Any, Optional
from groq import Groq
import json
import logging

logger = logging.getLogger(__name__)

class QuestionPlanner:
    """Generate all questions upfront based on content"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def plan_questions(
        self,
        content_cache: Dict[str, Any],
        difficulty: str = "medium",
        num_questions: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Plan ALL questions upfront using cached content
        
        Args:
            content_cache: Pre-loaded content from ContentCache
            difficulty: Target difficulty level
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"🧠 Planning {num_questions} questions at {difficulty} difficulty")
        
        # Get content from cache
        combined_text = self._get_text_from_cache(content_cache, max_chars=2000)
        base64_images = self._get_images_from_cache(content_cache, max_images=3)
        
        # Prepare multimodal content for Groq API
        content = []
        
        planning_prompt = f"""
You are an expert question planner. Based on the content below, create exactly {num_questions} questions that require ONE WORD answers.

Content Overview:
{combined_text}

Requirements:
1. Each question must be answerable with exactly ONE WORD
2. Target difficulty: {difficulty}
3. Questions should test key concepts, definitions, and main ideas
4. Focus on concrete nouns, verbs, adjectives that are central to the topic
5. Avoid questions requiring explanations or multiple concepts

Examples of good one-word answer questions:
- "What algorithm is primarily discussed?" → "backpropagation"
- "What language is used for implementation?" → "Python"
- "What type of network is described?" → "neural"

Return a JSON array with this exact format:
[
    {{
        "question": "What is the main algorithm discussed?",
        "difficulty": "medium",
        "expected_answer_type": "algorithm name",
        "topic_area": "machine learning"
    }}
]

Generate exactly {num_questions} one-word answer questions:
"""
        
        content.append({"type": "text", "text": planning_prompt})
        
        # Add images for visual context
        for base64_img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_img}
            })
        
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": content
                }],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=1000,
                temperature=0.7
            )
            
            planning_text = response.choices[0].message.content.strip()
            logger.info(f"🧠 Raw planning response: {planning_text}")
            
            # Extract JSON
            question_plan = self._extract_json_from_response(planning_text)
            
            # Validate and ensure we have the right number of questions
            if len(question_plan) != num_questions:
                logger.warning(f"Expected {num_questions} questions, got {len(question_plan)}")
            
            # Add metadata to each question
            for i, q in enumerate(question_plan):
                q["question_id"] = i + 1
                q["planned_at"] = None
            
            logger.info(f"📋 Successfully planned {len(question_plan)} questions")
            return question_plan
            
        except Exception as e:
            logger.error(f"❌ Question planning failed: {str(e)}")
            return self._create_fallback_questions(content_cache, num_questions)
    
    def _get_text_from_cache(self, cache: Dict[str, Any], max_chars: int) -> str:
        """Extract text content from cache"""
        combined = "\n\n".join([
            f"Topic: {doc['topic_name']}\n{doc['markdown']}" 
            for doc in cache["documents"]
        ])
        return combined[:max_chars] if len(combined) > max_chars else combined
    
    def _get_images_from_cache(self, cache: Dict[str, Any], max_images: int) -> List[str]:
        """Extract base64 images from cache"""
        return [img["base64"] for img in cache["images"][:max_images]]
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response"""
        try:
            # Find JSON array in response
            if '[' in response_text and ']' in response_text:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                json_str = response_text[json_start:json_end]
                question_plan = json.loads(json_str)
                
                # Validate structure
                if isinstance(question_plan, list) and len(question_plan) > 0:
                    for q in question_plan:
                        if not isinstance(q, dict) or "question" not in q:
                            raise ValueError("Invalid question structure")
                    return question_plan
                else:
                    raise ValueError("Empty or invalid question list")
            else:
                raise ValueError("No JSON array found in response")
                
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            raise
    
    def _create_fallback_questions(self, cache: Dict[str, Any], num_questions: int) -> List[Dict[str, Any]]:
        """Create fallback questions when LLM planning fails"""
        logger.warning("Creating fallback questions due to planning failure")
        
        # Extract key terms from content
        all_text = " ".join([doc["markdown"] for doc in cache["documents"]])
        words = all_text.split()
        
        # Find potential answer words (longer, alphabetic)
        key_terms = []
        for word in words:
            cleaned = word.strip('.,!?()[]{}":;')
            if len(cleaned) > 4 and cleaned.isalpha() and cleaned.islower():
                key_terms.append(cleaned)
        
        # Remove duplicates and take first N
        unique_terms = list(dict.fromkeys(key_terms))[:num_questions]
        
        fallback_questions = []
        for i, term in enumerate(unique_terms):
            fallback_questions.append({
                "question": f"What concept does '{term}' relate to in this topic?",
                "difficulty": "medium",
                "expected_answer_type": "concept",
                "topic_area": "general",
                "question_id": i + 1,
                "planned_at": None,
                "fallback": True
            })
        
        # Fill remaining slots if needed
        while len(fallback_questions) < num_questions:
            fallback_questions.append({
                "question": f"What is a key term from this content? (Question {len(fallback_questions) + 1})",
                "difficulty": "easy",
                "expected_answer_type": "term",
                "topic_area": "general",
                "question_id": len(fallback_questions) + 1,
                "planned_at": None,
                "fallback": True
            })
        
        return fallback_questions[:num_questions]

    def adaptive_difficulty_adjust(self, questions: List[Dict[str, Any]], performance_score: float) -> List[Dict[str, Any]]:
        """Adjust remaining questions based on user performance"""
        if performance_score >= 80:
            # Make remaining questions harder
            for q in questions:
                if q.get("difficulty") == "easy":
                    q["difficulty"] = "medium"
                elif q.get("difficulty") == "medium":
                    q["difficulty"] = "hard"
        elif performance_score <= 40:
            # Make remaining questions easier
            for q in questions:
                if q.get("difficulty") == "hard":
                    q["difficulty"] = "medium"
                elif q.get("difficulty") == "medium":
                    q["difficulty"] = "easy"
        
        return questions
