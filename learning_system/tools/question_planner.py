from typing import List, Dict, Any, Optional
from groq import Groq
import json
import logging
import re
from tools.content_loader import ContentCache
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
        combined_text_with_images = ContentCache.get_combined_text_with_images(content_cache, max_chars=2000)
        # Prepare multimodal content for Groq API
        content = []
        
        planning_prompt = f"""
You are an expert question planner. Based on the content below, create exactly {num_questions} questions.

Content Overview:
{combined_text_with_images}

Requirements:
1. Each question should test understanding of key concepts
2. Target difficulty: {difficulty}
3. Questions should be clear and focused
4. Focus on important topics from the content

CRITICAL: Return ONLY a valid JSON array. No explanatory text before or after.

Return exactly this format:
[
    {{
        "question": "What is the main concept discussed?",
        "difficulty": "{difficulty}",
        "expected_answer_type": "concept",
        "topic_area": "general"
    }}
]

Generate exactly {num_questions} questions:
"""
        
        try:
            # CHANGED: Text-only request with larger model
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": planning_prompt}],
                model="llama-3.3-70b-versatile",  # Larger text-only model
                max_tokens=1000,
                temperature=0.7
            )
            
            planning_text = response.choices[0].message.content.strip()
            logger.info(f"🧠 Raw planning response: {planning_text}")
            
            # Extract JSON using robust method
            question_plan = self._extract_json_array(planning_text)  # ← FIXED: Remove "safe_"
            
            if not question_plan:
                raise ValueError("No valid JSON questions found in response")
            
    # ... rest of your code

            
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
    
    def _extract_json_array(self, response_text: str) -> List[Dict[str, Any]]:
        """Safely extract JSON array from LLM response with extra text"""
        logger.info("🔍 Extracting JSON from response...")
        
        # Method 1: Try to find complete JSON array
        try:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                # Clean up potential issues
                json_str = re.sub(r'\n\s*\n', '\n', json_str)  # Remove extra newlines
                json_str = json_str.strip()
                
                parsed_json = json.loads(json_str)
                logger.info(f"✅ Successfully extracted {len(parsed_json)} questions")
                return parsed_json
                
        except json.JSONDecodeError as e:
            logger.warning(f"Method 1 failed: {e}")
            pass
        
        # Method 2: Extract individual JSON objects and combine
        try:
            json_objects = []
            # Find all JSON object patterns
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    obj = json.loads(match.strip())
                    if 'question' in obj:  # Validate it's a question object
                        json_objects.append(obj)
                except json.JSONDecodeError:
                    continue
            
            if json_objects:
                logger.info(f"✅ Extracted {len(json_objects)} questions using method 2")
                return json_objects
                
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
            pass
        
        # Method 3: Try to extract using line-by-line parsing
        try:
            json_objects = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        if 'question' in obj:
                            json_objects.append(obj)
                    except json.JSONDecodeError:
                        continue
            
            if json_objects:
                logger.info(f"✅ Extracted {len(json_objects)} questions using method 3")
                return json_objects
                
        except Exception as e:
            logger.warning(f"Method 3 failed: {e}")
            pass
        
        # Method 4: Fallback - return empty list
        logger.error("❌ All JSON extraction methods failed")
        return []
    
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
