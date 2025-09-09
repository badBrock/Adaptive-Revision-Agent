from typing import Dict, Any, Optional ,List
from groq import Groq
import re
import logging

logger = logging.getLogger(__name__)

class AnswerScorer:
    """Evaluate answers using LLM-based scoring"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def score_answer(
        self,
        question: str,
        user_answer: str,
        content_cache: Dict[str, Any],
        feedback: Optional[str] = None,
        question_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Score a one-word answer using cached content
        
        Args:
            question: The question asked
            user_answer: User's one-word answer
            content_cache: Pre-loaded content cache
            feedback: Optional feedback context
            question_info: Optional question metadata
            
        Returns:
            Dictionary with score and scoring details
        """
        logger.info(f"🎯 Scoring answer: '{user_answer}' for question: '{question[:50]}...'")
        
        # Get reference content from cache
        reference_text = self._get_reference_text(content_cache, max_chars=800)
        base64_images = self._get_reference_images(content_cache, max_images=2)
        
        # Prepare scoring prompt
        scoring_prompt = f"""
You are an expert evaluator. Score this one-word answer from 0-100 based on correctness and relevance.

Question: {question}
User's Answer: {user_answer}
Expected Answer Type: {question_info.get('expected_answer_type', 'concept') if question_info else 'concept'}
Difficulty Level: {question_info.get('difficulty', 'medium') if question_info else 'medium'}

Reference Content (first 800 chars):
{reference_text}

Scoring Criteria:
- 90-100: Perfect or excellent answer, directly correct
- 80-89: Very good answer, mostly correct with minor issues
- 70-79: Good answer, correct but not ideal
- 60-69: Acceptable answer, partially correct
- 40-59: Poor answer, somewhat related but incorrect
- 20-39: Bad answer, barely related
- 0-19: Wrong or completely unrelated answer

Additional Context:
{f"Feedback: {feedback}" if feedback else "No additional feedback"}

Return ONLY a number from 0-100:
"""
        
        # Prepare content for API call
        content = [{"type": "text", "text": scoring_prompt}]
        
        # Add images for context
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
                max_tokens=50,
                temperature=0
            )
            
            score_text = response.choices[0].message.content.strip()
            score = self._extract_score(score_text)
            
            # Additional validation based on answer characteristics
            validated_score = self._validate_score(user_answer, score, question_info)
            
            scoring_result = {
                "score": validated_score,
                "raw_score": score,
                "score_explanation": self._generate_score_explanation(validated_score),
                "llm_response": score_text,
                "validation_applied": validated_score != score
            }
            
            logger.info(f"🎯 Score assigned: {validated_score}/100")
            return scoring_result
            
        except Exception as e:
            logger.error(f"❌ LLM scoring failed: {str(e)}")
            return self._fallback_scoring(user_answer, question, question_info)
    
    def _get_reference_text(self, cache: Dict[str, Any], max_chars: int) -> str:
        """Extract reference text from content cache"""
        combined = "\n".join([doc["markdown"] for doc in cache["documents"]])
        return combined[:max_chars] if len(combined) > max_chars else combined
    
    def _get_reference_images(self, cache: Dict[str, Any], max_images: int) -> List[str]:
        """Extract reference images from content cache"""
        return [img["base64"] for img in cache["images"][:max_images]]
    
    def _extract_score(self, score_text: str) -> float:
        """Extract numeric score from LLM response"""
        # Find numbers in the response
        numbers = re.findall(r'\d+(?:\.\d+)?', score_text)
        
        if numbers:
            score = float(numbers[0])
            # Ensure score is in valid range
            return max(0, min(100, score))
        else:
            logger.warning(f"No numeric score found in: {score_text}")
            return 50.0  # Default middle score
    
    def _validate_score(self, user_answer: str, score: float, question_info: Optional[Dict[str, Any]]) -> float:
        """Apply validation rules to score"""
        # Basic validation based on answer characteristics
        if len(user_answer.strip()) == 0:
            return 0.0
        
        if len(user_answer.strip()) == 1:
            # Single character answers are usually poor
            return min(score, 30.0)
        
        if not user_answer.isalpha():
            # Non-alphabetic answers (numbers, symbols) may be less ideal
            return min(score, 70.0)
        
        if len(user_answer) > 20:
            # Very long "one-word" answers
            return min(score, 50.0)
        
        return score
    
    def _generate_score_explanation(self, score: float) -> str:
        """Generate human-readable score explanation"""
        if score >= 90:
            return "Excellent answer"
        elif score >= 80:
            return "Very good answer"
        elif score >= 70:
            return "Good answer"
        elif score >= 60:
            return "Acceptable answer"
        elif score >= 40:
            return "Poor answer"
        elif score >= 20:
            return "Bad answer"
        else:
            return "Wrong answer"
    
    def _fallback_scoring(self, user_answer: str, question: str, question_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback scoring when LLM fails"""
        logger.warning("Using fallback scoring due to LLM failure")
        
        # Simple heuristic-based scoring
        if len(user_answer.strip()) == 0:
            fallback_score = 0.0
        elif len(user_answer) == 1:
            fallback_score = 20.0
        elif not user_answer.isalpha():
            fallback_score = 40.0
        elif len(user_answer) <= 15:
            fallback_score = 60.0
        else:
            fallback_score = 30.0
        
        return {
            "score": fallback_score,
            "raw_score": fallback_score,
            "score_explanation": f"Fallback scoring: {self._generate_score_explanation(fallback_score)}",
            "llm_response": "Fallback scoring applied",
            "validation_applied": False,
            "fallback": True
        }

    def calculate_session_grade(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate final session grade from individual scores"""
        if not scores:
            return {
                "average_score": 0.0,
                "grade": "🔴 No Answers",
                "recommendation": "No questions were answered.",
                "performance_level": "none"
            }
        
        average_score = sum(scores) / len(scores)
        
        # Determine grade and recommendations
        if average_score >= 80:
            grade = "🟢 Pass"
            recommendation = "Excellent! You have strong understanding. Move to the next topic."
            performance_level = "excellent"
        elif average_score >= 60:
            grade = "🟡 Needs Revision"
            recommendation = "Partial understanding. Review the material and try more questions."
            performance_level = "partial"
        else:
            grade = "🔴 Retry Flashcards"
            recommendation = "Poor understanding. Restart with focused study of the material."
            performance_level = "poor"
        
        return {
            "average_score": average_score,
            "grade": grade,
            "recommendation": recommendation,
            "performance_level": performance_level,
            "total_questions": len(scores),
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 80]),
                "good": len([s for s in scores if 60 <= s < 80]),
                "poor": len([s for s in scores if s < 60])
            }
        }
