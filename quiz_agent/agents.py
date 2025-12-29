"""
LLM agents for question generation and scoring
"""

from groq import Groq
from config import DIFFICULTY_INSTRUCTIONS, SCORING_PROMPT_TEMPLATE

class AdaptiveQuestionGenerator:
    """Generate questions based on mastery"""
    
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, content: str, topic_name: str, mastery_score: float, system_prompt: str) -> str:
        """Generate question"""
        from config import BASIC_THRESHOLD, INTERMEDIATE_THRESHOLD
        
        if mastery_score < BASIC_THRESHOLD:
            difficulty = "basic"
        elif mastery_score < INTERMEDIATE_THRESHOLD:
            difficulty = "intermediate"
        else:
            difficulty = "advanced"
        
        difficulty_instruction = DIFFICULTY_INSTRUCTIONS[difficulty]
        
        user_prompt = f"""Generate ONE question about: "{topic_name}"

Content:
{content[:1500]}

{difficulty_instruction}

Return ONLY the question, no preamble.

Question:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            question = response.choices[0].message.content.strip()
            
            for prefix in ["Based on", "Here's", "Question:", "Based on the"]:
                if question.lower().startswith(prefix.lower()):
                    parts = question.split(":", 1)
                    if len(parts) > 1:
                        question = parts[1].strip()
            
            return question.strip('"').strip("'").strip()
        
        except Exception as e:
            return f"What are the key concepts in {topic_name}?"


class AnswerScorer:
    """Score answers using LLM"""
    
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def score(self, question: str, answer: str, reference: str, system_prompt: str) -> tuple[int, str]:
        
        prompt = SCORING_PROMPT_TEMPLATE.format(
            question=question,
            reference=reference[:800],
            answer=answer
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            score = 3
            feedback = ""
            
            for line in result.split('\n'):
                if line.startswith('SCORE:'):
                    try:
                        score = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('FEEDBACK:'):
                    feedback = line.split(':', 1)[1].strip()
            
            if not feedback:
                feedback = result
            
            return (score, feedback)
        
        except:
            return (3, "Unable to score answer")
