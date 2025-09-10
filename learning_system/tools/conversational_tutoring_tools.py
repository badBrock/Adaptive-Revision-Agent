# tools/conversational_tutoring_tools.py
from typing import Dict, Any, List
from groq import Groq
import os
import json
import logging

logger = logging.getLogger(__name__)

class ConversationalTutoringTools:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def refine_explanation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refines and simplifies explanations based on learner feedback until clarity is achieved.
        LLM decides how to refine based on confusion signals and conversation history.
        """
        current_explanation = input_data.get("current_explanation", "")
        learner_feedback = input_data.get("learner_feedback", "")
        conversation_history = input_data.get("conversation_history", [])
        topic = input_data.get("topic", "")
        
        # Let LLM decide how to refine based on context
        refine_prompt = f"""
        You are a patient tutor helping a confused student understand {topic}.
        
        Current explanation: {current_explanation}
        Student's feedback: {learner_feedback}
        Conversation so far: {json.dumps(conversation_history[-3:], indent=2)}
        
        The student is confused. Refine your explanation by:
        - Using simpler language if needed
        - Adding concrete examples or analogies
        - Breaking down complex parts
        - Addressing their specific confusion
        
        Provide a clearer, more helpful explanation:
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": refine_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=300,
                temperature=0.4
            )
            
            refined_explanation = response.choices[0].message.content.strip()
            
            return {
                "refined_explanation": refined_explanation,
                "iteration_type": "explanation_refinement",
                "topic": topic,
                "timestamp": input_data.get("timestamp"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Refine explanation failed: {e}")
            return {
                "refined_explanation": "I understand you're confused. Let me try a different approach to explain this concept.",
                "iteration_type": "fallback_explanation", 
                "success": False,
                "error": str(e)
            }

    def micro_lesson_generator(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates bite-sized lessons on prerequisite concepts or misunderstood topics.
        LLM determines what prerequisite knowledge is missing and creates targeted lessons.
        """
        missing_concept = input_data.get("missing_concept", "")
        learner_level = input_data.get("learner_level", "beginner")
        context = input_data.get("context", "")
        learning_history = input_data.get("learning_history", {})
        
        lesson_prompt = f"""
        A student is struggling with a concept because they're missing foundational knowledge.
        
        Missing concept: {missing_concept}
        Student level: {learner_level}
        Context: {context}
        What they've learned before: {json.dumps(learning_history, indent=2)}
        
        Create a focused micro-lesson that:
        1. Explains the missing concept simply
        2. Uses examples relevant to their level
        3. Connects to what they already know
        4. Builds confidence step by step
        
        Keep it concise but thorough. End with a simple check question.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": lesson_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=400,
                temperature=0.3
            )
            
            micro_lesson = response.choices[0].message.content.strip()
            
            return {
                "micro_lesson": micro_lesson,
                "concept_taught": missing_concept,
                "iteration_type": "prerequisite_teaching",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Micro lesson generation failed: {e}")
            return {
                "micro_lesson": f"Let's start with the basics of {missing_concept}. This is a fundamental concept that will help you understand the bigger picture.",
                "concept_taught": missing_concept,
                "iteration_type": "fallback_lesson",
                "success": False,
                "error": str(e)
            }

    def generate_leading_question(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates Socratic, guiding questions to help learners discover answers themselves.
        LLM crafts questions that guide without giving away the answer.
        """
        target_insight = input_data.get("target_insight", "")
        current_understanding = input_data.get("current_understanding", "")
        topic = input_data.get("topic", "")
        previous_questions = input_data.get("previous_questions", [])
        
        socratic_prompt = f"""
        You are using the Socratic method to guide a student to discover: {target_insight}
        
        Topic: {topic}
        Student's current understanding: {current_understanding}
        Previous questions asked: {json.dumps(previous_questions, indent=2)}
        
        Generate a leading question that:
        - Guides them toward the insight without giving it away
        - Builds on their current understanding
        - Is different from previous questions
        - Encourages them to think and reason
        - Helps them make the connection themselves
        
        Ask just one strategic question:
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": socratic_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=150,
                temperature=0.5
            )
            
            leading_question = response.choices[0].message.content.strip()
            
            return {
                "leading_question": leading_question,
                "target_insight": target_insight,
                "iteration_type": "socratic_questioning",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Leading question generation failed: {e}")
            return {
                "leading_question": f"Can you think of a way {topic} might relate to something you already understand?",
                "target_insight": target_insight,
                "iteration_type": "fallback_question",
                "success": False,
                "error": str(e)
            }

# Tool instances with descriptions for LLM
tutoring_tools = ConversationalTutoringTools()

CHATBOT_TOOLS = [
    {
        "name": "refine_explanation", 
        "description": "Iteratively refines explanations based on learner confusion and feedback until understanding is achieved",
        "func": tutoring_tools.refine_explanation
    },
    {
        "name": "micro_lesson_generator",
        "description": "Creates targeted bite-sized lessons on prerequisite concepts the learner is missing", 
        "func": tutoring_tools.micro_lesson_generator
    },
    {
        "name": "generate_leading_question",
        "description": "Generates Socratic questions to guide learner discovery without giving direct answers",
        "func": tutoring_tools.generate_leading_question
    }
]
