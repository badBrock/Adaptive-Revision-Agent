# import requests
# import json
# import os
# from typing import List, Dict, Optional
# from pathlib import Path
# import time
# import random

# class GroqLLMClient:
#     """Client for Groq API integration"""
    
#     def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = "https://api.groq.com/openai/v1/chat/completions"
#         self.headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
    
#     def generate_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
#         """Generate completion using Groq API"""
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": "You are a helpful AI tutor who creates engaging quiz questions to test knowledge and understanding."},
#                 {"role": "user", "content": prompt}
#             ],
#             "max_tokens": max_tokens,
#             "temperature": temperature
#         }
        
#         try:
#             response = requests.post(self.base_url, json=payload, headers=self.headers)
#             response.raise_for_status()
            
#             result = response.json()
#             return result["choices"][0]["message"]["content"].strip()
            
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Error calling Groq API: {e}")
#             return None
#         except KeyError as e:
#             print(f"❌ Unexpected API response format: {e}")
#             return None

# class QuizGenerator:
#     """Generates different types of quiz questions from documents"""
    
#     def __init__(self, llm_client: GroqLLMClient):
#         self.llm = llm_client
#         self.question_types = [
#             "comprehension",
#             "application", 
#             "analysis",
#             "recall",
#             "synthesis"
#         ]
    
#     def generate_quiz_from_docs(self, docs: List[Dict], num_questions: int = 5, quiz_type: str = "mixed") -> List[Dict]:
#         """Generate quiz questions from search result documents"""
        
#         if not docs:
#             print("❌ No documents provided for quiz generation")
#             return []
        
#         # Combine document content
#         combined_content = self._prepare_content(docs)
        
#         if not combined_content:
#             print("❌ No valid content found in documents")
#             return []
        
#         print(f"🧠 Generating {num_questions} quiz questions from {len(docs)} documents...")
        
#         # Generate questions based on type
#         if quiz_type == "mixed":
#             questions = self._generate_mixed_questions(combined_content, num_questions)
#         else:
#             questions = self._generate_specific_type_questions(combined_content, num_questions, quiz_type)
        
#         return questions
    
#     def _prepare_content(self, docs: List[Dict]) -> str:
#         """Prepare document content for quiz generation"""
#         content_parts = []
        
#         for i, doc in enumerate(docs[:5], 1):  # Limit to top 5 docs to avoid token limits
#             context = doc.get('context', '')
#             file_name = doc.get('file_name', f'Document {i}')
            
#             if context.strip():
#                 content_parts.append(f"=== {file_name} ===\n{context}\n")
        
#         return '\n'.join(content_parts)
    
#     def _generate_mixed_questions(self, content: str, num_questions: int) -> List[Dict]:
#         """Generate mixed types of questions"""
        
#         prompt = f"""
# Based on the following content from personal notes, generate {num_questions} diverse quiz questions to test understanding and knowledge retention.

# Content:
# {content[:3000]}  # Limit content length

# Generate questions of different types:
# 1. Factual recall questions
# 2. Concept understanding questions  
# 3. Application questions
# 4. Analysis questions
# 5. Connection/synthesis questions

# Format each question as:
# TYPE: [question_type]
# Q: [question]
# HINT: [optional hint]
# ---

# Make questions challenging but fair, focusing on key concepts and practical applications.
# """
        
#         response = self.llm.generate_completion(prompt, max_tokens=1500, temperature=0.8)
        
#         if not response:
#             return self._generate_fallback_questions(content, num_questions)
        
#         return self._parse_questions(response)
    
#     def _generate_specific_type_questions(self, content: str, num_questions: int, question_type: str) -> List[Dict]:
#         """Generate questions of a specific type"""
        
#         type_prompts = {
#             "comprehension": "Create questions that test basic understanding and comprehension of the main concepts.",
#             "application": "Create questions that ask how to apply the concepts in practical scenarios.",
#             "analysis": "Create questions that require breaking down and analyzing the information.",
#             "recall": "Create questions that test memory and recall of specific facts and details.",
#             "synthesis": "Create questions that require combining multiple concepts or ideas."
#         }
        
#         prompt = f"""
# Based on the following content, generate {num_questions} quiz questions focused on {question_type}.

# {type_prompts.get(question_type, '')}

# Content:
# {content[:3000]}

# Format each question as:
# TYPE: {question_type}
# Q: [question]
# HINT: [optional hint]
# ---
# """
        
#         response = self.llm.generate_completion(prompt, max_tokens=1200, temperature=0.7)
        
#         if not response:
#             return self._generate_fallback_questions(content, num_questions)
        
#         return self._parse_questions(response)
    
#     def _parse_questions(self, response: str) -> List[Dict]:
#         """Parse LLM response into structured questions"""
#         questions = []
#         question_blocks = response.split('---')
        
#         for block in question_blocks:
#             block = block.strip()
#             if not block:
#                 continue
            
#             lines = [line.strip() for line in block.split('\n') if line.strip()]
            
#             question_data = {
#                 'type': 'general',
#                 'question': '',
#                 'hint': None
#             }
            
#             for line in lines:
#                 if line.startswith('TYPE:'):
#                     question_data['type'] = line.replace('TYPE:', '').strip()
#                 elif line.startswith('Q:'):
#                     question_data['question'] = line.replace('Q:', '').strip()
#                 elif line.startswith('HINT:'):
#                     question_data['hint'] = line.replace('HINT:', '').strip()
            
#             if question_data['question']:
#                 questions.append(question_data)
        
#         return questions
    
#     def _generate_fallback_questions(self, content: str, num_questions: int) -> List[Dict]:
#         """Generate fallback questions if LLM fails"""
        
#         fallback_questions = [
#             {
#                 'type': 'comprehension',
#                 'question': 'What are the main concepts discussed in these notes?',
#                 'hint': 'Look for recurring themes and key topics'
#             },
#             {
#                 'type': 'recall',
#                 'question': 'List three important points mentioned in the content.',
#                 'hint': 'Focus on facts, definitions, or specific details'
#             },
#             {
#                 'type': 'application',
#                 'question': 'How could you apply this knowledge in a real-world scenario?',
#                 'hint': 'Think about practical uses and implementations'
#             },
#             {
#                 'type': 'analysis',
#                 'question': 'What questions do you still have after reviewing this content?',
#                 'hint': 'Consider what aspects need further clarification'
#             },
#             {
#                 'type': 'synthesis',
#                 'question': 'Summarize the key insights from this content in your own words.',
#                 'hint': 'Focus on the most important takeaways'
#             }
#         ]
        
#         return random.sample(fallback_questions, min(num_questions, len(fallback_questions)))

# class InteractiveQuiz:
#     """Interactive quiz interface"""
    
#     def __init__(self):
#         self.score = 0
#         self.total_questions = 0
#         self.user_answers = []
    
#     def run_quiz(self, questions: List[Dict], time_limit: Optional[int] = None):
#         """Run an interactive quiz session"""
        
#         if not questions:
#             print("❌ No questions available for quiz")
#             return
        
#         print(f"\n🎯 **KNOWLEDGE TEST STARTING**")
#         print(f"📚 Questions: {len(questions)}")
#         if time_limit:
#             print(f"⏰ Time limit: {time_limit} seconds per question")
#         print("=" * 60)
        
#         self.total_questions = len(questions)
#         self.score = 0
#         self.user_answers = []
        
#         for i, question_data in enumerate(questions, 1):
#             self._ask_question(i, question_data, time_limit)
        
#         self._show_quiz_results()
    
#     def _ask_question(self, question_num: int, question_data: Dict, time_limit: Optional[int]):
#         """Ask a single question"""
        
#         print(f"\n📝 **Question {question_num}/{self.total_questions}**")
#         print(f"🏷️  Type: {question_data['type'].title()}")
#         print(f"❓ {question_data['question']}")
        
#         if question_data.get('hint'):
#             show_hint = input("💡 Show hint? (y/n): ").strip().lower()
#             if show_hint.startswith('y'):
#                 print(f"💡 Hint: {question_data['hint']}")
        
#         print("\n" + "─" * 40)
        
#         start_time = time.time()
        
#         try:
#             if time_limit:
#                 print(f"⏰ You have {time_limit} seconds to answer...")
            
#             user_answer = input("✍️  Your answer: ").strip()
            
#             elapsed_time = time.time() - start_time
            
#             if time_limit and elapsed_time > time_limit:
#                 print(f"⏰ Time's up! ({elapsed_time:.1f}s)")
#                 user_answer = "[TIMEOUT]"
            
#         except KeyboardInterrupt:
#             print("\n🛑 Quiz interrupted by user")
#             return
        
#         # Store answer
#         self.user_answers.append({
#             'question': question_data['question'],
#             'type': question_data['type'],
#             'answer': user_answer,
#             'time_taken': elapsed_time
#         })
        
#         # Simple evaluation (could be enhanced with LLM scoring)
#         if user_answer and user_answer != "[TIMEOUT]":
#             print("✅ Answer recorded!")
#             self.score += 1  # Basic scoring - all attempts get points
#         else:
#             print("❌ No valid answer provided")
        
#         print("─" * 40)
    
#     def _show_quiz_results(self):
#         """Display quiz results"""
        
#         print(f"\n🎉 **QUIZ COMPLETED!**")
#         print("=" * 50)
#         print(f"📊 Score: {self.score}/{self.total_questions}")
#         print(f"📈 Participation Rate: {(self.score/self.total_questions)*100:.1f}%")
        
#         if self.user_answers:
#             avg_time = sum(answer['time_taken'] for answer in self.user_answers) / len(self.user_answers)
#             print(f"⏱️  Average Response Time: {avg_time:.1f}s")
        
#         # Show summary by question type
#         type_summary = {}
#         for answer in self.user_answers:
#             q_type = answer['type']
#             if q_type not in type_summary:
#                 type_summary[q_type] = {'total': 0, 'answered': 0}
#             type_summary[q_type]['total'] += 1
#             if answer['answer'] and answer['answer'] != "[TIMEOUT]":
#                 type_summary[q_type]['answered'] += 1
        
#         if type_summary:
#             print(f"\n📋 **Performance by Question Type:**")
#             for q_type, stats in type_summary.items():
#                 rate = (stats['answered'] / stats['total']) * 100
#                 print(f"   {q_type.title()}: {stats['answered']}/{stats['total']} ({rate:.0f}%)")
        
#         print("=" * 50)
        
#         # Ask if user wants to review answers
#         review = input("\n🔍 Review your answers? (y/n): ").strip().lower()
#         if review.startswith('y'):
#             self._review_answers()
    
#     def _review_answers(self):
#         """Review quiz answers"""
        
#         print(f"\n📋 **ANSWER REVIEW**")
#         print("=" * 60)
        
#         for i, answer in enumerate(self.user_answers, 1):
#             print(f"\n{i}. **{answer['type'].title()} Question**")
#             print(f"Q: {answer['question']}")
#             print(f"A: {answer['answer']}")
#             print(f"Time: {answer['time_taken']:.1f}s")
#             print("─" * 40)
