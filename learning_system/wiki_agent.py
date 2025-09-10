from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging
import wikipedia
import re
from collections import Counter

# Import our modular tools
from tools.content_loader import ContentCache
from state_manager import StateManager
from groq import Groq

logger = logging.getLogger(__name__)

class WikiAgentState(TypedDict):
    user_id: str
    topic_name: str
    session_id: str
    content_cache: Dict[str, Any]
    extracted_keywords: List[str]
    content_domain: str
    enhanced_search_terms: List[str]
    wiki_content: str
    question_plan: List[Dict[str, Any]]
    total_questions: int
    current_question_index: int
    current_question: str
    user_answer: str
    feedback: str
    current_score: float
    qa_history: List[Dict[str, Any]]
    scores: List[float]
    average_score: float
    grade: str
    recommendation: str
    session_status: str

# Node 1: Load content and analyze domain context
def load_content_and_analyze_domain(state: WikiAgentState) -> WikiAgentState:
    """Load content and analyze domain using LLM for context-aware processing"""
    logger.info(f"📁 Loading content for topic: {state['topic_name']}")
    
    try:
        # Load session config using StateManager
        state_manager = StateManager()
        session_file = state_manager.find_latest_session(state['user_id'], state['topic_name'])
        
        if not session_file:
            raise FileNotFoundError(f"No session file found for {state['user_id']} - {state['topic_name']}")
        
        session_config = state_manager.load_quiz_session(session_file)
        if not session_config:
            raise ValueError("Failed to load session configuration")
        
        # Get folder path from md_file_path
        md_file_path = session_config['md_file_path']
        folder_path = os.path.dirname(md_file_path)
        
        # Use ContentCache to load and cache all content ONCE
        content_cache = ContentCache.load_content_cache(folder_path)
        
        # Extract text content for analysis
        combined_text = ContentCache.get_combined_text(content_cache, max_chars=2000)
        
        # Step 1: Extract keywords (existing logic)
        extracted_keywords = extract_keywords_from_markdown(combined_text, max_keywords=5)
        logger.info(f"📝 Extracted keywords: {extracted_keywords}")
        
        # Step 2: Analyze domain context using LLM
        content_domain = analyze_content_domain(combined_text)
        logger.info(f"🔍 Detected domain: {content_domain}")
        
        # Step 3: Generate enhanced search terms using domain context
        enhanced_search_terms = generate_enhanced_search_terms(
            extracted_keywords, content_domain, combined_text
        )
        logger.info(f"🎯 Enhanced search terms: {enhanced_search_terms}")
        
        return {
            **state,
            "content_cache": content_cache,
            "extracted_keywords": extracted_keywords,
            "content_domain": content_domain,
            "enhanced_search_terms": enhanced_search_terms,
            "session_id": os.path.basename(session_file),
            "current_question_index": 0,
            "qa_history": [],
            "scores": [],
            "session_status": "domain_analyzed"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load content: {str(e)}")
        return {
            **state,
            "session_status": "error"
        }

# Node 2: Fetch Wikipedia content using enhanced search terms
def fetch_enhanced_wikipedia_content(state: WikiAgentState) -> WikiAgentState:
    """Fetch Wikipedia content using enhanced, context-aware search terms"""
    logger.info("🌐 Fetching Wikipedia content using enhanced search terms...")
    
    try:
        wikipedia.set_lang("en")
        wiki_content = ""
        successful_searches = []
        
        # Search using enhanced terms with relevance validation
        for search_term in state['enhanced_search_terms'][:3]:  # Limit to top 3
            wiki_result = smart_wikipedia_search(search_term)
            if wiki_result:
                # Validate relevance before adding
                if validate_content_relevance(wiki_result, state['content_domain']):
                    wiki_content += wiki_result
                    successful_searches.append(search_term)
                    logger.info(f"✅ Added relevant content for: {search_term}")
                else:
                    logger.warning(f"⚠️ Content not relevant, skipping: {search_term}")
        
        # Enhanced fallback with domain context
        if not wiki_content.strip():
            fallback_term = f"{state['content_domain']} {state['extracted_keywords'][0] if state['extracted_keywords'] else state['topic_name']}"
            wiki_result = smart_wikipedia_search(fallback_term)
            if wiki_result:
                wiki_content = wiki_result
                successful_searches.append(fallback_term)
            else:
                wiki_content = f"This topic relates to {state['content_domain']} concepts, specifically focusing on {', '.join(state['extracted_keywords'])}."
        
        logger.info(f"✅ Wikipedia content fetched for: {successful_searches}")
        
        return {
            **state,
            "wiki_content": wiki_content,
            "session_status": "wikipedia_loaded"
        }
        
    except Exception as e:
        logger.error(f"❌ Wikipedia fetch failed: {str(e)}")
        return {
            **state,
            "wiki_content": f"Unable to fetch Wikipedia content. Topic involves {state['content_domain']} concepts.",
            "session_status": "wikipedia_fallback"
        }

# Helper Functions for Context-Aware Processing

def extract_keywords_from_markdown(md_text: str, max_keywords: int = 5) -> List[str]:
    """Extract relevant keywords from markdown content"""
    stopwords = {
        "the", "and", "is", "in", "to", "of", "a", "for", "with", "on", "as", "that", 
        "by", "an", "important", "topic", "using", "during", "it", "uses", "this", 
        "are", "we", "can", "was", "be", "but", "will", "have", "has", "when", 
        "where", "what", "how", "why", "also", "may", "such", "more", "make", 
        "used", "use", "get", "one", "two", "first", "second", "example", "like",
        "then", "than", "these", "those", "they", "them", "their", "there", "here"
    }
    
    # Remove markdown syntax
    text = re.sub(r'(!?\[.*?\]\(.*?\))', '', md_text)
    text = re.sub(r'[`*#>-]', ' ', text)
    text = text.lower()
    
    # Extract words (3+ letters)
    words = re.findall(r'\b[a-z]{3,}\b', text)
    filtered = [w for w in words if w not in stopwords]
    
    # Count frequencies and get top keywords
    counts = Counter(filtered)
    common = counts.most_common(max_keywords)
    
    keywords = [word for word, count in common]
    return keywords

def analyze_content_domain(content: str) -> str:
    """Analyze content using LLM to understand the domain/field"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        domain_analysis_prompt = f"""
You are an expert at understanding text domains. Analyze this content and identify the main field/domain in 2-3 words.

Content excerpt (first 800 characters):
{content[:800]}

Examples of good domain responses:
- neural networks
- machine learning  
- data science

Respond with ONLY the 2-3 word domain phrase, nothing else.
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": domain_analysis_prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=10,
            temperature=0.1
        )
        
        domain = response.choices[0].message.content.strip().lower()
        logger.info(f"🎯 LLM identified domain: {domain}")
        return domain
        
    except Exception as e:
        logger.warning(f"Domain analysis failed: {e}")
        # Fallback: simple keyword-based domain detection
        content_lower = content.lower()
        if any(term in content_lower for term in ['neural', 'network', 'gradient', 'weight']):
            return "neural networks"
        elif any(term in content_lower for term in ['machine', 'learning', 'algorithm', 'model']):
            return "machine learning"
        else:
            return "computer science"

def generate_enhanced_search_terms(keywords: List[str], domain: str, content: str) -> List[str]:
    """Generate enhanced Wikipedia search terms using LLM with domain context"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        enhancement_prompt = f"""
You are an expert in knowledge retrieval. Given the domain "{domain}" and these keywords extracted from academic content, generate precise Wikipedia search terms that would find the most relevant articles.

Domain: {domain}
Keywords: {', '.join(keywords)}
Content preview: {content[:300]}...

For each keyword, create a specific Wikipedia search term by adding appropriate context from the domain. 

Examples:
- If domain is "neural networks" and keyword is "weight" → "neural network weights"  
- If domain is "machine learning" and keyword is "initialization" → "weight initialization"

Generate exactly 3 enhanced search terms that would find relevant Wikipedia articles in this domain.
Return only the search terms, one per line, no explanations:
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": enhancement_prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=100,
            temperature=0.3
        )
        
        enhanced_terms = [line.strip() for line in response.choices[0].message.content.strip().split('\n') if line.strip()]
        
        # Clean and validate terms
        enhanced_terms = enhanced_terms[:3]  # Take max 3
        if not enhanced_terms:
            # Fallback: manually combine keywords with domain
            enhanced_terms = [f"{keyword} {domain}" for keyword in keywords[:3]]
        
        logger.info(f"🎯 Enhanced search terms generated: {enhanced_terms}")
        return enhanced_terms
        
    except Exception as e:
        logger.warning(f"Search term enhancement failed: {e}")
        # Fallback: combine keywords with domain
        return [f"{keyword} {domain}" for keyword in keywords[:3]]

def smart_wikipedia_search(search_term: str) -> Optional[str]:
    """Smart Wikipedia search with multiple fallback strategies"""
    try:
        # Strategy 1: Try search first to get better matches
        search_results = wikipedia.search(search_term, results=3)
        
        if search_results:
            for result in search_results:
                try:
                    summary = wikipedia.summary(result, sentences=3)
                    return f"\n--- {result} ---\n{summary}\n"
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try first disambiguation option
                    if e.options:
                        try:
                            summary = wikipedia.summary(e.options[0], sentences=3)
                            return f"\n--- {e.options[0]} ---\n{summary}\n"
                        except:
                            continue
                except wikipedia.exceptions.PageError:
                    continue
        
        # Strategy 2: Try direct page access
        try:
            summary = wikipedia.summary(search_term, sentences=3)
            return f"\n--- {search_term} ---\n{summary}\n"
        except:
            pass
            
    except Exception as e:
        logger.warning(f"Smart search failed for '{search_term}': {e}")
    
    return None

def validate_content_relevance(wiki_content: str, expected_domain: str) -> bool:
    """Validate if Wikipedia content is relevant to the expected domain"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        relevance_prompt = f"""
Determine if this Wikipedia content is relevant to the domain "{expected_domain}".

Wikipedia content (first 300 chars):
{wiki_content[:300]}

Expected domain: {expected_domain}

Is this content relevant to the expected domain? Respond with only "YES" or "NO".
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": relevance_prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=5,
            temperature=0.1
        )
        
        relevance = response.choices[0].message.content.strip().upper()
        return relevance == "YES"
        
    except Exception as e:
        logger.warning(f"Relevance validation failed: {e}")
        # If validation fails, assume relevant to avoid losing content
        return True

# Node 3: Generate questions using LLM based on Wikipedia content (SAME AS BEFORE)
def plan_wikipedia_questions(state: WikiAgentState) -> WikiAgentState:
    """Generate questions from Wikipedia content using LLM"""
    logger.info("🧠 Planning questions from Wikipedia content...")
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        planning_prompt = f"""
You are an expert question creator. Based on the Wikipedia content below, create exactly 3 interesting questions that expand knowledge beyond basic study material.

Wikipedia Content:
{state['wiki_content']}

Domain Context: {state['content_domain']}
Original Keywords: {', '.join(state['extracted_keywords'])}

Create questions that:
- Test broader understanding and context in the domain of {state['content_domain']}
- Connect to real-world applications
- Explore historical or research aspects
- Are answerable from the Wikipedia content provided

CRITICAL: Return ONLY a valid JSON array. No explanatory text before or after.

Return exactly this format:
[
  {{
    "question": "When was this concept first developed and by whom?",
    "difficulty": "medium",
    "expected_answer_type": "historical",
    "topic_area": "background"
  }},
  {{
    "question": "What real-world applications use this technology today?",
    "difficulty": "medium", 
    "expected_answer_type": "application",
    "topic_area": "practical"
  }},
  {{
    "question": "What are the main advantages of this approach?",
    "difficulty": "easy",
    "expected_answer_type": "concept",
    "topic_area": "understanding"
  }}
]
"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": planning_prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=600,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON using robust method
        question_plan = safe_extract_json_array(response_text)
        
        if not question_plan:
            raise ValueError("No valid JSON questions found in response")
        
        # Add metadata to each question
        for i, q in enumerate(question_plan):
            q["question_id"] = i + 1
            q["source"] = "wikipedia"
        
        total_questions = len(question_plan)
        logger.info(f"📋 Generated {total_questions} questions from Wikipedia content")
        
        return {
            **state,
            "question_plan": question_plan,
            "total_questions": total_questions,
            "session_status": "questions_planned"
        }
        
    except Exception as e:
        logger.error(f"❌ Question planning failed: {str(e)}")
        # Create fallback questions using enhanced search terms
        fallback_questions = []
        for i, term in enumerate(state['enhanced_search_terms'][:3]):
            fallback_questions.append({
                "question": f"What can you tell me about {term}?",
                "difficulty": "medium",
                "expected_answer_type": "concept",
                "topic_area": "general",
                "question_id": i + 1,
                "source": "fallback"
            })
        
        return {
            **state,
            "question_plan": fallback_questions,
            "total_questions": len(fallback_questions),
            "session_status": "questions_planned"
        }

def safe_extract_json_array(response_text: str) -> List[Dict[str, Any]]:
    """Safely extract JSON array from LLM response with extra text"""
    
    # Method 1: Try to find complete JSON array
    try:
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            json_str = re.sub(r'\n\s*\n', '\n', json_str)
            json_str = json_str.strip()
            
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Extract individual JSON objects
    try:
        json_objects = []
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                obj = json.loads(match.strip())
                if 'question' in obj:
                    json_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        if json_objects:
            return json_objects
            
    except Exception:
        pass
    
    return []

# Remaining nodes (same as before): present_wikipedia_question, collect_user_answer, 
# provide_feedback_and_score, session_manager, finalize_session

# Node 4: Present Wikipedia question (SAME AS BEFORE)
def present_wikipedia_question(state: WikiAgentState) -> WikiAgentState:
    current_index = state['current_question_index']
    question_plan = state['question_plan']
    
    if current_index >= len(question_plan):
        return {
            **state,
            "session_status": "questions_complete"
        }
    
    question_info = question_plan[current_index]
    current_question = question_info['question']
    
    print(f"\n🌐 Wikipedia Question {current_index + 1}/{state['total_questions']} ({question_info.get('difficulty', 'medium')})")
    print(f"📝 {current_question}")
    print(f"💡 Based on {state['content_domain']} knowledge from Wikipedia")
    
    return {
        **state,
        "current_question": current_question,
        "session_status": "question_presented"
    }

# Node 5: Collect user answer (SAME AS BEFORE)
def collect_user_answer(state: WikiAgentState) -> WikiAgentState:
    user_answer = input("\n👤 Your answer: ").strip()
    logger.info(f"📝 User answered: '{user_answer[:50]}...'")
    
    return {
        **state,
        "user_answer": user_answer,
        "session_status": "answer_collected"
    }

# Node 6: Provide feedback and score (SAME AS BEFORE WITH DOMAIN CONTEXT)
def provide_feedback_and_score(state: WikiAgentState) -> WikiAgentState:
    current_question = state['current_question']
    user_answer = state['user_answer']
    wiki_context = state['wiki_content'][:1000]
    
    scoring_prompt = f"""
Evaluate this user's answer based on the Wikipedia content provided.

Domain: {state['content_domain']}
Question: {current_question}
User's Answer: {user_answer}

Wikipedia Context: {wiki_context}

Score from 0-100 based on:
- Accuracy relative to Wikipedia content (50%)
- Completeness of answer (30%)
- Relevance and understanding (20%)

Provide brief feedback in one sentence, then give numeric score.
Format: "Feedback: [one sentence] | Score: [0-100]"
"""
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": scoring_prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=150,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Robust score extraction
        score_match = re.search(r'Score:\s*(\d+\.?\d*)', response_text, re.IGNORECASE)
        if score_match:
            current_score = float(score_match.group(1))
            feedback = response_text.split(score_match.group(0))[0].replace("Feedback:", "").strip()
        else:
            numbers = re.findall(r'\b(\d{1,2}\.?\d*)\b', response_text)
            valid_scores = [float(num) for num in numbers if 0 <= float(num) <= 100]
            
            if valid_scores:
                current_score = valid_scores[0]
                feedback = response_text.replace(str(current_score), "").replace("Feedback:", "").strip()
            else:
                current_score = 50.0
                feedback = response_text.replace("Feedback:", "").strip()
        
        current_score = max(0, min(100, current_score))
        
    except Exception as e:
        logger.error(f"❌ Feedback generation failed: {str(e)}")
        feedback = f"Thank you for your answer about {state['content_domain']}!"
        current_score = 50.0
    
    print(f"\n💭 Feedback: {feedback}")
    print(f"🎯 Score: {current_score:.1f}/100")
    
    # Add to history
    qa_entry = {
        "question": current_question,
        "answer": user_answer,
        "feedback": feedback,
        "score": current_score,
        "timestamp": datetime.now().isoformat(),
        "source": "wikipedia",
        "domain": state['content_domain']
    }
    
    qa_history = state['qa_history'] + [qa_entry]
    scores = state['scores'] + [current_score]
    
    return {
        **state,
        "feedback": feedback,
        "current_score": current_score,
        "qa_history": qa_history,
        "scores": scores,
        "current_question_index": state['current_question_index'] + 1,
        "session_status": "answer_scored"
    }

# Node 7: Session manager (SAME AS BEFORE)
def session_manager(state: WikiAgentState) -> str:
    current_index = state['current_question_index']
    total_questions = state['total_questions']
    
    if current_index < total_questions:
        return "present_wikipedia_question"
    else:
        return "finalize_session"

# Node 8: Finalize session (SAME AS BEFORE)
def finalize_session(state: WikiAgentState) -> WikiAgentState:
    scores = state['scores']
    
    if not scores:
        average_score = 0
        grade = "🔴 No Answers"
        recommendation = "Please attempt the Wikipedia questions."
    else:
        average_score = sum(scores) / len(scores)
        
        if average_score >= 60:
            grade = "🟢 Wikipedia Master"
            recommendation = f"Excellent! You have strong broader knowledge of {state['content_domain']}."
        elif average_score >= 40:
            grade = "🟡 Good Context"
            recommendation = f"Good understanding. Keep exploring {state['content_domain']} topics."
        else:
            grade = "🔴 Needs Exploration"
            recommendation = f"Explore more {state['content_domain']} articles to broaden your knowledge."
    
    # Save results using StateManager
    try:
        state_manager = StateManager()
        session_result = {
            "user_id": state['user_id'],
            "topic_name": state['topic_name'],
            "session_id": state['session_id'],
            "questions_answered": len(scores),
            "individual_scores": scores,
            "average_score": average_score,
            "grade": grade,
            "recommendation": recommendation,
            "qa_history": state['qa_history'],
            "completion_time": datetime.now().isoformat(),
            "session_type": "wiki_agent",
            "keywords_used": state['extracted_keywords'],
            "content_domain": state['content_domain'],
            "enhanced_search_terms": state['enhanced_search_terms']
        }
        
        state_manager.save_quiz_results(state['user_id'], state['topic_name'], session_result)
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {str(e)}")
    
    print(f"\n" + "="*60)
    print(f"🌐 WIKIPEDIA AGENT SESSION COMPLETE!")
    print(f"="*60)
    print(f"📚 Topic: {state['topic_name']}")
    print(f"🎯 Domain: {state['content_domain']}")
    print(f"🔍 Keywords Used: {', '.join(state['extracted_keywords'])}")
    print(f"🎯 Enhanced Search Terms: {', '.join(state['enhanced_search_terms'])}")
    print(f"❓ Questions Answered: {len(scores)}")
    print(f"📊 Individual Scores: {[f'{s:.1f}' for s in scores]}")
    print(f"🏆 Average Score: {average_score:.2f}/100")
    print(f"🎯 Final Grade: {grade}")
    print(f"💡 Recommendation: {recommendation}")
    print(f"="*60)
    
    return {
        **state,
        "average_score": average_score,
        "grade": grade,
        "recommendation": recommendation,
        "session_status": "completed"
    }

# Build the workflow
def create_wiki_agent_workflow():
    """Create the Context-Aware Wiki Agent LangGraph workflow"""
    workflow = StateGraph(WikiAgentState)
    
    # Add nodes
    workflow.add_node("load_content_and_analyze_domain", load_content_and_analyze_domain)
    workflow.add_node("fetch_enhanced_wikipedia_content", fetch_enhanced_wikipedia_content)
    workflow.add_node("plan_wikipedia_questions", plan_wikipedia_questions)
    workflow.add_node("present_wikipedia_question", present_wikipedia_question)
    workflow.add_node("collect_user_answer", collect_user_answer)
    workflow.add_node("provide_feedback_and_score", provide_feedback_and_score)
    workflow.add_node("finalize_session", finalize_session)
    
    # Define edges
    workflow.set_entry_point("load_content_and_analyze_domain")
    workflow.add_edge("load_content_and_analyze_domain", "fetch_enhanced_wikipedia_content")
    workflow.add_edge("fetch_enhanced_wikipedia_content", "plan_wikipedia_questions")
    workflow.add_edge("plan_wikipedia_questions", "present_wikipedia_question")
    workflow.add_edge("present_wikipedia_question", "collect_user_answer")
    workflow.add_edge("collect_user_answer", "provide_feedback_and_score")
    
    # Conditional edge for session management
    workflow.add_conditional_edges(
        "provide_feedback_and_score",
        session_manager,
        {
            "present_wikipedia_question": "present_wikipedia_question",
            "finalize_session": "finalize_session"
        }
    )
    
    workflow.add_edge("finalize_session", END)
    
    return workflow.compile()

# Main function
def run_wiki_agent_session(user_id: str, topic_name: str):
    """Run Context-Aware Wiki Agent session"""
    print(f"\n🌐 Context-Aware Wiki Agent - Zero-Knowledge Wikipedia Questions")
    print(f"👤 User: {user_id}")
    print(f"📚 Topic: {topic_name}")
    print("="*60)
    
    initial_state: WikiAgentState = {
        "user_id": user_id,
        "topic_name": topic_name,
        "session_id": "",
        "content_cache": {},
        "extracted_keywords": [],
        "content_domain": "",
        "enhanced_search_terms": [],
        "wiki_content": "",
        "question_plan": [],
        "total_questions": 0,
        "current_question_index": 0,
        "current_question": "",
        "user_answer": "",
        "feedback": "",
        "current_score": 0.0,
        "qa_history": [],
        "scores": [],
        "average_score": 0.0,
        "grade": "",
        "recommendation": "",
        "session_status": "starting"
    }
    
    try:
        workflow = create_wiki_agent_workflow()
        result = workflow.invoke(
            initial_state,
            config={"recursion_limit": 30}  # Increased for additional processing steps
        )
        
        return {
            "status": "success",
            "average_score": result['average_score'],
            "grade": result['grade'],
            "recommendation": result['recommendation'],
            "questions_answered": len(result['scores']),
            "qa_history": result['qa_history'],
            "content_domain": result['content_domain'],
            "enhanced_search_terms": result['enhanced_search_terms']
        }
        
    except Exception as e:
        logger.error(f"❌ Wiki Agent workflow failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
