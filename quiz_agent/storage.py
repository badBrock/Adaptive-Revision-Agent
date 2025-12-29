"""
Storage handlers for topics and sessions
"""

import json
import hashlib
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from models import Topic
from config import KEYWORD_MIN_LENGTH, TOPIC_NAME_MAX_WORDS

class TopicKnowledgeBase:
    """Manages topic-based learning progress"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._initialize()
    
    def _initialize(self):
        if not self.file_path.exists():
            self._write({"topics": {}, "keyword_to_topic": {}})
    
    def _read(self) -> dict:
        try:
            return json.loads(self.file_path.read_text())
        except:
            return {"topics": {}, "keyword_to_topic": {}}
    
    def _write(self, data: dict):
        self.file_path.write_text(json.dumps(data, indent=2))
    
    def _clean_topic_name(self, name: str) -> str:
        """Clean topic name from markdown artifacts"""
        cleaned = name.replace('**', '').replace('#', '').replace('*', '')
        cleaned = ' '.join(cleaned.split())
        
        words = cleaned.split()
        if len(words) > 0:
            meaningful_words = [w for w in words if len(w) > KEYWORD_MIN_LENGTH and not w.isdigit()][:TOPIC_NAME_MAX_WORDS]
            if meaningful_words:
                return ' '.join(word.capitalize() for word in meaningful_words)
        
        return "General Topic"
    
    def identify_topic(self, keywords: list[str], content_sample: str) -> tuple[str, str, float]:
        """Identify topic from keywords or create new one"""
        data = self._read()
        
        clean_keywords = []
        for kw in keywords[:10]:
            cleaned = kw.replace('**', '').replace('#', '').strip()
            if len(cleaned) > KEYWORD_MIN_LENGTH and not cleaned.isdigit():
                clean_keywords.append(cleaned)
        
        if not clean_keywords:
            clean_keywords = ["general", "topic"]
        
        keyword_set = set(kw.lower() for kw in clean_keywords[:5])
        
        best_match_id = None
        best_match_score = 0
        
        for topic_id, topic_data in data["topics"].items():
            topic_keywords = set(kw.lower() for kw in topic_data["keywords"])
            overlap = len(keyword_set & topic_keywords)
            
            if overlap >= 2:
                match_score = overlap / len(keyword_set | topic_keywords)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_id = topic_id
        
        if best_match_id and best_match_score > 0.3:
            topic_data = data["topics"][best_match_id]
            return (best_match_id, topic_data["name"], topic_data["mastery_score"])
        else:
            topic_name = self._generate_topic_name(clean_keywords, content_sample)
            topic_id = f"topic_{hashlib.md5(topic_name.encode()).hexdigest()[:12]}"
            
            new_topic = Topic(
                id=topic_id,
                name=topic_name,
                keywords=clean_keywords[:5],
                mastery_score=0.0,
                question_count=0,
                total_score=0,
                last_updated=datetime.now().isoformat(),
                content_samples=[content_sample[:200]]
            )
            
            data["topics"][topic_id] = asdict(new_topic)
            
            for kw in clean_keywords[:5]:
                data["keyword_to_topic"][kw.lower()] = topic_id
            
            self._write(data)
            
            return (topic_id, topic_name, 0.0)
    
    def _generate_topic_name(self, keywords: list[str], content: str) -> str:
        """Generate meaningful topic name"""
        meaningful = [kw for kw in keywords if len(kw) > 3 and not any(char.isdigit() for char in kw)]
        
        if not meaningful:
            meaningful = keywords
        
        top = meaningful[:3]
        
        if len(top) == 1:
            return self._clean_topic_name(top[0])
        elif len(top) == 2:
            return self._clean_topic_name(f"{top[0]} and {top[1]}")
        else:
            return self._clean_topic_name(f"{top[0]}, {top[1]}, {top[2]}")
    
    def update_topic_score(self, topic_id: str, new_score: int, question_text: str, answer: str, feedback: str):
        """Update topic mastery"""
        data = self._read()
        
        if topic_id not in data["topics"]:
            return
        
        topic = data["topics"][topic_id]
        
        topic["question_count"] += 1
        topic["total_score"] += new_score
        topic["mastery_score"] = topic["total_score"] / topic["question_count"]
        topic["last_updated"] = datetime.now().isoformat()
        
        if "questions" not in topic:
            topic["questions"] = []
        
        topic["questions"].append({
            "question": question_text,
            "answer": answer,
            "score": new_score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        data["topics"][topic_id] = topic
        self._write(data)
    
    def get_topic(self, topic_id: str) -> dict:
        data = self._read()
        return data["topics"].get(topic_id, {})
    
    def get_all_topics(self) -> list[dict]:
        data = self._read()
        return list(data["topics"].values())


class SessionStorage:
    """Store session data"""
    
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
    
    def save_session(self, session_id: str, session_data: dict):
        session_file = self.sessions_dir / f"{session_id}.json"
        session_file.write_text(json.dumps(session_data, indent=2))
