import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StateManager:
    """Centralized state management for user profiles and quiz sessions"""
    
    def __init__(self, base_path: str = "./states"):
        self.base_path = base_path
        self.user_profiles_path = os.path.join(base_path, "user_profiles")
        self.quiz_sessions_path = os.path.join(base_path, "quiz_sessions")
        self.quiz_results_path = os.path.join(base_path, "quiz_results")
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for path in [self.user_profiles_path, self.quiz_sessions_path, self.quiz_results_path]:
            os.makedirs(path, exist_ok=True)
    
    # User Profile Management
    def load_user_state(self, user_id: str) -> Dict[str, Any]:
        """Load user's learning state from JSON file"""
        file_path = os.path.join(self.user_profiles_path, f"{user_id}.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    user_state = json.load(f)
                logger.info(f"✅ Loaded user state for '{user_id}'")
                return user_state
            except Exception as e:
                logger.error(f"❌ Failed to load user state for '{user_id}': {str(e)}")
                return self._create_default_user_state(user_id)
        else:
            logger.info(f"🆕 Creating new user state for '{user_id}'")
            return self._create_default_user_state(user_id)
    
    def save_user_state(self, user_id: str, user_state: Dict[str, Any]) -> bool:
        """Save user's learning state to JSON file"""
        file_path = os.path.join(self.user_profiles_path, f"{user_id}.json")
        
        try:
            # Update timestamp
            user_state["last_updated"] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(user_state, f, indent=4, ensure_ascii=False)
            
            logger.info(f"✅ Saved user state for '{user_id}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save user state for '{user_id}': {str(e)}")
            return False
    
    def _create_default_user_state(self, user_id: str) -> Dict[str, Any]:
        """Create default user state structure"""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "last_session": None,
            "topics": {},  # {topic_name: {score: float, attempts: int, last_studied: str, best_score: float}}
            "learning_preferences": {
                "difficulty_preference": "medium",
                "session_length": "standard",
                "question_count": 8
            },
            "total_sessions": 0,
            "overall_progress": 0.0,
            "learning_streak": 0,
            "achievements": []
        }
    
    def update_topic_progress(self, user_id: str, topic_name: str, quiz_results: Dict[str, Any]) -> bool:
        """Update user's progress for a specific topic"""
        user_state = self.load_user_state(user_id)
        
        if "topics" not in user_state:
            user_state["topics"] = {}
        
        if topic_name not in user_state["topics"]:
            user_state["topics"][topic_name] = {
                "score": 0,
                "attempts": 0,
                "last_studied": None,
                "best_score": 0,
                "first_attempt": datetime.now().isoformat()
            }
        
        topic_data = user_state["topics"][topic_name]
        current_score = quiz_results.get("average_score", 0)
        
        # Update topic data
        topic_data["score"] = current_score
        topic_data["attempts"] += 1
        topic_data["last_studied"] = datetime.now().isoformat()
        topic_data["best_score"] = max(topic_data.get("best_score", 0), current_score)
        
        # Update overall progress
        user_state["total_sessions"] += 1
        user_state["last_session"] = datetime.now().isoformat()
        
        # Calculate overall progress
        if user_state["topics"]:
            avg_score = sum(t.get("score", 0) for t in user_state["topics"].values()) / len(user_state["topics"])
            user_state["overall_progress"] = avg_score
        
        # Update learning streak
        self._update_learning_streak(user_state, current_score)
        
        return self.save_user_state(user_id, user_state)
    
    def _update_learning_streak(self, user_state: Dict[str, Any], current_score: float):
        """Update learning streak based on performance"""
        if current_score >= 70:  # Good performance threshold
            user_state["learning_streak"] = user_state.get("learning_streak", 0) + 1
        else:
            user_state["learning_streak"] = 0
    
    # Quiz Session Management
    def create_quiz_session(self, user_id: str, topic_name: str, session_config: Dict[str, Any]) -> str:
        """Create a new quiz session file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = f"{user_id}_{topic_name}_{timestamp}.json"
        session_path = os.path.join(self.quiz_sessions_path, session_filename)
        
        session_data = {
            "session_id": session_filename,
            "user_id": user_id,
            "topic_name": topic_name,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            **session_config
        }
        
        try:
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"📄 Created quiz session: {session_filename}")
            return session_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create quiz session: {str(e)}")
            return None
    
    def load_quiz_session(self, session_path: str) -> Optional[Dict[str, Any]]:
        """Load quiz session data"""
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            return session_data
        except Exception as e:
            logger.error(f"❌ Failed to load quiz session: {str(e)}")
            return None
    
    def find_latest_session(self, user_id: str, topic_name: str) -> Optional[str]:
        """Find the latest session file for a user and topic"""
        pattern = f"{user_id}_{topic_name}_*.json"
        session_files = list(Path(self.quiz_sessions_path).glob(pattern))
        
        if session_files:
            # Return the most recently created file
            latest_file = max(session_files, key=os.path.getctime)
            return str(latest_file)
        else:
            return None
    
    # Quiz Results Management
    def save_quiz_results(self, user_id: str, topic_name: str, results: Dict[str, Any]) -> str:
        """Save quiz results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{user_id}_{topic_name}_results_{timestamp}.json"
        results_path = os.path.join(self.quiz_results_path, results_filename)
        
        results_data = {
            "results_id": results_filename,
            "user_id": user_id,
            "topic_name": topic_name,
            "completed_at": datetime.now().isoformat(),
            **results
        }
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"📊 Saved quiz results: {results_filename}")
            return results_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save quiz results: {str(e)}")
            return None
    
    def load_quiz_results(self, results_path: str) -> Optional[Dict[str, Any]]:
        """Load quiz results"""
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            return results_data
        except Exception as e:
            logger.error(f"❌ Failed to load quiz results: {str(e)}")
            return None
    
    # Analytics and Reporting
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        user_state = self.load_user_state(user_id)
        
        if not user_state.get("topics"):
            return {
                "user_id": user_id,
                "total_sessions": 0,
                "topics_studied": 0,
                "overall_progress": 0.0,
                "learning_streak": 0,
                "performance_summary": "No learning history"
            }
        
        topics = user_state["topics"]
        
        analytics = {
            "user_id": user_id,
            "total_sessions": user_state.get("total_sessions", 0),
            "topics_studied": len(topics),
            "overall_progress": user_state.get("overall_progress", 0.0),
            "learning_streak": user_state.get("learning_streak", 0),
            "best_topics": self._get_best_topics(topics),
            "struggling_topics": self._get_struggling_topics(topics),
            "recent_activity": self._get_recent_activity(topics),
            "performance_trend": self._calculate_performance_trend(topics)
        }
        
        return analytics
    
    def _get_best_topics(self, topics: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Get topics with best performance"""
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: x[1].get("best_score", 0),
            reverse=True
        )
        
        return [
            {
                "topic": topic,
                "best_score": data.get("best_score", 0),
                "attempts": data.get("attempts", 0)
            }
            for topic, data in sorted_topics[:limit]
        ]
    
    def _get_struggling_topics(self, topics: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Get topics that need improvement"""
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: x[1].get("score", 0)
        )
        
        struggling = [
            {
                "topic": topic,
                "current_score": data.get("score", 0),
                "attempts": data.get("attempts", 0)
            }
            for topic, data in sorted_topics
            if data.get("score", 0) < 70  # Below "good" threshold
        ]
        
        return struggling[:limit]
    
    def _get_recent_activity(self, topics: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent learning activity"""
        topics_with_dates = [
            (topic, data)
            for topic, data in topics.items()
            if data.get("last_studied")
        ]
        
        sorted_by_date = sorted(
            topics_with_dates,
            key=lambda x: x[1]["last_studied"],
            reverse=True
        )
        
        return [
            {
                "topic": topic,
                "last_studied": data["last_studied"],
                "score": data.get("score", 0)
            }
            for topic, data in sorted_by_date[:limit]
        ]
    
    def _calculate_performance_trend(self, topics: Dict[str, Any]) -> str:
        """Calculate overall performance trend"""
        if len(topics) < 2:
            return "insufficient_data"
        
        scores = [data.get("score", 0) for data in topics.values()]
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 80:
            return "excellent"
        elif avg_score >= 70:
            return "good"
        elif avg_score >= 60:
            return "needs_improvement"
        else:
            return "requires_review"

    # Cleanup utilities
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old session files"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for session_file in Path(self.quiz_sessions_path).glob("*.json"):
            if session_file.stat().st_mtime < cutoff_time:
                try:
                    session_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {session_file}: {str(e)}")
        
        logger.info(f"🧹 Cleaned up {cleaned_count} old session files")
        return cleaned_count
