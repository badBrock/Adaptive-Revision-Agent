from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

class MetaLearningModule:
    """Memory-Enhanced Meta-Learning for adaptive tutoring"""
    
    def __init__(self, storage_path: str = "./user_profiles"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def _get_profile_path(self, user_id: str) -> str:
        return os.path.join(self.storage_path, f"{user_id}_meta_profile.json")
    
    def _load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user's meta-learning profile"""
        profile_path = self._get_profile_path(user_id)
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "topics": {},
            "learning_patterns": {},
            "agent_effectiveness": {
                "quiz_agent": {"success_rate": 0.5, "adaptations": []},
                "teaching_agent": {"success_rate": 0.5, "adaptations": []}
            }
        }
    
    def _save_profile(self, user_id: str, profile: Dict[str, Any]):
        """Save user's meta-learning profile"""
        profile["last_updated"] = datetime.now().isoformat()
        profile_path = self._get_profile_path(user_id)
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def record_session_outcome(self, user_id: str, topic: str, agent_type: str, 
                              session_data: Dict[str, Any]):
        """Record the outcome of a learning session"""
        profile = self._load_profile(user_id)
        
        # Initialize topic if new
        if topic not in profile["topics"]:
            profile["topics"][topic] = {
                "sessions": [],
                "error_patterns": {},
                "learning_velocity": 1.0,
                "optimal_difficulty": "medium",
                "preferred_question_types": []
            }
        
        # Record session
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "score": session_data.get("average_score", 0),
            "questions_answered": session_data.get("questions_answered", 0),
            "time_taken": session_data.get("time_taken", 0),
            "mistakes": session_data.get("mistakes", []),
            "feedback_effectiveness": session_data.get("feedback_rating", 3)  # 1-5 scale
        }
        
        profile["topics"][topic]["sessions"].append(session_record)
        
        # Update agent effectiveness
        self._update_agent_effectiveness(profile, agent_type, session_data)
        
        # Analyze learning patterns
        self._analyze_learning_patterns(profile, topic)
        
        self._save_profile(user_id, profile)
        logger.info(f"📊 Meta-learning recorded: {user_id} - {topic} - {agent_type}")
    
    def _update_agent_effectiveness(self, profile: Dict, agent_type: str, session_data: Dict):
        """Track how effective each agent is for this user"""
        agent_stats = profile["agent_effectiveness"][agent_type]
        
        # Simple effectiveness metric based on score improvement
        score = session_data.get("average_score", 0)
        if score >= 80:
            effectiveness = 0.9
        elif score >= 60:
            effectiveness = 0.7
        else:
            effectiveness = 0.3
        
        # Update running average
        current_rate = agent_stats["success_rate"]
        agent_stats["success_rate"] = (current_rate * 0.8) + (effectiveness * 0.2)
    
    def _analyze_learning_patterns(self, profile: Dict, topic: str):
        """Analyze user's learning patterns for this topic"""
        sessions = profile["topics"][topic]["sessions"]
        if len(sessions) < 2:
            return
        
        recent_sessions = sessions[-5:]  # Last 5 sessions
        
        # Calculate learning velocity (score improvement rate)
        scores = [s["score"] for s in recent_sessions]
        if len(scores) > 1:
            velocity = (scores[-1] - scores[0]) / len(scores)
            profile["topics"][topic]["learning_velocity"] = velocity
        
        # Analyze error patterns
        all_mistakes = []
        for session in recent_sessions:
            all_mistakes.extend(session.get("mistakes", []))
        
        # Count mistake types (this would be more sophisticated in practice)
        error_counts = {}
        for mistake in all_mistakes:
            error_type = mistake.get("type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        profile["topics"][topic]["error_patterns"] = error_counts
    
    def get_learning_recommendations(self, user_id: str, topic: str) -> Dict[str, Any]:
        """Get personalized learning recommendations"""
        profile = self._load_profile(user_id)
        
        if topic not in profile["topics"]:
            return {
                "recommended_agent": "quiz_agent",
                "difficulty": "medium",
                "focus_areas": [],
                "reasoning": "New topic - starting with assessment"
            }
        
        topic_data = profile["topics"][topic]
        sessions = topic_data["sessions"]
        
        if not sessions:
            return {
                "recommended_agent": "quiz_agent",
                "difficulty": "medium", 
                "focus_areas": [],
                "reasoning": "No session history - starting with assessment"
            }
        
        # Analyze recent performance
        recent_sessions = sessions[-3:]
        avg_score = sum(s["score"] for s in recent_sessions) / len(recent_sessions)
        
        # Check agent effectiveness
        quiz_effectiveness = profile["agent_effectiveness"]["quiz_agent"]["success_rate"]
        teaching_effectiveness = profile["agent_effectiveness"]["teaching_agent"]["success_rate"]
        
        # Determine recommendation
        if avg_score < 40:
            recommended_agent = "quiz_agent"
            difficulty = "easy"
            reasoning = "Low scores - need fundamental review"
        elif avg_score < 70:
            # Choose more effective agent
            if quiz_effectiveness > teaching_effectiveness:
                recommended_agent = "quiz_agent"
            else:
                recommended_agent = "teaching_agent"
            difficulty = "medium"
            reasoning = f"Moderate scores - using more effective agent ({recommended_agent})"
        else:
            recommended_agent = "teaching_agent"
            difficulty = "hard"
            reasoning = "High scores - ready for teaching/explanation practice"
        
        # Identify focus areas from error patterns
        error_patterns = topic_data.get("error_patterns", {})
        focus_areas = sorted(error_patterns.keys(), key=lambda x: error_patterns[x], reverse=True)[:3]
        
        return {
            "recommended_agent": recommended_agent,
            "difficulty": difficulty,
            "focus_areas": focus_areas,
            "reasoning": reasoning,
            "learning_velocity": topic_data.get("learning_velocity", 1.0),
            "agent_effectiveness": {
                "quiz": quiz_effectiveness,
                "teaching": teaching_effectiveness
            }
        }
    
    def get_agent_adaptations(self, user_id: str, topic: str, agent_type: str) -> Dict[str, Any]:
        """Get specific adaptations for an agent based on user's learning patterns"""
        profile = self._load_profile(user_id)
        recommendations = self.get_learning_recommendations(user_id, topic)
        
        adaptations = {
            "difficulty_adjustment": recommendations["difficulty"],
            "focus_areas": recommendations["focus_areas"],
            "learning_velocity": recommendations.get("learning_velocity", 1.0)
        }
        
        if agent_type == "quiz_agent":
            adaptations.update({
                "question_types": self._get_optimal_question_types(profile, topic),
                "pacing": "fast" if adaptations["learning_velocity"] > 0.5 else "slow"
            })
        elif agent_type == "teaching_agent":
            adaptations.update({
                "explanation_style": self._get_optimal_explanation_style(profile, topic),
                "prompt_complexity": "advanced" if recommendations["difficulty"] == "hard" else "basic"
            })
        
        return adaptations
    
    def _get_optimal_question_types(self, profile: Dict, topic: str) -> List[str]:
        """Determine optimal question types for this user"""
        # This would analyze which question types the user performs best on
        return ["definition", "application", "analysis"]  # Simplified
    
    def _get_optimal_explanation_style(self, profile: Dict, topic: str) -> str:
        """Determine optimal explanation style for this user"""
        # This would analyze which explanation styles work best
        return "detailed"  # Simplified
