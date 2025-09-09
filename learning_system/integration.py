import os
import json
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_integrated_learning_system(user_id: str, content_folder_path: str) -> Dict[str, Any]:
    """
    Complete integrated learning system workflow:
    Boss Agent → Quiz Agent → Boss Agent (with results)
    """
    print(f"\n🤖 INTEGRATED LEARNING SYSTEM")
    print(f"="*60)
    print(f"👤 User: {user_id}")
    print(f"📁 Content: {content_folder_path}")
    print(f"="*60)
    
    try:
        # Step 1: Run Boss Agent to decide what to study (WITHOUT quiz_results)
        print(f"\n🧠 STEP 1: Boss Agent Decision Making...")
        from boss_agent import run_boss_agent_session
        
        boss_result = run_boss_agent_session(user_id, content_folder_path, quiz_results=None)
        
        # Check if Boss Agent completed successfully and provided a recommendation
       # Replace the current validation with:
        if boss_result.get('session_status') in ['error', 'workflow_error']:
            print(f"❌ Boss Agent failed: {boss_result.get('error_message', 'Unknown error')}")
            return {
                "status": "boss_agent_failed",
                "error": boss_result.get('error_message', 'Boss Agent did not complete successfully'),
                "boss_result": boss_result
            }

        
        # Extract recommendation
        decision = boss_result.get('decision', {})
        recommended_topic = decision.get('recommended_topic')
        
        if not recommended_topic:
            print(f"❌ Boss Agent did not recommend a topic")
            return {
                "status": "no_topic_recommended",
                "error": "Boss Agent failed to recommend a topic",
                "boss_result": boss_result
            }
        
        print(f"✅ Boss Agent recommends studying: '{recommended_topic}'")
        print(f"📚 Learning Strategy: {decision.get('learning_strategy', 'unknown')}")
        
        # Step 2: Run Quiz Agent for the recommended topic
        print(f"\n🎯 STEP 2: Quiz Agent Session...")
        from quiz_agent import run_quiz_agent_session
        
        quiz_result = run_quiz_agent_session(user_id, recommended_topic)
        
        # Check if Quiz Agent completed successfully
        if quiz_result.get('status') != 'success':
            print(f"❌ Quiz Agent failed: {quiz_result.get('error', 'Unknown error')}")
            return {
                "status": "quiz_agent_failed",
                "error": quiz_result.get('error', 'Quiz Agent failed'),
                "boss_result": boss_result,
                "quiz_result": quiz_result
            }
        
        print(f"✅ Quiz completed!")
        print(f"🏆 Final Score: {quiz_result['average_score']:.2f}/100")
        print(f"🎯 Grade: {quiz_result['grade']}")
        
        # Step 3: Update Boss Agent with quiz results
        print(f"\n📊 STEP 3: Updating Boss Agent with Quiz Results...")
        
        # Run Boss Agent again WITH quiz results to update progress
        final_boss_result = run_boss_agent_session(
            user_id, 
            content_folder_path, 
            quiz_results=quiz_result  # Pass quiz results for progress update
        )
        
        # Check if progress update was successful
        if final_boss_result.get('session_status') not in ['progress_updated', 'completed_successfully']:
            print(f"⚠️ Warning: Progress update may have failed: {final_boss_result.get('error_message')}")
        else:
            print(f"✅ Learning progress updated successfully!")
        
        # Calculate overall progress
        updated_user_state = final_boss_result.get('user_state', {})
        overall_progress = updated_user_state.get('overall_progress', 0.0)
        
        # Final summary
        print(f"\n🎉 LEARNING SESSION COMPLETE!")
        print(f"="*60)
        print(f"📚 Topic Studied: {recommended_topic}")
        print(f"❓ Questions Answered: {quiz_result['questions_answered']}")
        print(f"🏆 Session Score: {quiz_result['average_score']:.2f}/100")
        print(f"🎯 Grade: {quiz_result['grade']}")
        print(f"📈 Overall Progress: {overall_progress:.1f}%")
        print(f"💡 Recommendation: {quiz_result['recommendation']}")
        print(f"="*60)
        
        return {
            "status": "success",
            "user_id": user_id,
            "session_summary": {
                "topic_studied": recommended_topic,
                "learning_strategy": decision.get('learning_strategy'),
                "questions_answered": quiz_result['questions_answered'],
                "session_score": quiz_result['average_score'],
                "grade": quiz_result['grade'],
                "overall_progress": overall_progress
            },
            "boss_decision": decision,
            "quiz_results": quiz_result,
            "updated_user_state": updated_user_state
        }
        
    except ImportError as e:
        error_msg = f"Missing module: {str(e)}"
        print(f"❌ Import Error: {error_msg}")
        return {
            "status": "import_error",
            "error": error_msg
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ Unexpected Error: {error_msg}")
        logger.exception("Integration workflow failed")
        return {
            "status": "unexpected_error",
            "error": error_msg
        }

def main():
    """Main function with user input"""
    print("🎓 Integrated Learning System - Boss Agent + Quiz Agent")
    print("="*60)
    
    # Get user input
    try:
        user_id = input("Enter User ID: ").strip() or "student_001"
        content_folder = input("Enter content folder path: ").strip() or "./content"
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Validate content folder exists
    if not os.path.exists(content_folder):
        print(f"❌ Content folder does not exist: {content_folder}")
        print("Please create the folder and add some .md files for learning content.")
        return
    
    # Run integrated system
    result = run_integrated_learning_system(user_id, content_folder)
    
    # Handle different outcomes
    if result["status"] == "success":
        print(f"\n✅ Integration completed successfully!")
        
        # Ask if user wants to run another session
        try:
            another = input("\nWould you like to run another learning session? (y/n): ").strip().lower()
            if another == 'y' or another == 'yes':
                main()  # Recursive call for another session
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print(f"\n❌ Integration failed with status: {result['status']}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Debugging info
        if result.get('boss_result'):
            print(f"Boss Agent Status: {result['boss_result'].get('session_status')}")
        if result.get('quiz_result'):
            print(f"Quiz Agent Status: {result['quiz_result'].get('status')}")

if __name__ == "__main__":
    main()
