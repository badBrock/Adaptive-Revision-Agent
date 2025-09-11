from typing import Optional
from .fastmcp_client import LearningSystemMCPClient  # ✅ CORRECT - relative import

class MCPFactory:
    """Factory for FastMCP client"""
    
    _instance: Optional[LearningSystemMCPClient] = None
    
    @classmethod
    async def get_client(cls) -> LearningSystemMCPClient:
        """Get singleton FastMCP client instance"""
        if cls._instance is None:
            cls._instance = LearningSystemMCPClient()
            await cls._instance.connect()
        return cls._instance
    
    @classmethod
    async def reset(cls):
        """Reset client (useful for testing)"""
        if cls._instance:
            await cls._instance.disconnect()
        cls._instance = None
