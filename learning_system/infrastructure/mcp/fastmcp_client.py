import asyncio
import logging
from typing import Dict, Any, List
from fastmcp import Client
import os
import json

class LearningSystemMCPClient:
    """FastMCP client wrapper for the learning system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None
        self.connected = False
        
        # Path to our MCP server
        server_path = os.path.join(os.path.dirname(__file__), "learning_mcp_server.py")
        self.server_path = server_path
    
    async def connect(self):
        """Connect to FastMCP server via stdio"""
        try:
            self.logger.info("🔗 Connecting to Learning System FastMCP server...")
            
            # Create client using the correct FastMCP 2.0 API
            # The server path should point to our MCP server script
            self.client = Client(self.server_path)
            
            # Connect using context manager
            await self.client.__aenter__()
            
            self.connected = True
            self.logger.info("✅ Connected to FastMCP server!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to FastMCP server: {e}")
            raise
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool via FastMCP"""
        if not self.connected:
            await self.connect()
        
        self.logger.info(f"🔧 Calling FastMCP tool: {tool_name}")
        
        try:
            # Use the correct FastMCP 2.0 API for calling tools
            result = await self.client.call_tool(tool_name, kwargs)
            
            # Extract content from MCP response
            if result and hasattr(result, 'content') and result.content:
                # Get the first content item (usually TextContent)
                content = result.content[0]
                if hasattr(content, 'text'):
                    # Parse JSON if it's a string
                    try:
                        parsed_result = json.loads(content.text)
                        self.logger.info(f"✅ FastMCP tool '{tool_name}' completed")
                        return parsed_result
                    except json.JSONDecodeError:
                        return {"result": content.text, "status": "success"}
                else:
                    return {"result": str(content), "status": "success"}
            else:
                return {"result": "No content", "status": "success"}
                
        except Exception as e:
            self.logger.error(f"❌ FastMCP tool '{tool_name}' failed: {e}")
            raise
    
    async def list_tools(self) -> List[str]:
        """List available tools from FastMCP server"""
        if not self.connected:
            await self.connect()
        
        try:
            tools_response = await self.client.list_tools()
            return [tool.name for tool in tools_response.tools]
        except Exception as e:
            self.logger.error(f"❌ Failed to list FastMCP tools: {e}")
            return []
    
    async def disconnect(self):
        """Cleanup FastMCP connections"""
        if self.client and self.connected:
            try:
                await self.client.__aexit__(None, None, None)
            except:
                pass
            self.connected = False
            self.logger.info("🔌 Disconnected from FastMCP server")
