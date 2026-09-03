"""
Model Context Protocol (MCP) Client for OpsMind.
Connects to the OpsMind FastMCP server to discover tool schemas and invoke tools
over standard MCP protocol semantics.
"""

from typing import Dict, List, Any, Optional
from app.config import settings
from app.utils.logger import logfire


class OpsMindMCPClient:
    """
    Client interface for OpsMind MCP Server.
    Supports tool discovery, JSON schema introspection, and tool execution.
    """
    def __init__(self, mcp_server_instance=None):
        if mcp_server_instance is None:
            from app.tools.mcp_server import get_mcp_server
            self._server = get_mcp_server()
        else:
            self._server = mcp_server_instance

    def list_tools(self) -> List[Dict[str, Any]]:
        """List discoverable MCP tools with their JSON Schema specifications."""
        tools = []
        if hasattr(self._server, "_tool_manager") and hasattr(self._server._tool_manager, "_tools"):
            for name, tool_obj in self._server._tool_manager._tools.items():
                tools.append({
                    "name": name,
                    "description": tool_obj.description or "",
                    "inputSchema": getattr(tool_obj, "parameters", {})
                })
        return tools

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve schema for a specific MCP tool."""
        tools = self.list_tools()
        for t in tools:
            if t["name"] == tool_name:
                return t
        return None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke an MCP tool through the MCP protocol / tool manager.
        """
        # Guardrail check before invocation
        from app.guardrails.rails import guard_tool_call
        is_valid, err_msg = guard_tool_call(tool_name, arguments)
        if not is_valid:
            return {
                "status": "blocked",
                "tool": tool_name,
                "error": err_msg
            }

        # Check tool registration in FastMCP
        if hasattr(self._server, "_tool_manager") and tool_name in self._server._tool_manager._tools:
            tool_obj = self._server._tool_manager._tools[tool_name]
            try:
                # FastMCP tool functions can be called with arguments
                fn = tool_obj.fn
                result = fn(**arguments)
                return result
            except Exception as e:
                logfire.error(f"❌ Error in MCP Client invoking '{tool_name}': {e}")
                return {
                    "status": "error",
                    "tool": tool_name,
                    "error": str(e)
                }

        return {
            "status": "error",
            "tool": tool_name,
            "error": f"Tool '{tool_name}' not registered in MCP Server."
        }


# Global singleton MCP client
mcp_client = OpsMindMCPClient()
