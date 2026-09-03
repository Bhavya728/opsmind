"""
MCP Tools Implementation for OpsMind Research & Knowledge System.
Provides explicit tool execution, typed argument validation, MCP protocol handler,
and LangChain tool interfaces for multi-agent workflows.
Integrates with the official Python MCP SDK (FastMCP).
"""

from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

from app.config import settings
from app.tools.schemas import (
    WebSearchInput,
    KnowledgeSearchInput,
    DocumentLookupInput,
    MCP_TOOL_DEFINITIONS,
)
from app.tools.mcp_server import raw_web_search, raw_knowledge_search, raw_document_lookup, list_registered_mcp_tools
from app.tools.mcp_client import mcp_client
from app.utils.logger import logfire


@tool(args_schema=WebSearchInput)
def web_search(query: str, num_results: Optional[int] = 3) -> Dict[str, Any]:
    """
    Search the web for recent external technical articles, benchmarks, framework docs, and updates.
    Routes through MCP Client / Server when MCP mode is enabled.
    """
    if settings.USE_MCP_SERVER:
        return mcp_client.call_tool("web_search", {"query": query, "num_results": num_results or 3})
    return raw_web_search(query=query, num_results=num_results or 3)


@tool(args_schema=KnowledgeSearchInput)
def knowledge_search(query: str, category: Optional[str] = None, limit: Optional[int] = 3) -> Dict[str, Any]:
    """
    Search the internal technical knowledge base using ChromaDB vector similarity.
    Retrieves dense semantic chunks with metadata and source citations.
    Routes through MCP Client / Server when MCP mode is enabled.
    """
    args = {"query": query, "limit": limit or 3}
    if category:
        args["category"] = category
    if settings.USE_MCP_SERVER:
        return mcp_client.call_tool("knowledge_search", args)
    return raw_knowledge_search(query=query, category=category, limit=limit or 3)


@tool(args_schema=DocumentLookupInput)
def document_lookup(doc_id: str) -> Dict[str, Any]:
    """
    Fetch the complete internal knowledge document and structured sections by document ID.
    Routes through MCP Client / Server when MCP mode is enabled.
    """
    if settings.USE_MCP_SERVER:
        return mcp_client.call_tool("document_lookup", {"doc_id": doc_id})
    return raw_document_lookup(doc_id=doc_id)


ALL_MCP_TOOLS = [
    web_search,
    knowledge_search,
    document_lookup
]

TOOL_MAP = {
    "web_search": web_search,
    "knowledge_search": knowledge_search,
    "document_lookup": document_lookup
}


def get_mcp_tool_definitions() -> List[Dict[str, Any]]:
    """Return tool schemas conforming to Model Context Protocol specification."""
    # List directly from FastMCP tool registry or defined schemas
    try:
        mcp_tools = mcp_client.list_tools()
        if mcp_tools:
            return mcp_tools
    except Exception as e:
        logfire.warning(f"⚠️ Note reading FastMCP tool registry: {e}")
    return MCP_TOOL_DEFINITIONS


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP Server Tool Invocation Handler with Guardrail validation.
    Executes tool with validated arguments through MCP Client.
    """
    return mcp_client.call_tool(tool_name=tool_name, arguments=arguments)
