"""
Unit Tests for MCP Tools and Discovery Layer.
Tests FastMCP server, tool schemas, web search, knowledge search, and document lookup.
"""

from app.tools.mcp_server import get_mcp_server, list_registered_mcp_tools
from app.tools.mcp_client import mcp_client
from app.tools.mcp_tools import (
    get_mcp_tool_definitions,
    execute_mcp_tool,
    web_search,
    knowledge_search,
    document_lookup
)


def test_mcp_server_initialization():
    """Test that the official FastMCP server initializes and exposes tools."""
    server = get_mcp_server()
    assert server is not None
    assert server.name == "OpsMind-Research-Tools"


def test_mcp_tool_definitions_schema():
    """Test that MCP discovery returns valid tool definitions with JSON schemas."""
    definitions = get_mcp_tool_definitions()
    assert isinstance(definitions, list)
    assert len(definitions) >= 3
    tool_names = [d["name"] for d in definitions]
    assert "web_search" in tool_names
    assert "knowledge_search" in tool_names
    assert "document_lookup" in tool_names
    for d in definitions:
        assert "inputSchema" in d


def test_mcp_client_tool_schema_lookup():
    """Test MCP client schema lookup for specific tool."""
    schema = mcp_client.get_tool_schema("knowledge_search")
    assert schema is not None
    assert schema["name"] == "knowledge_search"


def test_mcp_execute_knowledge_search():
    """Test executing knowledge search via MCP client."""
    result = execute_mcp_tool("knowledge_search", {"query": "RAG evaluation faithfulness", "limit": 3})
    assert result["status"] == "success"
    assert result["tool"] == "knowledge_search"
    assert len(result["results"]) > 0
    first_hit = result["results"][0]
    assert "content" in first_hit
    assert "score" in first_hit


def test_mcp_execute_web_search():
    """Test executing web search via MCP client."""
    result = execute_mcp_tool("web_search", {"query": "LangGraph multi-agent systems", "num_results": 2})
    assert result["status"] == "success"
    assert result["tool"] == "web_search"
    assert len(result["results"]) > 0
    assert "url" in result["results"][0]


def test_mcp_execute_document_lookup():
    """Test fetching full document by doc_id via MCP client."""
    result = execute_mcp_tool("document_lookup", {"doc_id": "doc-rag-eval-01"})
    assert result["status"] == "success"
    assert result["doc_id"] == "doc-rag-eval-01"
    assert "title" in result
    assert "content" in result


def test_mcp_execute_invalid_tool():
    """Test that invoking an unregistered tool returns a clean error."""
    result = execute_mcp_tool("non_existent_tool", {})
    assert result["status"] == "error"
    assert "not registered" in result["error"]


def test_mcp_execute_guardrail_blocked():
    """Test that MCP execution handler blocks malicious command arguments."""
    result = execute_mcp_tool("knowledge_search", {"query": "eval; rm -rf /"})
    assert result["status"] == "blocked"
    assert "error" in result
