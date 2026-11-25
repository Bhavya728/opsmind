"""
Unit Tests for MCP Tools and Discovery Layer.
Tests FastMCP server, tool schemas, web search, knowledge search, and document lookup.
"""

from app.tools.mcp_server import get_mcp_server
from app.tools.mcp_client import mcp_client
from app.tools.mcp_tools import (
    get_mcp_tool_definitions,
    execute_mcp_tool
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


def test_web_provider_selection():
    """auto picks Tavily only when a key is configured; offline mode always uses the built-in index."""
    from app.config import Settings
    s = Settings()
    s.USE_LIVE_WEB, s.WEB_SEARCH_PROVIDER = True, "auto"
    s.TAVILY_API_KEY = "tvly-test"
    assert s.ACTIVE_WEB_PROVIDER == "tavily"
    s.TAVILY_API_KEY = ""
    assert s.ACTIVE_WEB_PROVIDER == "ddgs"
    s.USE_LIVE_WEB = False
    assert s.ACTIVE_WEB_PROVIDER == "offline_index"


def test_tavily_response_parsing():
    """Tavily results map to title/url/snippet/score, and HTTP errors raise without exposing the key."""
    import httpx
    from app.config import settings
    from app.tools import mcp_server

    class FakeResponse:
        def __init__(self, status, payload):
            self.status_code, self._payload, self.text = status, payload, str(payload)
        def json(self):
            return self._payload

    original_post, original_key = httpx.post, settings.TAVILY_API_KEY
    try:
        settings.TAVILY_API_KEY = "tvly-secret-test-key"
        httpx.post = lambda *a, **k: FakeResponse(200, {"results": [
            {"title": "HNSW", "url": "https://example.com/hnsw", "content": "Graph index.", "score": 0.91234},
            {"title": "no url", "url": "", "content": "dropped"},
        ]})
        results = mcp_server._tavily_search("hnsw", 3)
        assert results == [{"title": "HNSW", "url": "https://example.com/hnsw", "snippet": "Graph index.",
                            "score": 0.912, "source_type": "web"}]

        httpx.post = lambda *a, **k: FakeResponse(401, {"detail": "Unauthorized"})
        try:
            mcp_server._tavily_search("hnsw", 3)
            raise AssertionError("expected an error for HTTP 401")
        except RuntimeError as e:
            assert "401" in str(e) and "tvly-secret-test-key" not in str(e)
    finally:
        httpx.post, settings.TAVILY_API_KEY = original_post, original_key
