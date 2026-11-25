"""
Official Model Context Protocol (MCP) Server for OpsMind.
Uses the official Python MCP SDK (FastMCP) to expose research, knowledge retrieval,
and document lookup tools with explicit JSON schemas and standard protocol handlers.
"""

import re
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.services.retrieval.chroma_service import chroma_service, _terms
from app.data.knowledge_base import WEB_KNOWLEDGE_INDEX
from app.utils.logger import logfire

# Instantiate official FastMCP server
mcp_server = FastMCP(
    name="OpsMind-Research-Tools",
    instructions="OpsMind Model Context Protocol server exposing web search, dense vector knowledge retrieval, and document lookup."
)


def _clamp(value: Optional[int], default: int) -> int:
    return min(max(1, value or default), settings.MAX_TOOL_RESULTS)


AD_URL_PATTERN = re.compile(r"/aclick\?|[?&]ad_domain=|duckduckgo\.com/y\.js")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _tavily_search(query: str, max_res: int, recent: bool = False) -> List[Dict[str, Any]]:
    """Search via the Tavily API (requires TAVILY_API_KEY)."""
    import httpx

    if not (settings.TAVILY_API_KEY or "").strip():
        raise RuntimeError("WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set.")
    response = httpx.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY.strip()}"},
        json={"query": query, "max_results": max_res, "search_depth": settings.TAVILY_SEARCH_DEPTH,
              **({"time_range": settings.WEB_RECENT_TIME_RANGE} if recent else {})},
        timeout=settings.WEB_SEARCH_TIMEOUT,
    )
    if response.status_code != 200:
        # Never echo the request (it carries the key); the response body explains the error.
        raise RuntimeError(f"Tavily HTTP {response.status_code}: {response.text[:200]}")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": round(float(r["score"]), 3) if isinstance(r.get("score"), (int, float)) else None,
            "source_type": "web",
        }
        for r in response.json().get("results", [])
        if r.get("url")
    ]


def _ddgs_web_search(query: str, max_res: int, recent: bool = False) -> List[Dict[str, Any]]:
    from ddgs import DDGS

    # ddgs drops some engines when given a comma list, so try each configured engine in order.
    client = DDGS(timeout=settings.WEB_SEARCH_TIMEOUT)
    raw_results: List[Dict[str, Any]] = []
    errors = []
    for engine in [e.strip() for e in settings.WEB_SEARCH_BACKEND.split(",") if e.strip()]:
        try:
            raw_results = client.text(query, max_results=max_res, backend=engine,
                                      timelimit=settings.WEB_RECENT_TIME_RANGE[0] if recent else None) or []
        except Exception as e:
            errors.append(f"{engine}: {e}")
            continue
        if raw_results:
            break
    if not raw_results and errors:
        raise RuntimeError("; ".join(errors))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
            "source_type": "web",
        }
        for r in raw_results
        if r.get("href") and not AD_URL_PATTERN.search(r["href"])
    ]


def _offline_web_search(query: str, max_res: int, recent: bool = False) -> List[Dict[str, Any]]:
    """Search the small built-in web index (offline/test mode only)."""
    q_terms = _terms(query)
    scored = []
    for item in WEB_KNOWLEDGE_INDEX:
        item_terms = _terms(f"{item['title']} {item['snippet']} {' '.join(item['keywords'])}")
        coverage = len(q_terms & item_terms) / len(q_terms) if q_terms else 0.0
        if coverage >= settings.MIN_RELEVANCE_SCORE:
            scored.append({
                "score": round(coverage, 3),
                "title": item["title"],
                "url": item["url"],
                "snippet": item["snippet"],
                "source_type": "web_article",
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_res]


def raw_web_search(query: str, num_results: Optional[int] = None, recent: bool = False) -> Dict[str, Any]:
    """Search the live web (USE_LIVE_WEB=true) or the built-in offline index. Never fabricates results."""
    max_res = _clamp(num_results, settings.WEB_TOP_K)
    backend = settings.ACTIVE_WEB_PROVIDER
    search_fns = {"tavily": _tavily_search, "ddgs": _ddgs_web_search, "offline_index": _offline_web_search}
    try:
        if backend not in search_fns:
            raise ValueError(f"Unknown WEB_SEARCH_PROVIDER '{backend}'; use tavily, ddgs or auto.")
        results = search_fns[backend](query.strip(), max_res, recent)
    except Exception as e:
        logfire.warning(f"🌐 Web search failed ({backend}): {e}")
        return {"status": "error", "tool": "web_search", "backend": backend, "query": query,
                "count": 0, "results": [], "error": str(e)}

    return {
        "status": "success",
        "tool": "web_search",
        "backend": backend,
        "query": query,
        "count": len(results),
        "results": results,
    }


def raw_knowledge_search(query: str, category: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Execute semantic similarity retrieval against ChromaDB knowledge store."""
    hits = chroma_service.query(query_text=query, n_results=_clamp(limit, settings.RAG_TOP_K), category=category)
    return {
        "status": "success",
        "tool": "knowledge_search",
        "query": query,
        "count": len(hits),
        "results": hits,
    }


def raw_document_lookup(doc_id: str) -> Dict[str, Any]:
    """Fetch full structured knowledge document by doc_id."""
    doc_id_clean = doc_id.strip()
    doc = chroma_service.get_document(doc_id_clean)

    if not doc:
        available_ids = [d["id"] for d in chroma_service.list_documents()]
        return {
            "status": "not_found",
            "tool": "document_lookup",
            "doc_id": doc_id_clean,
            "message": f"Document '{doc_id_clean}' not found. Available documents: {available_ids}"
        }

    return {
        "status": "success",
        "tool": "document_lookup",
        "doc_id": doc_id_clean,
        "title": doc.get("title", ""),
        "category": doc.get("category", ""),
        "tags": doc.get("tags", []),
        "summary": doc.get("summary", ""),
        "content": doc.get("content", "")
    }


# Register MCP tools on FastMCP server instance
@mcp_server.tool(name="web_search", description="Search external technical literature, web articles, and benchmarks.")
def mcp_tool_web_search(query: str, num_results: Optional[int] = None, recent: bool = False) -> Dict[str, Any]:
    """MCP Tool: Web Search (recent=True restricts results to WEB_RECENT_TIME_RANGE)"""
    return raw_web_search(query=query, num_results=num_results, recent=recent)


@mcp_server.tool(name="knowledge_search", description="Search internal technical knowledge base using ChromaDB dense semantic vector similarity.")
def mcp_tool_knowledge_search(query: str, category: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """MCP Tool: Knowledge Search"""
    return raw_knowledge_search(query=query, category=category, limit=limit)


@mcp_server.tool(name="document_lookup", description="Retrieve complete knowledge document and structured sections by document ID.")
def mcp_tool_document_lookup(doc_id: str) -> Dict[str, Any]:
    """MCP Tool: Document Lookup"""
    return raw_document_lookup(doc_id=doc_id)


def get_mcp_server() -> FastMCP:
    """Return configured FastMCP server instance."""
    return mcp_server


def list_registered_mcp_tools() -> List[Dict[str, Any]]:
    """Introspect and list all tools exposed by the MCP server with JSON schemas."""
    tools_list = []
    for name, tool_obj in mcp_server._tool_manager._tools.items():
        tools_list.append({
            "name": name,
            "description": tool_obj.description or "",
            "inputSchema": tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
        })
    return tools_list
