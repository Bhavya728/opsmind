"""
OpsMind — Agentic Research & Knowledge System FastAPI Backend.

Provides:
- POST /research (Multi-agent research report generation with latency tracking)
- GET /status & GET /health (Runtime diagnostics, LLM backend, embedding backend, MCP status)
- GET /mcp/tools (MCP Tool discovery)
- POST /mcp/tools/execute (MCP Tool invocation)
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.config import settings
from app.agents.graph import investigate_query
from app.tools.mcp_tools import get_mcp_tool_definitions, execute_mcp_tool
from app.services.retrieval.chroma_service import chroma_service
from app.gateway.client import get_llm_runtime_status
from app.utils.logger import logfire


app = FastAPI(
    title="OpsMind — Agentic Research & Knowledge System API",
    description="Multi-agent AI research assistant with ChromaDB RAG, MCP tool layer, LLM Gateway, and runtime Guardrails.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )


# Request & Response Schemas
class ResearchRequest(BaseModel):
    query: str = Field(..., example="What are the major approaches to RAG evaluation?")
    thread_id: Optional[str] = Field(None, example="thread_abc123")


class ResearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    agents_used: List[str]
    tools_used: List[str]
    confidence: float
    plan: Optional[List[str]] = None
    latencies: Optional[Dict[str, float]] = None
    guardrail_blocked: bool = False


class MCPExecuteRequest(BaseModel):
    tool_name: str = Field(..., example="knowledge_search")
    arguments: Dict[str, Any] = Field(..., example={"query": "RAG evaluation metrics", "limit": 3})


@app.get("/")
def root():
    return {
        "system": "OpsMind — Agentic Research & Knowledge System",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "status_url": "/status"
    }


@app.get("/status")
def status_diagnostics():
    """
    Runtime diagnostics endpoint reporting exact active backends and subsystem statuses.
    Essential for interview verification and environment introspection.
    """
    chroma_status = chroma_service.get_status()
    llm_backend = get_llm_runtime_status()
    mode = "live" if (settings.USE_LIVE_LLM and not settings.LOCAL_MODE) else "offline"
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "mode": mode,
        "llm_backend": llm_backend,
        "embedding_backend": chroma_status.get("embedding_backend", "deterministic"),
        "mcp_enabled": settings.USE_MCP_SERVER,
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "guardrails_enabled": settings.GUARDRAILS_ENABLED,
        "vector_store": {
            "collection": chroma_status.get("collection_name", "opsmind_research_kb"),
            "document_chunks": chroma_status.get("document_count", 0),
            "state": chroma_status.get("status", "connected")
        }
    }


@app.get("/health")
def health_check():
    """System health check endpoint."""
    return status_diagnostics()


@app.post("/research", response_model=ResearchResponse)
def research_endpoint(req: ResearchRequest):
    """
    Main research endpoint:
    Processes question through Input Guardrails, 5-Agent LangGraph workflow,
    ChromaDB RAG, Web Research, Evidence Critique, and Report Generation.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        start_time = time.time()
        result = investigate_query(query=req.query, thread_id=req.thread_id)
        latency = round(time.time() - start_time, 3)
        logfire.info(f"⚡ Research completed in {latency}s for query: '{req.query[:40]}'")
        
        return ResearchResponse(
            query=result.get("query", req.query),
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            agents_used=result.get("agents_used", []),
            tools_used=result.get("tools_used", []),
            confidence=result.get("confidence", 0.0),
            plan=result.get("plan", []),
            latencies=result.get("latencies", {}),
            guardrail_blocked=result.get("guardrail_blocked", False)
        )
    except Exception as e:
        logfire.error(f"❌ Research endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
def list_mcp_tools():
    """Returns tool schemas according to the Model Context Protocol specification."""
    return {
        "tools": get_mcp_tool_definitions()
    }


@app.post("/mcp/tools/execute")
def execute_mcp_tool_endpoint(req: MCPExecuteRequest):
    """Executes an MCP tool with runtime parameter guardrail validation."""
    result = execute_mcp_tool(tool_name=req.tool_name, arguments=req.arguments)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
