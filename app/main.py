"""
OpsMind — Agentic Research & Knowledge System FastAPI Backend.

Provides:
- GET / (web UI, served from ui/index.html)
- POST /research (Multi-agent research report generation with latency tracking)
- POST /research/stream (same, as Server-Sent Events with per-agent progress)
- GET /status & GET /health (Runtime diagnostics, LLM backend, embedding backend, MCP status)
- GET /mcp/tools (MCP Tool discovery)
- POST /mcp/tools/execute (MCP Tool invocation)
- GET /evals/results (latest evaluation run, for the UI dashboard)
- GET /knowledge/documents (indexed knowledge-base documents)

Run with: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import json
import os
import queue
import threading
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from app.config import settings
from app.agents.graph import investigate_query
from app.tools.mcp_tools import get_mcp_tool_definitions, execute_mcp_tool
from app.services.retrieval.chroma_service import chroma_service
from app.gateway.client import get_llm_runtime_status
from app.utils.logger import logfire

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_INDEX = os.path.join(PROJECT_ROOT, "ui", "index.html")
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evals", "evaluation_results.json")

app = FastAPI(
    title="OpsMind — Agentic Research & Knowledge System API",
    description="Multi-agent AI research assistant with ChromaDB RAG, MCP tool layer, LLM Gateway, and runtime Guardrails.",
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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
    rejected_sources: List[Dict[str, Any]] = []
    citations: List[str] = []
    agents_used: List[str]
    tools_used: List[str]
    confidence: float
    critic_notes: str = ""
    evidence_quality: Optional[float] = None
    claim_support: Optional[float] = None
    claims: List[Dict[str, Any]] = []
    evidence_gaps: List[str] = []
    research_rounds: int = 1
    revisions: int = 0
    token_usage: Dict[str, int] = {}
    plan: Optional[List[str]] = None
    routing: Optional[Dict[str, bool]] = None
    latencies: Optional[Dict[str, float]] = None
    warnings: List[str] = []
    guardrail_blocked: bool = False


class MCPExecuteRequest(BaseModel):
    tool_name: str = Field(..., example="knowledge_search")
    arguments: Dict[str, Any] = Field(..., example={"query": "RAG evaluation metrics", "limit": 3})


@app.get("/", include_in_schema=False)
def ui():
    """Serves the single-page web UI."""
    return FileResponse(UI_INDEX, media_type="text/html")


@app.get("/info")
def info():
    return {
        "system": "OpsMind — Agentic Research & Knowledge System",
        "version": settings.VERSION,
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
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "mode": settings.MODE,
        "llm_models": {"fast": settings.GROQ_FAST_MODEL, "strong": settings.GROQ_MODEL} if settings.USE_LIVE_LLM else None,
        "web_search": settings.ACTIVE_WEB_PROVIDER,
        "llm_backend": llm_backend,
        "embedding_backend": chroma_status.get("embedding_backend", "deterministic"),
        "mcp_enabled": settings.USE_MCP_SERVER,
        "langsmith_tracing": settings.LANGSMITH_TRACING,
        "guardrails_enabled": settings.GUARDRAILS_ENABLED,
        "vector_store": {
            "collection": chroma_status["collection_name"],
            "documents": chroma_status["document_count"],
            "document_chunks": chroma_status["chunk_count"],
            "state": chroma_status["status"]
        }
    }


@app.get("/health")
def health_check():
    """System health check endpoint."""
    return status_diagnostics()


def _run_research(req: ResearchRequest, on_event=None) -> ResearchResponse:
    start_time = time.time()
    result = investigate_query(query=req.query, thread_id=req.thread_id, on_event=on_event)
    logfire.info(f"⚡ Research completed in {round(time.time() - start_time, 3)}s for query: '{req.query[:40]}'")
    return ResearchResponse(**{k: v for k, v in result.items() if k in ResearchResponse.model_fields})


def _failure_detail(e: Exception) -> str:
    logfire.error(f"❌ Research endpoint error: {type(e).__name__}: {e}")
    if type(e).__name__ == "AuthenticationError":
        return ("The LLM provider rejected the API key. Check GROQ_API_KEY in .env, "
                "then restart the server (settings are read at startup).")
    return f"Research failed: {type(e).__name__}. Check the server log for details."


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
        return _run_research(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=_failure_detail(e))


@app.post("/research/stream")
def research_stream_endpoint(req: ResearchRequest):
    """
    Same as POST /research, streamed as Server-Sent Events:
    {"type": "node", ...} as each agent finishes, then one {"type": "result", "data": ...}
    or {"type": "error", "detail": ...}.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    events: "queue.Queue" = queue.Queue()

    def worker():
        try:
            events.put({"type": "result", "data": _run_research(req, on_event=events.put).model_dump()})
        except ValueError as e:
            events.put({"type": "error", "detail": str(e)})
        except Exception as e:
            events.put({"type": "error", "detail": _failure_detail(e)})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@app.get("/evals/results")
def eval_results():
    """Latest evaluation run written by `python evals/evaluator.py`; null when there is none yet."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        return {"results": None}
    try:
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return {"results": json.load(f)}
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Could not read evaluation results: {e}")


@app.get("/knowledge/documents")
def knowledge_documents():
    """Documents currently indexed in the ChromaDB knowledge base."""
    return {"knowledge_dir": settings.KNOWLEDGE_DIR, "documents": chroma_service.list_documents()}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
