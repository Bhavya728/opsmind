"""
Knowledge / RAG Agent Node for OpsMind.
Performs semantic retrieval across ChromaDB internal vector store using HuggingFace / deterministic embeddings.
"""

import time
from typing import Dict, Any, List
from app.agents.state import ResearchState
from app.tools.mcp_tools import execute_mcp_tool
from app.utils.logger import logfire


def rag_knowledge_node(state: ResearchState) -> Dict[str, Any]:
    """
    Retrieves internal technical documents from ChromaDB based on planner search queries.
    """
    start_time = time.time()
    existing_latencies = dict(state.get("latencies", {}))

    if not state.get("need_rag", True):
        logfire.info("📚 [RAG Knowledge Agent] Skipped by Planner decision.")
        existing_latencies["rag"] = 0.0
        return {
            "rag_findings": [],
            "latencies": existing_latencies,
            "current_step": "RAG retrieval skipped"
        }

    search_queries = state.get("search_queries", [state.get("query", "")])
    query = state.get("query", "")
    logfire.info(f"📚 [RAG Knowledge Agent] Querying ChromaDB knowledge base.")
    
    findings: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    evidence_extracted: List[str] = []
    seen_ids = set()

    # Query ChromaDB for main query and sub-queries
    queries_to_run = [query] + [q for q in search_queries if q != query]
    for q in queries_to_run[:2]:
        result = execute_mcp_tool("knowledge_search", {"query": q, "limit": 3})
        tools_used.append("knowledge_search")
        
        if result.get("status") == "success" and result.get("results"):
            for chunk in result["results"]:
                cid = chunk.get("id", chunk.get("doc_id", ""))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    findings.append(chunk)
                    evidence_extracted.append(f"Internal Doc [{chunk.get('title')} - {chunk.get('doc_id')}]: {chunk.get('content')[:200]}")

    elapsed = round(time.time() - start_time, 4)
    existing_latencies["rag"] = elapsed

    return {
        "rag_findings": findings,
        "observed_evidence": evidence_extracted,
        "latencies": existing_latencies,
        "current_step": "RAG retrieval completed",
        "agents_used": ["Knowledge RAG Agent"],
        "tools_used": ["knowledge_search"] if tools_used else [],
        "messages": [{"role": "assistant", "content": f"RAG Knowledge Agent retrieved {len(findings)} relevant knowledge chunks in {elapsed}s."}]
    }
