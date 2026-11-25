"""
Knowledge / RAG Agent Node for OpsMind.
Performs semantic retrieval across ChromaDB internal vector store using HuggingFace / deterministic embeddings.
"""

import time
from typing import Dict, Any, List
from app.agents.state import ResearchState
from app.config import settings
from app.tools.mcp_tools import execute_mcp_tool
from app.utils.logger import logfire


def rag_knowledge_node(state: ResearchState) -> Dict[str, Any]:
    """
    Retrieves internal knowledge chunks for the pending queries: the original query plus the planner's
    sub-queries in round 1, or the critic's follow-up queries in later rounds. Earlier findings are kept.
    Only chunks above MIN_RELEVANCE_SCORE are returned by the retrieval service.
    Only invoked when the Planner routes the query to the knowledge base.
    """
    start_time = time.time()
    query = state.get("query", "")
    research_round = state.get("research_round", 1)
    search_queries = state.get("search_queries") or []
    logfire.info(f"📚 [RAG Knowledge Agent] Querying ChromaDB knowledge base (round {research_round}).")

    previous = list(state.get("rag_findings") or [])
    best_by_chunk: Dict[str, Dict[str, Any]] = {c.get("id") or c.get("doc_id"): c for c in previous}
    warnings: List[str] = []
    pending = ([query] if research_round == 1 else []) + [sq for sq in search_queries if sq != query]

    for q in pending:
        result = execute_mcp_tool("knowledge_search", {"query": q, "limit": settings.RAG_TOP_K})
        if result.get("status") != "success":
            warnings.append(f"Knowledge search failed for '{q}': {result.get('error', result.get('status'))}")
            continue
        for chunk in result.get("results", []):
            cid = chunk.get("id") or chunk.get("doc_id")
            if cid not in best_by_chunk or chunk["score"] > best_by_chunk[cid]["score"]:
                best_by_chunk[cid] = chunk

    previous_ids = {c.get("id") for c in previous}
    new_chunks = sorted((c for k, c in best_by_chunk.items() if c.get("id") not in previous_ids),
                        key=lambda c: c["score"], reverse=True)
    # Earlier rounds' findings stay first so their citation ids and verdicts remain stable.
    findings = previous + new_chunks[:max(0, settings.MAX_CONTEXT_SOURCES - len(previous))]
    evidence_extracted = [
        f"Internal Doc [{c.get('title')} - {c.get('doc_id')}]: {c.get('content', '')[:200]}" for c in findings[len(previous):]
    ]
    elapsed = round(time.time() - start_time, 4)

    return {
        "rag_findings": findings,
        "observed_evidence": evidence_extracted,
        "latencies": {"rag" if research_round == 1 else f"rag_r{research_round}": elapsed},
        "current_step": "RAG retrieval completed",
        "agents_used": ["Knowledge RAG Agent"],
        "tools_used": ["knowledge_search"],
        "warnings": warnings,
        "messages": [{"role": "assistant", "content": f"RAG Knowledge Agent retrieved {len(findings) - len(previous)} new knowledge chunks (round {research_round}) in {elapsed}s."}]
    }
