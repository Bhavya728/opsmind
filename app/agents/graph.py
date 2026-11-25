"""
LangGraph Multi-Agent Orchestration Graph for OpsMind Research & Knowledge System.

Workflow:
1. Input Guardrail Verification (pattern rails, before the graph)
2. Query Planner: scope check, query decomposition, retrieval routing
3. Planner-routed retrieval, in parallel when both are needed:
   - Web Research Agent (external search via MCP)
   - Knowledge / RAG Agent (ChromaDB dense retrieval via MCP)
4. Evidence / Critic Agent (per-source relevance verdicts, evidence quality); may send
   follow-up queries back to retrieval once (MAX_RESEARCH_ROUNDS)
5. Report Generation Agent (synthesis with citation grounding check)
6. Claim Verifier (checks report claims against cited sources, sets final confidence);
   may send the report back for one revision (MAX_REPORT_REVISIONS)

Out-of-scope queries end after the planner with a refusal.
"""

import time
from typing import Dict, Any, Callable, List, Optional
from langgraph.graph import StateGraph, END

from app.agents.state import ResearchState
from app.agents.nodes.query_planner import query_planner_node
from app.agents.nodes.web_research_agent import web_research_node
from app.agents.nodes.rag_knowledge_agent import rag_knowledge_node
from app.agents.nodes.evidence_critic_agent import evidence_critic_node
from app.agents.nodes.report_generator_agent import report_generator_node
from app.agents.nodes.claim_verifier_agent import claim_verifier_node
from app.config import settings
from app.guardrails.rails import guard_input
from app.utils.logger import logfire


def route_after_planning(state: ResearchState) -> List[str]:
    """Fan out to the retrieval agents the planner selected, or stop for out-of-scope queries."""
    if not state.get("in_scope", True):
        return [END]
    routes = []
    if state.get("need_web"):
        routes.append("web_research")
    if state.get("need_rag"):
        routes.append("rag_knowledge")
    return routes or ["evidence_critic"]


def _retrieval_routes(state: ResearchState) -> List[str]:
    return [name for name, needed in (("web_research", state.get("need_web")), ("rag_knowledge", state.get("need_rag"))) if needed]


def route_after_critic(state: ResearchState) -> List[str]:
    """Run another retrieval round when the critic asked for follow-up queries, else write the report."""
    if state.get("followup_queries") and _retrieval_routes(state):
        return _retrieval_routes(state)
    return ["report_generator"]


def route_after_report(state: ResearchState) -> str:
    return "claim_verifier" if settings.CLAIM_VERIFICATION_ENABLED else END


def route_after_verification(state: ResearchState) -> str:
    return "report_generator" if state.get("revision_pending") else END


def create_research_graph():
    """Build and compile the LangGraph 5-Agent Research Workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("query_planner", query_planner_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("rag_knowledge", rag_knowledge_node)
    workflow.add_node("evidence_critic", evidence_critic_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("claim_verifier", claim_verifier_node)

    workflow.set_entry_point("query_planner")
    workflow.add_conditional_edges(
        "query_planner",
        route_after_planning,
        ["web_research", "rag_knowledge", "evidence_critic", END],
    )
    workflow.add_edge("web_research", "evidence_critic")
    workflow.add_edge("rag_knowledge", "evidence_critic")
    workflow.add_conditional_edges("evidence_critic", route_after_critic,
                                   ["web_research", "rag_knowledge", "report_generator"])
    workflow.add_conditional_edges("report_generator", route_after_report, ["claim_verifier", END])
    workflow.add_conditional_edges("claim_verifier", route_after_verification, ["report_generator", END])

    return workflow.compile()


# Global compiled graph instance
research_graph = create_research_graph()


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _blocked_response(query: str, reason: str, gate: str, latencies: Dict[str, float]) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": reason,
        "sources": [],
        "agents_used": [gate],
        "tools_used": [],
        "confidence": 0.0,
        "plan": [],
        "latencies": latencies,
        "guardrail_blocked": True,
        "guardrail_response": reason,
        "citations": [],
        "warnings": [],
    }


# Node-update fields forwarded to progress listeners (enough for a UI to show which agent runs next).
_PROGRESS_FIELDS = ("in_scope", "need_web", "need_rag", "followup_queries", "revision_pending")


def investigate_query(
    query: str,
    thread_id: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Primary orchestration entry point for researching a query.
    Enforces Input Guardrails -> Multi-Agent LangGraph -> Output Guardrails.
    Raises on orchestration or LLM failure so callers can report it instead of returning a fake answer.

    on_event, if given, is called with {"type": "node", "node", "elapsed", "info"} as each step finishes.
    """
    start_total_time = time.time()
    # Normalize whitespace (trailing newlines from the UI text box, tabs, repeated spaces).
    query = " ".join(query.split())
    if len(query) > settings.MAX_QUERY_CHARS:
        raise ValueError(f"Query is {len(query)} characters; the limit is {settings.MAX_QUERY_CHARS} (MAX_QUERY_CHARS).")

    is_blocked, block_reason = guard_input(query)
    if is_blocked:
        logfire.warning(f"🛡️ Query blocked at input guardrail: {block_reason}")
        elapsed = round(time.time() - start_total_time, 4)
        return _blocked_response(query, block_reason, "Input Guardrail Gate", {"guardrail": elapsed, "total": elapsed})

    def emit(node: str, info: Optional[Dict[str, Any]] = None) -> None:
        if on_event:
            on_event({"type": "node", "node": node, "elapsed": round(time.time() - start_total_time, 3), "info": info or {}})

    emit("input_guardrail")

    initial_state: ResearchState = {
        "query": query,
        "thread_id": thread_id,
        "plan": [],
        "in_scope": True,
        "scope_reason": "",
        "need_rag": False,
        "need_web": False,
        "needs_recent": False,
        "search_queries": [],
        "focus_areas": [],
        "current_step": "Starting research",
        "status": "in_progress",
        "rag_findings": [],
        "web_findings": [],
        "research_round": 1,
        "observed_evidence": [],
        "source_verdicts": {},
        "verified_sources": [],
        "rejected_sources": [],
        "followup_queries": [],
        "critic_score": 0.0,
        "evidence_quality": 0.0,
        "confidence": 0.0,
        "critic_notes": "",
        "evidence_gaps": [],
        "report": "",
        "citations": [],
        "invalid_citations": [],
        "revision_count": 0,
        "revision_feedback": "",
        "revision_pending": False,
        "claims": [],
        "claim_support": None,
        "latencies": {},
        "token_usage": {},
        "agents_used": [],
        "tools_used": [],
        "warnings": [],
        "guardrail_blocked": False,
        "guardrail_response": None,
        "messages": [{"role": "user", "content": query}],
    }

    final_state: Dict[str, Any] = initial_state
    for mode, chunk in research_graph.stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = chunk
        else:
            for node, update in chunk.items():
                emit(node, {k: v for k, v in (update or {}).items() if k in _PROGRESS_FIELDS})
    latencies = dict(final_state.get("latencies", {}))
    latencies["total"] = round(time.time() - start_total_time, 4)

    if not final_state.get("in_scope", True):
        reason = final_state.get("scope_reason") or "The question is outside the research scope."
        refusal = (
            f"🛡️ This question is outside OpsMind's research scope ({settings.RESEARCH_SCOPE}). {reason}"
        )
        logfire.info(f"🛡️ Planner topic guard refused query: {reason}")
        response = _blocked_response(query, refusal, "Query Planner (topic guard)", latencies)
        response["agents_used"] = _unique(final_state.get("agents_used", []))
        return response

    sources = []
    seen = set()
    for s in final_state.get("verified_sources", []):
        if s["citation_id"] in seen:
            continue
        seen.add(s["citation_id"])
        sources.append({
            "id": s["citation_id"],
            "title": s["title"],
            "type": s["type"],
            "url": s.get("url"),
            "score": s.get("score"),
            "content": s.get("content", ""),
        })

    return {
        "query": query,
        "answer": final_state.get("report", ""),
        "sources": sources,
        "rejected_sources": list({
            s["citation_id"]: {"id": s["citation_id"], "title": s["title"], "type": s["type"]}
            for s in final_state.get("rejected_sources", [])
            if s["citation_id"] not in seen
        }.values()),
        "citations": final_state.get("citations", []),
        "agents_used": _unique(final_state.get("agents_used", [])),
        "tools_used": _unique(final_state.get("tools_used", [])),
        "confidence": final_state.get("confidence", 0.0),
        "evidence_quality": final_state.get("evidence_quality", 0.0),
        "claim_support": final_state.get("claim_support"),
        "claims": final_state.get("claims", []),
        "critic_notes": final_state.get("critic_notes", ""),
        "evidence_gaps": final_state.get("evidence_gaps", []),
        "research_rounds": final_state.get("research_round", 1),
        "revisions": final_state.get("revision_count", 0),
        "token_usage": {**final_state.get("token_usage", {}), "total": sum(final_state.get("token_usage", {}).values())},
        "plan": final_state.get("plan", []),
        "routing": {
            "knowledge_base": final_state.get("need_rag", False),
            "web": final_state.get("need_web", False),
        },
        "latencies": latencies,
        "warnings": _unique(final_state.get("warnings", [])),
        "guardrail_blocked": False,
    }
