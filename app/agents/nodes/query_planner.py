"""
Query Planner Node for OpsMind Research & Knowledge System.
Checks the query is within the configured research scope, plans the investigation,
and decides which retrieval paths (internal knowledge base, web, or both) to run.
"""

import time
from datetime import date
from typing import Dict, Any, List
from app.agents.state import ResearchState
from app.agents.llm_json import parse_json_object, response_text, token_usage, fallback_note
from app.config import settings
from app.gateway.client import get_llm
from app.services.retrieval.chroma_service import chroma_service
from app.utils.logger import logfire


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return default


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def query_planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Analyzes the user's research query, breaks it down into sub-queries,
    and decides whether to invoke Web Research, RAG Knowledge Search, or both.
    """
    start_time = time.time()
    query = state.get("query", "")
    logfire.info(f"🧠 [Query Planner] Analyzing research request: '{query[:60]}'")

    planner_prompt = f"""You are the Query Planner Agent of OpsMind, a technical research assistant.

Today's date: {date.today().isoformat()}. For questions about the latest or recent developments, aim search queries at current information.
Research scope: {settings.RESEARCH_SCOPE}.
Internal knowledge base documents: {chroma_service.describe_corpus() or "none"}.

Research Query: "{query}"

Decide:
- in_scope: is the query a genuine question within the research scope?
- need_rag: would the internal knowledge base documents above help answer it?
- need_web: does it need external, recent, or broader information than the knowledge base holds?
- needs_recent: does the answer depend on the latest or recent information (current versions, recent releases, news)?
- search_queries: up to {settings.MAX_SEARCH_QUERIES} focused search queries that together cover the question.
- focus_areas and plan: the aspects to cover and the steps to research them.

Respond with only a JSON object:
{{
    "in_scope": true,
    "scope_reason": "one sentence",
    "need_rag": true,
    "need_web": true,
    "needs_recent": false,
    "search_queries": ["..."],
    "focus_areas": ["..."],
    "plan": ["1. ...", "2. ..."]
}}
"""
    llm = get_llm(task_type="planning")
    response = llm.invoke(planner_prompt)
    parsed = parse_json_object(response_text(response))
    warnings: List[str] = [n for n in [fallback_note(response, "Query Planner")] if n]

    if parsed is None:
        warnings.append("Planner returned no valid JSON; searched both sources with the original query.")
        logfire.warning(f"⚠️ [Query Planner] {warnings[-1]}")
        parsed = {}

    in_scope = _as_bool(parsed.get("in_scope"), True) if settings.TOPIC_GUARD_ENABLED else True
    search_queries = _as_str_list(parsed.get("search_queries"))[:settings.MAX_SEARCH_QUERIES] or [query]
    plan = _as_str_list(parsed.get("plan"))
    need_rag = _as_bool(parsed.get("need_rag"), True)
    need_web = _as_bool(parsed.get("need_web"), True)

    elapsed = round(time.time() - start_time, 4)
    routes = [name for name, needed in (("knowledge base", need_rag), ("web", need_web)) if needed]
    summary = (
        f"Query Planner judged the query out of scope in {elapsed}s." if not in_scope else
        f"Query Planner created a {len(plan)}-step plan routed to {', '.join(routes) or 'no retrieval'} in {elapsed}s."
    )

    return {
        "in_scope": in_scope,
        "scope_reason": str(parsed.get("scope_reason", "")),
        "need_rag": need_rag,
        "need_web": need_web,
        "needs_recent": _as_bool(parsed.get("needs_recent"), False),
        "search_queries": search_queries,
        "focus_areas": _as_str_list(parsed.get("focus_areas")),
        "plan": plan,
        "current_step": "Planning complete",
        "research_round": 1,
        "latencies": {"planner": elapsed},
        "token_usage": {"planner": token_usage(response)},
        "agents_used": ["Query Planner"],
        "warnings": warnings,
        "messages": [{"role": "assistant", "content": summary}]
    }
