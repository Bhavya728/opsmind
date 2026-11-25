"""
LangGraph Agent State definition for OpsMind Research & Knowledge System.
Tracks research goals, planning decisions, retrieved RAG chunks, web findings,
evidence verification, confidence, citations, latencies, and generated reports.

Web research and RAG retrieval run in parallel, so every field both branches write
declares a reducer that merges their updates.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**(left or {}), **(right or {})}


def last_value(left: Any, right: Any) -> Any:
    return right


def sum_dicts(left: Optional[Dict[str, int]], right: Optional[Dict[str, int]]) -> Dict[str, int]:
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = merged.get(k, 0) + v
    return merged


class ResearchState(TypedDict):
    # Core query & session metadata
    query: str
    thread_id: Optional[str]

    # Execution & Routing Plan
    plan: List[str]
    in_scope: bool
    scope_reason: str
    need_rag: bool
    need_web: bool
    needs_recent: bool
    search_queries: List[str]
    focus_areas: List[str]
    current_step: Annotated[str, last_value]
    status: str

    # Research findings from RAG & Web
    rag_findings: List[Dict[str, Any]]
    web_findings: List[Dict[str, Any]]

    # Evidence Evaluation & Verification
    research_round: int
    observed_evidence: Annotated[List[str], operator.add]
    source_verdicts: Annotated[Dict[str, bool], merge_dicts]   # source key -> accepted, across rounds
    verified_sources: List[Dict[str, Any]]
    rejected_sources: List[Dict[str, Any]]
    followup_queries: List[str]
    critic_score: float
    evidence_quality: float
    confidence: float
    critic_notes: str
    evidence_gaps: List[str]

    # Report Output
    report: str
    citations: List[str]
    invalid_citations: List[str]
    revision_count: int
    revision_feedback: str
    revision_pending: bool

    # Claim verification (after the report)
    claims: List[Dict[str, Any]]
    claim_support: Optional[float]

    # Timing, cost & performance tracking (per-node)
    latencies: Annotated[Dict[str, float], merge_dicts]
    token_usage: Annotated[Dict[str, int], sum_dicts]

    # Audit telemetry
    agents_used: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]

    # Guardrail and Message History
    guardrail_blocked: bool
    guardrail_response: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]
