"""
LangGraph Agent State definition for OpsMind Research & Knowledge System.
Tracks research goals, planning decisions, retrieved RAG chunks, web findings,
evidence verification, confidence, citations, latencies, and generated reports.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class ResearchState(TypedDict):
    # Core query & session metadata
    query: str
    thread_id: Optional[str]
    
    # Execution & Routing Plan
    plan: List[str]
    need_rag: bool
    need_web: bool
    search_queries: List[str]
    focus_areas: List[str]
    current_step: str
    status: str
    
    # Research findings from RAG & Web
    rag_findings: List[Dict[str, Any]]
    web_findings: List[Dict[str, Any]]
    
    # Evidence Evaluation & Verification
    observed_evidence: Annotated[List[str], operator.add]
    verified_sources: List[Dict[str, Any]]
    critic_score: float
    confidence: float
    critic_notes: str
    
    # Report Output
    report: str
    
    # Timing & Performance Tracking (per-node & total)
    latencies: Dict[str, float]
    
    # Audit telemetry
    agents_used: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    
    # Guardrail and Message History
    guardrail_blocked: bool
    guardrail_response: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # Final Output payload
    final_output: Dict[str, Any]


# Aliases for backward compatibility
AgentState = ResearchState
IncidentState = ResearchState
