"""
LangGraph Agent State definition for the Incident Investigation System.
Tracks multi-agent context, telemetry, observed evidence, root cause hypothesis, confidence, and actions.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class IncidentState(TypedDict):
    # Core request metadata
    incident_query: str
    service: str
    thread_id: Optional[str]
    
    # Execution & Routing Plan
    plan: List[str]
    investigation_paths: List[str]  # e.g. ["logs", "metrics", "deployments", "knowledge"]
    current_step: str
    status: str
    
    # Telemetry and Investigation Findings
    logs_data: List[Dict[str, Any]]
    metrics_data: Dict[str, Any]
    deployments_data: List[Dict[str, Any]]
    knowledge_data: List[Dict[str, Any]]
    
    # Extracted Evidence & Synthesis
    observed_evidence: Annotated[List[str], operator.add]
    inferred_cause: str
    confidence: float
    recommended_actions: List[str]
    tools_used: Annotated[List[str], operator.add]
    
    # Guardrail and Message History
    guardrail_blocked: bool
    guardrail_response: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # Final Output payload
    final_output: Dict[str, Any]


# Backward compatibility alias
AgentState = IncidentState
