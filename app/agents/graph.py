"""
LangGraph Multi-Agent Incident Investigation Workflow.

Orchestration Architecture:
- Entry: Supervisor / Planner Node (determines target service & investigation paths)
- Parallel/Conditional Investigation Nodes:
  * Log Investigation Agent (get_service_logs MCP tool)
  * Metrics Investigation Agent (get_service_metrics MCP tool)
  * Deployment Investigation Agent (get_deployment_history MCP tool)
  * Knowledge / RAG Agent (search_runbook & Qdrant semantic search)
- Correlation & Synthesis: Root Cause Analysis (RCA) Agent
- Final Output: Responder Agent (formats structured response & output guardrails)
"""

import os
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import IncidentState
from app.agents.nodes.supervisor_node import supervisor_planner_node
from app.agents.nodes.log_agent_node import log_investigation_node
from app.agents.nodes.metrics_agent_node import metrics_investigation_node
from app.agents.nodes.deployment_agent_node import deployment_agent_node
from app.agents.nodes.knowledge_agent_node import knowledge_rag_node
from app.agents.nodes.rca_agent_node import root_cause_analysis_node
from app.agents.nodes.responder_node import responder_node
from app.utils.logger import logfire


# --- Define LangGraph StateGraph ---
builder = StateGraph(IncidentState)

# Add all agent nodes
builder.add_node("supervisor", supervisor_planner_node)
builder.add_node("log_agent", log_investigation_node)
builder.add_node("metrics_agent", metrics_investigation_node)
builder.add_node("deployment_agent", deployment_agent_node)
builder.add_node("knowledge_agent", knowledge_rag_node)
builder.add_node("rca_agent", root_cause_analysis_node)
builder.add_node("responder", responder_node)

# Set entry point
builder.set_entry_point("supervisor")

# Routing logic from supervisor
def route_after_supervisor(state: IncidentState) -> str:
    paths = state.get("investigation_paths", [])
    if "logs" in paths:
        return "log_agent"
    elif "metrics" in paths:
        return "metrics_agent"
    elif "deployments" in paths:
        return "deployment_agent"
    elif "knowledge" in paths:
        return "knowledge_agent"
    return "rca_agent"


def route_after_logs(state: IncidentState) -> str:
    paths = state.get("investigation_paths", [])
    if "metrics" in paths:
        return "metrics_agent"
    elif "deployments" in paths:
        return "deployment_agent"
    elif "knowledge" in paths:
        return "knowledge_agent"
    return "rca_agent"


def route_after_metrics(state: IncidentState) -> str:
    paths = state.get("investigation_paths", [])
    if "deployments" in paths:
        return "deployment_agent"
    elif "knowledge" in paths:
        return "knowledge_agent"
    return "rca_agent"


def route_after_deployments(state: IncidentState) -> str:
    paths = state.get("investigation_paths", [])
    if "knowledge" in paths:
        return "knowledge_agent"
    return "rca_agent"


builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "log_agent": "log_agent",
        "metrics_agent": "metrics_agent",
        "deployment_agent": "deployment_agent",
        "knowledge_agent": "knowledge_agent",
        "rca_agent": "rca_agent"
    }
)

builder.add_conditional_edges(
    "log_agent",
    route_after_logs,
    {
        "metrics_agent": "metrics_agent",
        "deployment_agent": "deployment_agent",
        "knowledge_agent": "knowledge_agent",
        "rca_agent": "rca_agent"
    }
)

builder.add_conditional_edges(
    "metrics_agent",
    route_after_metrics,
    {
        "deployment_agent": "deployment_agent",
        "knowledge_agent": "knowledge_agent",
        "rca_agent": "rca_agent"
    }
)

builder.add_conditional_edges(
    "deployment_agent",
    route_after_deployments,
    {
        "knowledge_agent": "knowledge_agent",
        "rca_agent": "rca_agent"
    }
)

builder.add_edge("knowledge_agent", "rca_agent")
builder.add_edge("rca_agent", "responder")
builder.add_edge("responder", END)


# --- Build Checkpointer ---
def _build_checkpointer():
    local_mode = os.getenv("LOCAL_MODE", "true").lower() == "true"
    if local_mode:
        logfire.info("🧠 LangGraph Checkpointer: MemorySaver (LOCAL_MODE=true)")
        return MemorySaver()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from app.services.gcp.database_service import get_db_pool
        pool = get_db_pool()
        if pool is not None:
            saver = PostgresSaver(pool)
            saver.setup()
            return saver
    except Exception as e:
        logfire.warning(f"⚠️ PostgresSaver init failed: {e}. Using MemorySaver.")
    return MemorySaver()


checkpointer = _build_checkpointer()
incident_agent = builder.compile(checkpointer=checkpointer)

# Backward-compatibility alias
rag_agent = incident_agent
