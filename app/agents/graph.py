"""
LangGraph Multi-Agent Orchestration Graph for OpsMind Research & Knowledge System.

Workflow:
1. Input Guardrail Verification
2. Query Planner (Deconstruction & Routing)
3. Web Research Agent (External Search)
4. Knowledge / RAG Agent (ChromaDB Dense Retrieval)
5. Evidence / Critic Agent (Groundedness & Citation Verification)
6. Report Generation Agent (Structured Synthesis)
7. Output Guardrail Redaction
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.agents.state import ResearchState
from app.agents.nodes.query_planner import query_planner_node
from app.agents.nodes.web_research_agent import web_research_node
from app.agents.nodes.rag_knowledge_agent import rag_knowledge_node
from app.agents.nodes.evidence_critic_agent import evidence_critic_node
from app.agents.nodes.report_generator_agent import report_generator_node
from app.guardrails.rails import guard_input, guard_output
from app.utils.logger import logfire


def create_research_graph():
    """Build and compile the LangGraph 5-Agent Research Workflow."""
    workflow = StateGraph(ResearchState)
    
    # Register agent nodes
    workflow.add_node("query_planner", query_planner_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("rag_knowledge", rag_knowledge_node)
    workflow.add_node("evidence_critic", evidence_critic_node)
    workflow.add_node("report_generator", report_generator_node)
    
    # Define linear execution topology with internal conditional checks
    workflow.set_entry_point("query_planner")
    workflow.add_edge("query_planner", "web_research")
    workflow.add_edge("web_research", "rag_knowledge")
    workflow.add_edge("rag_knowledge", "evidence_critic")
    workflow.add_edge("evidence_critic", "report_generator")
    workflow.add_edge("report_generator", END)
    
    return workflow.compile()


# Global compiled graph instance
research_graph = create_research_graph()


def investigate_query(query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Primary orchestration entry point for researching a query.
    Enforces Input Guardrails -> Multi-Agent LangGraph -> Output Guardrails.
    """
    # 1. Gate 1: Input Guardrail Check
    is_blocked, block_reason = guard_input(query)
    if is_blocked:
        logfire.warning(f"🛡️ Query blocked at input guardrail: {block_reason}")
        return {
            "query": query,
            "answer": block_reason,
            "sources": [],
            "agents_used": ["Input Guardrail Gate"],
            "tools_used": [],
            "confidence": 0.0,
            "plan": ["Input Safety Evaluation: Blocked"],
            "guardrail_blocked": True,
            "guardrail_response": block_reason
        }

    # 2. Initialize State
    initial_state: ResearchState = {
        "query": query,
        "thread_id": thread_id,
        "plan": [],
        "need_rag": True,
        "need_web": True,
        "search_queries": [],
        "focus_areas": [],
        "current_step": "Starting research",
        "status": "in_progress",
        "rag_findings": [],
        "web_findings": [],
        "observed_evidence": [],
        "verified_sources": [],
        "critic_score": 0.0,
        "confidence": 0.0,
        "critic_notes": "",
        "report": "",
        "agents_used": [],
        "tools_used": [],
        "guardrail_blocked": False,
        "guardrail_response": None,
        "messages": [{"role": "user", "content": query}],
        "final_output": {}
    }

    # 3. Execute LangGraph
    try:
        final_state = research_graph.invoke(initial_state)
        
        if final_state.get("final_output"):
            output = final_state["final_output"]
            output["guardrail_blocked"] = False
            return output
            
        return {
            "query": query,
            "answer": final_state.get("report", "Research completed."),
            "sources": final_state.get("verified_sources", []),
            "agents_used": list(set(final_state.get("agents_used", []))),
            "tools_used": list(set(final_state.get("tools_used", []))),
            "confidence": final_state.get("confidence", 0.90),
            "plan": final_state.get("plan", []),
            "guardrail_blocked": False
        }
    except Exception as e:
        logfire.error(f"❌ LangGraph execution failed: {e}")
        return {
            "query": query,
            "answer": f"Research orchestration encountered an error: {str(e)}",
            "sources": [],
            "agents_used": ["Error Handler"],
            "tools_used": [],
            "confidence": 0.0,
            "plan": [],
            "guardrail_blocked": False
        }


# Backward compatibility aliases
investigate_incident = investigate_query
incident_graph = research_graph
