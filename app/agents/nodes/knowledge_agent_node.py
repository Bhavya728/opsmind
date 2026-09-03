"""
Knowledge / RAG Agent Node.
Searches Runbooks, SOPs, and Historical Postmortems via Qdrant Vector Retrieval and FlashRank Reranking.
"""

from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.tools.mcp_tools import search_runbook
from app.services.retrieval.qdrant_service import search_incident_knowledge
from app.guardrails.rails import guard_tool_call
from app.utils.logger import logfire


def knowledge_rag_node(state: IncidentState) -> Dict[str, Any]:
    """
    Knowledge RAG Agent:
    - Queries Qdrant vector database and search_runbook MCP tool
    - Retrieves matched runbooks, alert triggers, and past incident postmortems
    - Extracts mitigation steps and SOP recommendations
    """
    query = state.get("incident_query", "")
    service = state.get("service", "order-service")
    tool_args = {"query": query, "service": service, "limit": 3}
    
    with logfire.span("📚 Knowledge & RAG Agent", query=query[:50], service=service):
        is_safe, error_msg = guard_tool_call("search_runbook", tool_args)
        if not is_safe:
            return {
                "knowledge_data": [],
                "observed_evidence": [f"Knowledge tool security alert: {error_msg}"],
                "tools_used": ["search_runbook(BLOCKED)"]
            }
            
        # 1. MCP Tool Runbook Search
        rb_res = search_runbook.invoke(tool_args)
        runbooks = rb_res.get("results", [])
        
        # 2. Qdrant Semantic Retrieval with FlashRank Reranking
        qdrant_docs = search_incident_knowledge(query=query, service=service, limit=3)
        
        extracted_evidence = []
        combined_knowledge = []
        
        for rb in runbooks:
            combined_knowledge.append(rb)
            symptoms_str = ", ".join(rb.get("symptoms", []))
            extracted_evidence.append(f"Matched Runbook {rb.get('id')}: '{rb.get('title')}' (Trigger: {rb.get('alert_trigger')})")
            
        for doc in qdrant_docs:
            combined_knowledge.append(doc)
            extracted_evidence.append(f"Retrieved Knowledge Reference: {doc.get('source')} - {doc.get('title')}")

        return {
            "knowledge_data": combined_knowledge,
            "observed_evidence": extracted_evidence,
            "tools_used": ["search_runbook", "qdrant_retrieval"],
            "messages": [{"role": "knowledge_agent", "content": f"Retrieved {len(runbooks)} runbooks and {len(qdrant_docs)} knowledge documents from Qdrant."}]
        }
