"""
Deployment Investigation Agent Node.
Executes get_deployment_history MCP tool to correlate recent releases, configuration pushes, and code commits.
"""

from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.tools.mcp_tools import get_deployment_history
from app.guardrails.rails import guard_tool_call
from app.utils.logger import logfire


def deployment_agent_node(state: IncidentState) -> Dict[str, Any]:
    """
    Deployment Agent:
    - Queries MCP tool get_deployment_history for target service
    - Extracts recent release versions, changelog notes, and deployment statuses
    - Correlates releases with symptoms
    """
    service = state.get("service", "order-service")
    tool_args = {"service": service, "limit": 3}
    
    with logfire.span("🚀 Deployment Investigation Agent", service=service):
        is_safe, error_msg = guard_tool_call("get_deployment_history", tool_args)
        if not is_safe:
            return {
                "deployments_data": [],
                "observed_evidence": [f"Deployment tool security alert: {error_msg}"],
                "tools_used": ["get_deployment_history(BLOCKED)"]
            }
            
        result = get_deployment_history.invoke(tool_args)
        deployments = result.get("deployments", [])
        
        extracted_evidence = []
        for dep in deployments:
            v = dep.get("version", "v1.0")
            ts = dep.get("timestamp", "")
            author = dep.get("author", "unknown")
            changelog = dep.get("changelog", "")
            status = dep.get("status", "SUCCESS")
            
            summary = f"Deployment {v} ({status}) at {ts} by {author}: '{changelog}'"
            extracted_evidence.append(summary)

        return {
            "deployments_data": deployments,
            "observed_evidence": extracted_evidence,
            "tools_used": ["get_deployment_history"],
            "messages": [{"role": "deployment_agent", "content": f"Retrieved {len(deployments)} recent deployments for {service}."}]
        }
