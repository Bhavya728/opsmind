"""
Log Investigation Agent Node.
Executes get_service_logs MCP tool, analyzes error logs & stack traces, and extracts structured evidence.
"""

from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.tools.mcp_tools import get_service_logs
from app.guardrails.rails import guard_tool_call
from app.utils.logger import logfire


def log_investigation_node(state: IncidentState) -> Dict[str, Any]:
    """
    Log Agent:
    - Queries MCP tool get_service_logs for target service
    - Applies tool guardrails
    - Extracts observed evidence from stack traces & error messages
    """
    service = state.get("service", "order-service")
    tool_args = {"service": service, "severity": "WARN", "limit": 20}
    
    with logfire.span("📋 Log Investigation Agent", service=service):
        is_safe, error_msg = guard_tool_call("get_service_logs", tool_args)
        if not is_safe:
            return {
                "logs_data": [],
                "observed_evidence": [f"Log tool security alert: {error_msg}"],
                "tools_used": ["get_service_logs(BLOCKED)"]
            }
            
        result = get_service_logs.invoke(tool_args)
        logs = result.get("logs", [])
        
        extracted_evidence = []
        for log in logs:
            level = log.get("level", "INFO")
            msg = log.get("message", "")
            trace = log.get("trace_id", "")
            stack = log.get("stack_trace", "")
            
            if level in ("ERROR", "FATAL", "WARN"):
                summary = f"[{level}] {service}: {msg}"
                if stack:
                    # include top line of stack trace
                    top_stack = stack.split("\n")[0]
                    summary += f" (Trace: {top_stack})"
                extracted_evidence.append(summary)

        if not extracted_evidence and logs:
            extracted_evidence.append(f"Logs retrieved for {service} ({len(logs)} entries) show normal operation.")

        return {
            "logs_data": logs,
            "observed_evidence": extracted_evidence,
            "tools_used": ["get_service_logs"],
            "messages": [{"role": "log_agent", "content": f"Analyzed {len(logs)} log records for {service}; extracted {len(extracted_evidence)} error observations."}]
        }
