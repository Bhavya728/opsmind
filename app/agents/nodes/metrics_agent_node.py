"""
Metrics Investigation Agent Node.
Executes get_service_metrics MCP tool, analyzes time-series telemetry, and detects metric anomalies.
"""

from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.tools.mcp_tools import get_service_metrics
from app.guardrails.rails import guard_tool_call
from app.utils.logger import logfire


def metrics_investigation_node(state: IncidentState) -> Dict[str, Any]:
    """
    Metrics Agent:
    - Queries MCP tool get_service_metrics for target service
    - Analyzes latency (p95/p99), error rates, CPU/memory, connection saturation
    - Formulates observed telemetry evidence
    """
    service = state.get("service", "order-service")
    tool_args = {"service": service}
    
    with logfire.span("📊 Metrics Investigation Agent", service=service):
        is_safe, error_msg = guard_tool_call("get_service_metrics", tool_args)
        if not is_safe:
            return {
                "metrics_data": {},
                "observed_evidence": [f"Metrics tool security alert: {error_msg}"],
                "tools_used": ["get_service_metrics(BLOCKED)"]
            }
            
        result = get_service_metrics.invoke(tool_args)
        metrics_dict = result.get("metrics", {})
        
        extracted_evidence = []
        for metric_name, values in metrics_dict.items():
            if isinstance(values, list) and values:
                first_val = values[0]
                last_val = values[-1]
                max_val = max(values) if all(isinstance(v, (int, float)) for v in values) else last_val
                
                # Check for latency spike
                if "latency" in metric_name and isinstance(last_val, (int, float)) and last_val > 1000:
                    extracted_evidence.append(f"Metric Anomaly: {metric_name} spiked from {first_val}ms to {last_val}ms (Peak: {max_val}ms)")
                # Check for error rate spike
                elif "error_rate" in metric_name and isinstance(last_val, (int, float)) and last_val > 5.0:
                    extracted_evidence.append(f"Metric Anomaly: {metric_name} elevated to {last_val}%")
                # Check for connection saturation
                elif "active_db_connections" in metric_name:
                    extracted_evidence.append(f"Telemetry: {metric_name} reached {last_val} active slots")
                # Check for CPU saturation
                elif "cpu" in metric_name and isinstance(max_val, (int, float)) and max_val >= 85:
                    extracted_evidence.append(f"Metric Anomaly: CPU utilization reached critical threshold {max_val}%")
                # Check for memory saturation
                elif "memory" in metric_name and isinstance(max_val, (int, float)) and max_val > 450:
                    extracted_evidence.append(f"Metric Anomaly: Memory usage peaked at {max_val}MB")

        if not extracted_evidence and metrics_dict:
            extracted_evidence.append(f"Telemetry snapshot for {service}: {list(metrics_dict.keys())} within standard parameters.")

        return {
            "metrics_data": metrics_dict,
            "observed_evidence": extracted_evidence,
            "tools_used": ["get_service_metrics"],
            "messages": [{"role": "metrics_agent", "content": f"Analyzed telemetry series for {service}; detected {len(extracted_evidence)} metric anomalies."}]
        }
