"""
MCP Tools Implementation for Incident Investigation.
Provides robust functions with typed validation, descriptive errors, and LangChain tool wrappers.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

from app.tools.schemas import (
    GetServiceLogsInput,
    GetServiceMetricsInput,
    GetDeploymentHistoryInput,
    SearchRunbookInput,
    VALID_SERVICES,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")


def _load_json(filename: str, default: Any = None) -> Any:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        from app.data.incidents_catalog import init_storage
        init_storage()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return default if default is not None else []


@tool(args_schema=GetServiceLogsInput)
def get_service_logs(
    service: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    severity: Optional[str] = "ALL",
    limit: Optional[int] = 50
) -> Dict[str, Any]:
    """
    Retrieve structured log entries for a target microservice.
    Filters by severity (INFO, WARN, ERROR, FATAL, ALL) and returns relevant log lines with stack traces.
    """
    service_clean = service.strip().lower()
    severity_clean = (severity or "ALL").strip().upper()
    
    all_logs: List[Dict[str, Any]] = _load_json("logs.json", [])
    
    # Filter by service (fuzzy match if prefix matches)
    matching_logs = [
        log for log in all_logs
        if log.get("service", "").lower() == service_clean or service_clean in log.get("service", "").lower()
    ]
    
    if not matching_logs:
        # Check if service is known
        available = list({log.get("service") for log in all_logs if log.get("service")})
        return {
            "status": "warning",
            "service": service_clean,
            "count": 0,
            "message": f"No logs found for service '{service_clean}'. Available services: {available}",
            "logs": []
        }
        
    if severity_clean != "ALL":
        severity_order = {"DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4, "FATAL": 5}
        target_val = severity_order.get(severity_clean, 0)
        filtered = [
            log for log in matching_logs
            if severity_order.get(log.get("level", "").upper(), 0) >= target_val
        ]
    else:
        filtered = matching_logs
        
    limit_val = min(limit or 50, 100)
    result_logs = filtered[-limit_val:]
    
    return {
        "status": "success",
        "service": service_clean,
        "count": len(result_logs),
        "total_matched": len(filtered),
        "logs": result_logs
    }


@tool(args_schema=GetServiceMetricsInput)
def get_service_metrics(
    service: str,
    metric_name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve telemetry metrics (CPU, Memory, Latency p95/p99, Error Rates, Connection Pools) for a service.
    """
    service_clean = service.strip().lower()
    all_metrics: Dict[str, Any] = _load_json("metrics.json", {})
    
    matched_svc = None
    for k in all_metrics.keys():
        if k.lower() == service_clean or service_clean in k.lower():
            matched_svc = k
            break
            
    if not matched_svc:
        return {
            "status": "warning",
            "service": service_clean,
            "message": f"No telemetry metrics registered for service '{service_clean}'. Available: {list(all_metrics.keys())}",
            "metrics": {}
        }
        
    entries = all_metrics[matched_svc]
    if not entries:
        return {
            "status": "warning",
            "service": service_clean,
            "message": "Empty metric series for service",
            "metrics": {}
        }
        
    # Aggregate all metric keys across scenario entries for this service
    aggregated_metrics = {}
    for entry in entries:
        for k, v in entry.get("metrics", {}).items():
            aggregated_metrics[k] = v

    if metric_name:
        clean_name = metric_name.strip().lower()
        filtered_metrics = {
            k: v for k, v in aggregated_metrics.items()
            if clean_name in k.lower()
        }
        return {
            "status": "success",
            "service": matched_svc,
            "metrics": filtered_metrics
        }
        
    return {
        "status": "success",
        "service": matched_svc,
        "metrics": aggregated_metrics
    }


@tool(args_schema=GetDeploymentHistoryInput)
def get_deployment_history(
    service: str,
    limit: Optional[int] = 5
) -> Dict[str, Any]:
    """
    Retrieve recent deployment releases, commit changelogs, authors, and rollback statuses for a service.
    """
    service_clean = service.strip().lower()
    all_deps: Dict[str, Any] = _load_json("deployments.json", {})
    
    matched_svc = None
    for k in all_deps.keys():
        if k.lower() == service_clean or service_clean in k.lower():
            matched_svc = k
            break
            
    if not matched_svc:
        return {
            "status": "warning",
            "service": service_clean,
            "message": f"No deployment history found for service '{service_clean}'. Available: {list(all_deps.keys())}",
            "deployments": []
        }
        
    deps = all_deps[matched_svc]
    limit_val = min(limit or 5, 10)
    recent_deps = deps[:limit_val]
    
    return {
        "status": "success",
        "service": matched_svc,
        "count": len(recent_deps),
        "deployments": recent_deps
    }


@tool(args_schema=SearchRunbookInput)
def search_runbook(
    query: str,
    service: Optional[str] = None,
    limit: Optional[int] = 3
) -> Dict[str, Any]:
    """
    Search engineering runbooks, SOPs, and troubleshooting guides for matching symptoms, alert triggers, or mitigation steps.
    """
    query_clean = query.strip().lower()
    service_clean = (service or "").strip().lower()
    
    runbooks: List[Dict[str, Any]] = _load_json("runbooks.json", [])
    
    scored_results = []
    keywords = set(re.findall(r"\w+", query_clean))
    
    for rb in runbooks:
        score = 0.0
        # Service match bonus
        if service_clean and (service_clean in rb.get("service", "").lower() or rb.get("service", "").lower() in service_clean):
            score += 3.0
            
        rb_text = f"{rb.get('title', '')} {rb.get('alert_trigger', '')} {' '.join(rb.get('symptoms', []))} {' '.join(rb.get('investigation_steps', []))} {' '.join(rb.get('mitigation_steps', []))}".lower()
        
        for kw in keywords:
            if kw in rb_text:
                score += 1.0
                
        if score > 0.5:
            scored_results.append({
                "score": round(score, 2),
                "id": rb.get("id"),
                "service": rb.get("service"),
                "title": rb.get("title"),
                "alert_trigger": rb.get("alert_trigger"),
                "symptoms": rb.get("symptoms"),
                "investigation_steps": rb.get("investigation_steps"),
                "mitigation_steps": rb.get("mitigation_steps"),
                "escalation_path": rb.get("escalation_path")
            })
            
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    limit_val = min(limit or 3, 5)
    top_results = scored_results[:limit_val]
    
    return {
        "status": "success",
        "query": query,
        "count": len(top_results),
        "results": top_results
    }


ALL_MCP_TOOLS = [
    get_service_logs,
    get_service_metrics,
    get_deployment_history,
    search_runbook
]
