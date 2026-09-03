"""
Supervisor / Planner Agent Node.
Analyzes incident symptoms, detects target microservice, and generates the investigation plan.
Determines conditional investigation paths (logs, metrics, deployments, runbooks).
"""

import re
import json
from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.gateway.client import get_llm
from app.tools.schemas import VALID_SERVICES
from app.utils.logger import logfire


def _infer_service_from_query(query: str) -> str:
    """Extracts or infers target microservice from the incident text."""
    q_lower = query.lower()
    for svc in VALID_SERVICES:
        if svc in q_lower:
            return svc
        # Short token matching
        prefix = svc.split("-")[0]
        if prefix in q_lower and prefix not in ["kube", "cert"]:
            return svc
            
    if "postgres" in q_lower or "database" in q_lower or "sql" in q_lower or "wal" in q_lower:
        return "payment-service" if "payment" in q_lower else ("order-service" if "order" in q_lower else "user-service")
    if "jwt" in q_lower or "auth" in q_lower or "token" in q_lower or "login" in q_lower:
        return "auth-service"
    if "payment" in q_lower or "stripe" in q_lower or "checkout" in q_lower:
        return "payment-service"
    if "order" in q_lower or "cart" in q_lower:
        return "order-service"
    if "redis" in q_lower or "stock" in q_lower or "inventory" in q_lower:
        return "inventory-service"
    if "dns" in q_lower or "gateway" in q_lower or "ingress" in q_lower or "429" in q_lower or "503" in q_lower:
        return "api-gateway"
    if "kafka" in q_lower or "rabbit" in q_lower or "notification" in q_lower or "email" in q_lower:
        return "notification-service"
    if "analytics" in q_lower or "s3" in q_lower or "elastic" in q_lower or "report" in q_lower:
        return "analytics-service"

    return "order-service"


def supervisor_planner_node(state: IncidentState) -> Dict[str, Any]:
    """
    Supervisor Node:
    - Identifies target service
    - Formulates investigation plan
    - Configures investigation routing paths
    """
    query = state.get("incident_query", "")
    current_service = state.get("service", "")
    
    if not current_service or current_service == "generic":
        inferred_service = _infer_service_from_query(query)
    else:
        inferred_service = current_service
        
    with logfire.span("🧭 Supervisor Planning", service=inferred_service, query=query[:60]):
        llm = get_llm(task_type="planning")
        
        prompt = (
            f"You are the Supervisor Incident Investigation Agent.\n"
            f"Incident Alert: '{query}'\n"
            f"Target Microservice: '{inferred_service}'\n\n"
            f"Task: Determine the investigation paths needed. Choose from: ['logs', 'metrics', 'deployments', 'knowledge'].\n"
            f"Respond with JSON format: {{\"service\": \"{inferred_service}\", \"investigation_paths\": [\"logs\", \"metrics\", \"deployments\", \"knowledge\"], \"plan\": [\"Step 1: ...\", \"Step 2: ...\"]}}"
        )
        
        try:
            resp = llm.invoke([{"role": "user", "content": prompt}])
            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0))
            paths = parsed.get("investigation_paths", ["logs", "metrics", "deployments", "knowledge"])
            plan = parsed.get("plan", [
                f"1. Query service logs for {inferred_service}",
                f"2. Inspect telemetry metrics for {inferred_service}",
                f"3. Check recent deployments for {inferred_service}",
                f"4. Search runbooks and past postmortems",
                f"5. Perform root cause synthesis"
            ])
        except Exception as e:
            logfire.warning(f"⚠️ Supervisor parsing fallback: {e}")
            paths = ["logs", "metrics", "deployments", "knowledge"]
            plan = [
                f"1. Query service logs for {inferred_service}",
                f"2. Inspect telemetry metrics for {inferred_service}",
                f"3. Check recent deployments for {inferred_service}",
                f"4. Search runbooks and past postmortems",
                f"5. Perform root cause synthesis"
            ]

        return {
            "service": inferred_service,
            "investigation_paths": paths,
            "plan": plan,
            "status": f"Supervisor planned investigation across: {', '.join(paths)}",
            "messages": [{"role": "supervisor", "content": f"Targeting service '{inferred_service}' with paths: {paths}"}]
        }
