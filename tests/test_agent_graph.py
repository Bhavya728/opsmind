"""
Multi-Agent LangGraph Workflow Tests.
"""

try:
    import pytest
except ImportError:
    from tests import compat as pytest
from app.agents.graph import incident_agent


def test_agent_graph_execution():
    initial_state = {
        "incident_query": "Order service p99 latency spiked to 4800ms following deployment of v2.4.1",
        "service": "order-service",
        "thread_id": "test_agent_session",
        "plan": [],
        "investigation_paths": [],
        "current_step": "supervisor",
        "status": "Starting",
        "logs_data": [],
        "metrics_data": {},
        "deployments_data": [],
        "knowledge_data": [],
        "observed_evidence": [],
        "inferred_cause": "",
        "confidence": 0.0,
        "recommended_actions": [],
        "tools_used": [],
        "guardrail_blocked": False,
        "guardrail_response": None,
        "messages": [{"role": "user", "content": "Order service latency spike"}],
        "final_output": {}
    }

    result = incident_agent.invoke(initial_state, config={"configurable": {"thread_id": "test_graph_1"}})
    
    # Verify outputs
    output = result.get("final_output", {})
    assert output is not None
    assert "root_cause" in output
    assert "confidence" in output
    assert output["confidence"] >= 0.80
    assert len(output["evidence"]) > 0
    assert len(output["tools_used"]) > 0
    assert len(output["recommended_actions"]) >= 1
    assert "order-service" in output["service"]


def test_agent_graph_auto_service_detection():
    initial_state = {
        "incident_query": "Kafka bootstrap broker DNS lookup failed causing notification-service crash loop",
        "service": "", # service not provided, must be inferred
        "thread_id": "test_auto_service",
        "plan": [],
        "investigation_paths": [],
        "current_step": "supervisor",
        "status": "Starting",
        "logs_data": [],
        "metrics_data": {},
        "deployments_data": [],
        "knowledge_data": [],
        "observed_evidence": [],
        "inferred_cause": "",
        "confidence": 0.0,
        "recommended_actions": [],
        "tools_used": [],
        "guardrail_blocked": False,
        "guardrail_response": None,
        "messages": [{"role": "user", "content": "Kafka lookup error"}],
        "final_output": {}
    }

    result = incident_agent.invoke(initial_state, config={"configurable": {"thread_id": "test_graph_2"}})
    output = result.get("final_output", {})
    assert "notification-service" in output["service"]
    assert "kafka" in output["root_cause"].lower() or "broker" in output["root_cause"].lower()
