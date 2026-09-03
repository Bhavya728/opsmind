"""
Unit and integration tests for MCP Tools.
"""

try:
    import pytest
except ImportError:
    from tests import compat as pytest
from pydantic import ValidationError

from app.tools.mcp_tools import (
    get_service_logs,
    get_service_metrics,
    get_deployment_history,
    search_runbook,
)
from app.tools.schemas import (
    GetServiceLogsInput,
    GetServiceMetricsInput,
    GetDeploymentHistoryInput,
    SearchRunbookInput,
)


def test_get_service_logs_valid():
    result = get_service_logs.invoke({"service": "order-service", "severity": "ERROR"})
    assert result["status"] == "success"
    assert result["service"] == "order-service"
    assert isinstance(result["logs"], list)
    assert result["count"] > 0
    # Every log entry should be an error or fatal
    for log in result["logs"]:
        assert log["level"] in ("ERROR", "FATAL")


def test_get_service_logs_unrecognized_service():
    result = get_service_logs.invoke({"service": "non-existent-microservice"})
    assert result["status"] == "warning"
    assert result["count"] == 0


def test_get_service_logs_schema_validation():
    with pytest.raises(ValidationError):
        GetServiceLogsInput(service="")
    with pytest.raises(ValidationError):
        GetServiceLogsInput(service="order-service", severity="INVALID_SEVERITY")
    with pytest.raises(ValidationError):
        GetServiceLogsInput(service="order-service", limit=500)


def test_get_service_metrics_valid():
    result = get_service_metrics.invoke({"service": "order-service"})
    assert result["status"] == "success"
    assert "p99_latency_ms" in result["metrics"]
    assert len(result["metrics"]["p99_latency_ms"]) > 0


def test_get_service_metrics_specific_metric():
    result = get_service_metrics.invoke({"service": "payment-service", "metric_name": "active_db_connections"})
    assert result["status"] == "success"
    assert "active_db_connections" in result["metrics"]


def test_get_deployment_history_valid():
    result = get_deployment_history.invoke({"service": "order-service", "limit": 2})
    assert result["status"] == "success"
    assert len(result["deployments"]) > 0
    first_dep = result["deployments"][0]
    assert "version" in first_dep
    assert "changelog" in first_dep


def test_search_runbook_valid():
    result = search_runbook.invoke({"query": "database connection pool exhaustion", "service": "order-service"})
    assert result["status"] == "success"
    assert result["count"] > 0
    first_rb = result["results"][0]
    assert "RB-ORD-01" in first_rb["id"]
    assert "mitigation_steps" in first_rb
