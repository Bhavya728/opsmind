"""
Pydantic schemas and typed models for MCP Tools in the Incident Investigation System.
Provides strict validation, descriptive field definitions, and error handling.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


VALID_SERVICES = [
    "order-service",
    "payment-service",
    "auth-service",
    "inventory-service",
    "api-gateway",
    "analytics-service",
    "notification-service",
    "user-service",
    "database",
    "kube-system",
    "cert-manager"
]

VALID_SEVERITIES = ["INFO", "WARN", "ERROR", "FATAL", "DEBUG", "ALL"]


class GetServiceLogsInput(BaseModel):
    service: str = Field(..., description="Target microservice name to retrieve logs for")
    start_time: Optional[str] = Field(None, description="ISO-8601 UTC start time filter (e.g. '2026-03-01T14:00:00Z')")
    end_time: Optional[str] = Field(None, description="ISO-8601 UTC end time filter")
    severity: Optional[str] = Field("ALL", description="Minimum or specific log severity: INFO, WARN, ERROR, FATAL, ALL")
    limit: Optional[int] = Field(50, ge=1, le=200, description="Maximum number of log entries to return")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("Service name cannot be empty")
        return clean

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> str:
        if not v:
            return "ALL"
        clean = v.strip().upper()
        if clean not in VALID_SEVERITIES:
            raise ValueError(f"Invalid severity '{v}'. Allowed: {VALID_SEVERITIES}")
        return clean


class GetServiceMetricsInput(BaseModel):
    service: str = Field(..., description="Target microservice name to retrieve metrics for")
    metric_name: Optional[str] = Field(None, description="Specific metric name (e.g. 'p99_latency_ms', 'db_cpu_utilization_pct', 'error_rate_pct')")
    start_time: Optional[str] = Field(None, description="ISO-8601 UTC start time window")
    end_time: Optional[str] = Field(None, description="ISO-8601 UTC end time window")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("Service name cannot be empty")
        return clean


class GetDeploymentHistoryInput(BaseModel):
    service: str = Field(..., description="Target microservice name to inspect deployment history")
    limit: Optional[int] = Field(5, ge=1, le=20, description="Number of recent deployments to fetch")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("Service name cannot be empty")
        return clean


class SearchRunbookInput(BaseModel):
    query: str = Field(..., description="Keywords or problem symptoms to search runbooks for")
    service: Optional[str] = Field(None, description="Optional service name filter")
    limit: Optional[int] = Field(3, ge=1, le=10, description="Max number of matching runbook entries")


class LogEntry(BaseModel):
    timestamp: str
    level: str
    service: str
    trace_id: Optional[str] = None
    message: str
    stack_trace: Optional[str] = None


class MetricSeries(BaseModel):
    service: str
    metrics: Dict[str, Any]
    summary: Optional[Dict[str, Any]] = None


class DeploymentRecord(BaseModel):
    deployment_id: str
    service: str
    version: str
    timestamp: str
    status: str
    author: str
    changelog: str


class RunbookResult(BaseModel):
    id: str
    service: str
    title: str
    alert_trigger: str
    symptoms: List[str]
    investigation_steps: List[str]
    mitigation_steps: List[str]
    score: Optional[float] = 1.0
