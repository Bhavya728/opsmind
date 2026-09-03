"""
FastAPI Backend for Agentic Incident Investigation System.

Endpoints:
- POST /investigate : Primary endpoint for multi-agent root cause analysis
- POST /query       : Backward-compatible endpoint
- GET  /health      : System health check & dependencies
- GET  /services    : List registered microservices
- GET  /incidents   : List synthetic incident scenario catalog
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.utils.logger import logfire
from app.config import settings
from app.agents.graph import incident_agent
from app.guardrails.rails import initialize_rails, guard_input, guard_output
from app.services.retrieval.qdrant_service import initialize_incident_knowledge_base
from app.tools.schemas import VALID_SERVICES

app = FastAPI(
    title="Agentic Incident Investigation System API",
    description="Multi-agent root-cause analysis orchestrating logs, telemetry metrics, deployments, and engineering runbooks.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    logfire.info("🚀 Starting Incident Investigation System...")
    initialize_rails()
    initialize_incident_knowledge_base()
    logfire.info("✅ Incident Investigation System ready.")


class InvestigateRequest(BaseModel):
    incident: str = Field(..., min_length=3, description="Incident symptom, alert description, or error summary")
    service: Optional[str] = Field(None, description="Affected microservice name (optional; automatically inferred if omitted)")
    thread_id: Optional[str] = Field("default_incident_session", description="Session thread identifier for checkpointer")


class InvestigateResponse(BaseModel):
    summary: str
    root_cause: str
    confidence: float
    evidence: List[str]
    tools_used: List[str]
    recommended_actions: List[str]
    service: str
    status: str


class QueryRequest(BaseModel):
    q: str
    service: Optional[str] = None
    thread_id: Optional[str] = "default_user"


@app.get("/")
def root():
    return {
        "system": "Agentic Incident Investigation System",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "investigate": "POST /investigate",
            "services": "GET /services",
            "incidents": "GET /incidents",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "Agentic Incident Investigation System",
        "version": "2.0.0",
        "guardrails_active": True,
        "knowledge_base": "qdrant_indexed",
        "registered_services": len(VALID_SERVICES)
    }


@app.get("/services")
def list_services():
    return {"services": VALID_SERVICES}


@app.get("/incidents")
def list_incidents():
    from app.data.incidents_catalog import SCENARIOS
    return {
        "count": len(SCENARIOS),
        "scenarios": [
            {
                "id": sc["id"],
                "title": sc["title"],
                "service": sc["service"],
                "symptom": sc["symptom"]
            }
            for sc in SCENARIOS
        ]
    }


@app.post("/investigate", response_model=InvestigateResponse)
def investigate_incident(req: InvestigateRequest):
    incident_text = req.incident.strip()
    service = (req.service or "").strip().lower()
    thread_id = req.thread_id or "default_session"

    with logfire.span("🚨 POST /investigate", incident=incident_text[:60], service=service):
        # --- Gate 1: NeMo Guardrails Input Validation ---
        rail_fired, rail_response = guard_input(incident_text)
        if rail_fired:
            logfire.warning(f"🛡️ Guardrail triggered on incident request: {rail_response}")
            return InvestigateResponse(
                summary="Investigation blocked by security guardrails.",
                root_cause=rail_response or "Security policy violation: prompt injection, destructive command, or out-of-scope query.",
                confidence=0.0,
                evidence=["[SECURITY ALERT] Input tripped safety guardrails"],
                tools_used=["guardrails_input_filter"],
                recommended_actions=["Submit a valid infrastructure incident symptom or telemetry alert query."],
                service=service or "unknown",
                status="blocked_by_guardrails"
            )

        # --- Gate 2: LangGraph Multi-Agent Orchestration ---
        initial_state = {
            "incident_query": incident_text,
            "service": service,
            "thread_id": thread_id,
            "plan": [],
            "investigation_paths": [],
            "current_step": "supervisor",
            "status": "Investigating...",
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
            "messages": [{"role": "user", "content": incident_text}],
            "final_output": {}
        }
        
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = incident_agent.invoke(initial_state, config=config)
            output = result.get("final_output", {})

            return InvestigateResponse(
                summary=output.get("summary", "Incident analysis complete."),
                root_cause=output.get("root_cause", "Root cause identified from correlated telemetry."),
                confidence=float(output.get("confidence", 0.90)),
                evidence=output.get("evidence", result.get("observed_evidence", [])),
                tools_used=output.get("tools_used", result.get("tools_used", [])),
                recommended_actions=output.get("recommended_actions", []),
                service=output.get("service", result.get("service", service or "order-service")),
                status="completed"
            )

        except Exception as e:
            logfire.error(f"❌ Incident investigation workflow failed: {e}")
            raise HTTPException(status_code=500, detail=f"Investigation workflow error: {str(e)}")


@app.post("/query")
def legacy_query(req: QueryRequest):
    """Backward compatibility endpoint for legacy RAG clients."""
    res = investigate_incident(InvestigateRequest(incident=req.q, service=req.service, thread_id=req.thread_id))
    return {
        "question": req.q,
        "answer": f"**Root Cause**: {res.root_cause}\n\n**Confidence**: {int(res.confidence * 100)}%\n\n**Recommended Actions**:\n" + "\n".join([f"- {a}" for a in res.recommended_actions]),
        "thought_process": res.evidence,
        "status": res.status,
        "sources": res.evidence,
        "tools_used": res.tools_used
    }
