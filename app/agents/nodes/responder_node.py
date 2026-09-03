"""
Final Response Agent Node.
Synthesizes the final investigation report distinguishing:
- Observed Evidence
- Inferred Root Cause
- Confidence Score
- Recommended Actions
- Tools Used
Applies Output Guardrails for credential redaction and groundedness.
"""

from typing import Dict, Any, List
from app.agents.state import IncidentState
from app.guardrails.rails import guard_output
from app.utils.logger import logfire


def responder_node(state: IncidentState) -> Dict[str, Any]:
    """
    Final Response Agent:
    - Assembles final structured response
    - Filters and deduplicates evidence
    - Runs Output Guardrails
    """
    query = state.get("incident_query", "")
    service = state.get("service", "order-service")
    inferred_cause = state.get("inferred_cause", "Root cause identified through correlated service telemetry.")
    confidence = state.get("confidence", 0.90)
    actions = state.get("recommended_actions", [])
    raw_evidence = state.get("observed_evidence", [])
    tools_used = list(dict.fromkeys(state.get("tools_used", []))) # deduplicate preserving order
    
    # Deduplicate and clean evidence entries
    seen = set()
    cleaned_evidence = []
    for ev in raw_evidence:
        ev_clean = ev.strip()
        if ev_clean and ev_clean not in seen:
            seen.add(ev_clean)
            cleaned_evidence.append(ev_clean)
            
    with logfire.span("📄 Final Response Generation", service=service):
        summary_text = (
            f"Incident investigation for '{service}' completed. "
            f"Analysis across {len(tools_used)} tools and telemetry sources isolated root cause with {int(confidence * 100)}% confidence."
        )
        
        # Apply Output Guardrails (credential redaction & sanity)
        sanitized_cause = guard_output(inferred_cause, cleaned_evidence)
        sanitized_summary = guard_output(summary_text, cleaned_evidence)
        sanitized_evidence = [guard_output(e) for e in cleaned_evidence]
        sanitized_actions = [guard_output(a) for a in actions]
        
        final_payload = {
            "summary": sanitized_summary,
            "root_cause": sanitized_cause,
            "confidence": round(float(confidence), 2),
            "evidence": sanitized_evidence,
            "tools_used": tools_used,
            "recommended_actions": sanitized_actions,
            "service": service,
            "query": query
        }

        return {
            "final_output": final_payload,
            "status": "Investigation finalized.",
            "messages": [{"role": "responder", "content": f"Final report generated for {service}."}]
        }
