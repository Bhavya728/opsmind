"""
OpsMind Evaluation Metrics — deterministic checks computed from the system's own output.
They need no LLM and run identically offline and live. Answer quality (correctness,
faithfulness, relevance) is graded separately by the independent judge in evals/judge.py.

Every metric returns None when it does not apply to a case, so averages only include
cases where the metric is meaningful.
"""

from typing import Any, Dict, List, Optional

TOOL_TO_ROUTE = {"knowledge_search": "knowledge_base", "web_search": "web"}


def guardrail_correct(expected_blocked: bool, actual_blocked: bool) -> float:
    """1.0 when the system blocked/refused exactly when the dataset says it should."""
    return 1.0 if bool(expected_blocked) == bool(actual_blocked) else 0.0


def routing_recall(expected_tools: List[str], routing: Optional[Dict[str, bool]]) -> Optional[float]:
    """
    Share of required retrieval paths the planner actually used. Extra paths are not penalized:
    searching the web in addition to the knowledge base is a valid choice.
    """
    required = [TOOL_TO_ROUTE[t] for t in expected_tools if t in TOOL_TO_ROUTE]
    if not required or routing is None:
        return None
    return round(sum(1 for r in required if routing.get(r)) / len(required), 3)


def _internal_ids(sources: List[Dict[str, Any]]) -> set:
    return {s.get("id") for s in sources if s.get("type") == "internal_doc"}


def source_recall(expected_ids: List[str], sources: List[Dict[str, Any]]) -> Optional[float]:
    """Share of the expected knowledge-base documents that the critic accepted."""
    if not expected_ids:
        return None
    return round(len(set(expected_ids) & _internal_ids(sources)) / len(set(expected_ids)), 3)


def source_precision(expected_ids: List[str], sources: List[Dict[str, Any]]) -> Optional[float]:
    """Share of accepted knowledge-base documents that the dataset lists as relevant."""
    accepted = _internal_ids(sources)
    if not expected_ids or not accepted:
        return None
    return round(len(set(expected_ids) & accepted) / len(accepted), 3)


def cites_sources(result: Dict[str, Any]) -> Optional[float]:
    """1.0 when the report cites at least one verified source; None when nothing was verified."""
    if not result.get("sources"):
        return None
    return 1.0 if result.get("citations") else 0.0


def mean(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 3) if present else None
