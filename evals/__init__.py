"""
OpsMind Evals Module.
"""

from evals.evaluator import run_evaluations, evaluate_single_query, run_full_evaluation, evaluate_single_sample
from evals.metrics import (
    calculate_faithfulness,
    calculate_answer_relevance,
    calculate_context_relevance,
    calculate_context_recall,
    evaluate_tool_selection,
    evaluate_guardrail_decision,
    evaluate_citations_presence
)

__all__ = [
    "run_evaluations",
    "evaluate_single_query",
    "run_full_evaluation",
    "evaluate_single_sample",
    "calculate_faithfulness",
    "calculate_answer_relevance",
    "calculate_context_relevance",
    "calculate_context_recall",
    "evaluate_tool_selection",
    "evaluate_guardrail_decision",
    "evaluate_citations_presence"
]
