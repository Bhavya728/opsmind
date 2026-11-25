"""
Unit tests for the OpsMind evaluation pipeline: deterministic metrics, the LLM judge's
scoring, dataset structure, and error handling in the evaluator.
"""

import json
from langchain_core.messages import AIMessage

from evals.metrics import guardrail_correct, routing_recall, source_recall, source_precision, cites_sources, mean
from evals.evaluator import evaluate_case, DATASET_PATH
import evals.evaluator as evaluator
import evals.judge as judge


def test_guardrail_correct():
    assert guardrail_correct(True, True) == 1.0
    assert guardrail_correct(False, False) == 1.0
    assert guardrail_correct(True, False) == 0.0


def test_routing_recall_requires_expected_paths_only():
    assert routing_recall(["knowledge_search"], {"knowledge_base": True, "web": True}) == 1.0
    assert routing_recall(["knowledge_search", "web_search"], {"knowledge_base": True, "web": False}) == 0.5
    assert routing_recall([], {"knowledge_base": True, "web": True}) is None


def test_source_recall_and_precision():
    sources = [{"id": "doc-a", "type": "internal_doc"}, {"id": "doc-b", "type": "internal_doc"},
               {"id": "web-1", "type": "web_article"}]
    assert source_recall(["doc-a", "doc-c"], sources) == 0.5
    assert source_precision(["doc-a"], sources) == 0.5
    assert source_recall([], sources) is None


def test_cites_sources():
    assert cites_sources({"sources": [{"id": "a"}], "citations": ["a"]}) == 1.0
    assert cites_sources({"sources": [{"id": "a"}], "citations": []}) == 0.0
    assert cites_sources({"sources": [], "citations": []}) is None
    assert mean([1.0, None, 0.0]) == 0.5


def test_golden_dataset_structure():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    assert len(samples) == data["total_cases"] >= 25
    assert len({s["id"] for s in samples}) == len(samples)
    for s in samples:
        for field in ("id", "question", "test_type", "expected_blocked", "expected_tools",
                      "expected_relevant_sources", "expected_answer_characteristics"):
            assert field in s, (s["id"], field)


def test_evaluate_case_offline_research():
    """Deterministic metrics are computed for a research case without the judge."""
    case = evaluate_case({"id": 1, "question": "What are the major approaches to RAG evaluation?",
                          "expected_tools": ["knowledge_search"], "expected_relevant_sources": ["doc-rag-eval-01"],
                          "expected_blocked": False}, use_judge=False)
    assert case["status"] == "ok"
    assert case["metrics"]["guardrail_accuracy"] == 1.0
    assert case["metrics"]["routing_recall"] == 1.0
    assert case["metrics"]["source_recall"] == 1.0
    assert case["metrics"].get("correctness") is None


def test_evaluate_case_records_errors():
    """A failed run is reported as an error, not scored as an answer."""
    original = evaluator.investigate_query
    try:
        def boom(q):
            raise RuntimeError("rate limited")
        evaluator.investigate_query = boom
        case = evaluate_case({"id": 99, "question": "x", "expected_blocked": False}, use_judge=False)
    finally:
        evaluator.investigate_query = original
    assert case["status"] == "error" and "rate limited" in case["error"] and case["passed"] is False


def test_judge_scoring_from_verdict():
    """Judge verdicts convert to correctness, faithfulness and relevance scores."""
    class FakeJudge:
        def invoke(self, prompt):
            return AIMessage(content="<think>reasoning</think>" + json.dumps({
                "characteristics": [{"index": 1, "covered": "yes"}, {"index": 2, "covered": "partial"},
                                    {"index": 3, "covered": "no"}],
                "claims_total": 10, "claims_unsupported": 2, "unsupported_examples": ["x"],
                "relevance": 4, "notes": "ok"}))
    original = judge.get_judge_llm
    try:
        judge.get_judge_llm = lambda: FakeJudge()
        verdict = judge.judge_answer("q", ["a", "b", "c"], "answer", [])
    finally:
        judge.get_judge_llm = original
    assert verdict["status"] == "ok"
    assert verdict["correctness"] == 0.5
    assert verdict["faithfulness"] == 0.8
    assert verdict["relevance"] == 0.75
