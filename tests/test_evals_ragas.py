"""
Unit & Integration Tests for RAGAS Evaluation and Custom Heuristics.
Tests metric calculation, golden dataset parsing, and evaluator output structure.
"""

import os
import json
from evals.metrics import (
    custom_faithfulness_heuristic,
    custom_answer_relevance_heuristic,
    custom_context_relevance_heuristic,
    custom_context_recall_heuristic,
    custom_tool_selection_accuracy,
    custom_guardrail_accuracy,
    custom_citation_compliance
)
from evals.evaluator import evaluate_single_query, DATASET_PATH


def test_ragas_package_metrics_importable():
    """Verify that official ragas metrics can be imported and initialized."""
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    assert faithfulness is not None
    assert answer_relevancy is not None
    assert context_precision is not None
    assert context_recall is not None
    assert hasattr(faithfulness, "name")
    assert hasattr(answer_relevancy, "name")


def test_custom_faithfulness_heuristic():
    """Test custom faithfulness heuristic calculation."""
    context = ["ChromaDB uses HNSW graph indexing for sub-10ms approximate nearest neighbor queries."]
    grounded_answer = "ChromaDB implements HNSW graph indexing for fast approximate nearest neighbor search."
    ungrounded_answer = "The capital of Mars is Olympus Mons and it runs quantum computing clusters."
    
    score_grounded = custom_faithfulness_heuristic(grounded_answer, context)
    score_ungrounded = custom_faithfulness_heuristic(ungrounded_answer, context)
    
    assert score_grounded >= 0.70
    assert score_ungrounded < 0.50


def test_custom_answer_relevance_heuristic():
    """Test custom answer relevance heuristic calculation."""
    question = "What are the major metrics for RAG evaluation?"
    relevant_answer = "The major metrics for RAG evaluation include Faithfulness, Answer Relevance, Context Precision, and Context Recall."
    irrelevant_answer = "To bake a chocolate cake, preheat the oven to 350 degrees."
    
    score_rel = custom_answer_relevance_heuristic(question, relevant_answer)
    score_irrel = custom_answer_relevance_heuristic(question, irrelevant_answer)
    
    assert score_rel >= 0.70
    assert score_irrel < 0.40


def test_custom_guardrail_accuracy():
    """Test custom guardrail accuracy scoring."""
    assert custom_guardrail_accuracy(expected_blocked=True, actual_blocked=True) == 1.0
    assert custom_guardrail_accuracy(expected_blocked=False, actual_blocked=False) == 1.0
    assert custom_guardrail_accuracy(expected_blocked=True, actual_blocked=False) == 0.0


def test_custom_citation_compliance():
    """Test citation compliance verification."""
    cited = "Retrieval-Augmented Generation decouples knowledge [Source: doc-rag-eval-01] and dynamic tiering [Source: doc-llm-gateway-03]."
    uncited = "Retrieval Augmented Generation decouples knowledge without citations."
    
    assert custom_citation_compliance(cited) == 1.0
    assert custom_citation_compliance(uncited) == 0.0


def test_golden_dataset_structure():
    """Verify that golden dataset contains 25 properly formatted test cases."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    assert len(samples) == 25
    for s in samples:
        assert "id" in s
        assert "question" in s
        assert "expected_blocked" in s


def test_evaluator_single_query_execution():
    """Test evaluating a single research question with structured output."""
    result = evaluate_single_query("What are the major approaches to RAG evaluation?")
    assert "question" in result
    assert "answer" in result
    assert "custom_heuristics" in result
    assert result["custom_heuristics"]["faithfulness"] >= 0.70
