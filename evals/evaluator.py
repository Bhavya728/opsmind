"""
OpsMind Automated Evaluation Pipeline.
Executes the golden dataset across RAG, Agent/Tool, and Guardrail evaluation metrics.
Saves comprehensive results to evals/evaluation_results.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

# Ensure project root is first in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agents.graph import investigate_query
from app.services.retrieval.chroma_service import chroma_service
from app.data.knowledge_base import KNOWLEDGE_DOCUMENTS, WEB_KNOWLEDGE_INDEX
from evals.metrics import (
    calculate_faithfulness,
    calculate_answer_relevance,
    calculate_context_relevance,
    calculate_context_recall,
    evaluate_tool_selection,
    evaluate_guardrail_decision,
    evaluate_citations_presence
)
from app.utils.logger import logfire

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")


def _resolve_context_texts(sources: List[Dict[str, Any]]) -> List[str]:
    """Resolve full text content for retrieved sources."""
    context_chunks = []
    for s in sources:
        doc_id = s.get("id", "")
        title = s.get("title", "")
        # Try finding in knowledge base
        matched_doc = None
        for doc in KNOWLEDGE_DOCUMENTS:
            if doc["id"] == doc_id or doc["id"] in doc_id:
                matched_doc = doc
                break
        if matched_doc:
            context_chunks.append(f"{matched_doc['title']}\n{matched_doc['content']}")
            continue
            
        # Try finding in web index
        matched_web = None
        for web in WEB_KNOWLEDGE_INDEX:
            if web["url"] == doc_id or web["title"] == title:
                matched_web = web
                break
        if matched_web:
            context_chunks.append(f"{matched_web['title']}\n{matched_web['snippet']}")
            continue

        context_chunks.append(f"{title} {doc_id}")
    return context_chunks


def evaluate_single_query(question: str, expected_evidence: Optional[List[str]] = None) -> Dict[str, Any]:
    """Evaluates a single research query and returns calculated metrics."""
    start_time = time.time()
    output = investigate_query(question)
    latency = round(time.time() - start_time, 3)
    
    answer = output.get("answer", "")
    sources = output.get("sources", [])
    context_texts = _resolve_context_texts(sources)
    
    faith_score = calculate_faithfulness(answer, context_texts)
    ans_rel_score = calculate_answer_relevance(question, answer)
    ctx_rel_score = calculate_context_relevance(question, context_texts)
    cite_score = evaluate_citations_presence(answer)
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_seconds": latency,
        "faithfulness": faith_score,
        "answer_relevance": ans_rel_score,
        "context_relevance": ctx_rel_score,
        "citation_score": cite_score,
        "confidence": output.get("confidence", 0.0)
    }


def run_evaluations(dataset_path: str = DATASET_PATH, output_path: str = RESULTS_PATH) -> Dict[str, Any]:
    """Execute complete evaluation suite against golden dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset.get("samples", [])
    total_samples = len(samples)
    logfire.info(f"🚀 Starting OpsMind Evaluation Run on {total_samples} test cases...")

    results = []
    rag_faithfulness_scores = []
    rag_answer_rel_scores = []
    rag_context_rel_scores = []
    rag_context_recall_scores = []
    tool_accuracy_scores = []
    guardrail_accuracy_scores = []
    citation_scores = []
    latencies = []
    total_tokens_estimated = 0

    for idx, sample in enumerate(samples):
        q_id = sample.get("id", idx + 1)
        question = sample.get("question", "")
        test_type = sample.get("test_type", "research")
        expected_blocked = sample.get("expected_blocked", False)
        expected_evidence = sample.get("expected_supporting_evidence", [])
        expected_tools = sample.get("expected_tools", [])

        logfire.info(f"[{idx+1}/{total_samples}] Running eval case #{q_id}: '{question[:50]}'")
        start_time = time.time()
        
        # Execute System
        try:
            output = investigate_query(question)
            latency = round(time.time() - start_time, 3)
        except Exception as e:
            latency = round(time.time() - start_time, 3)
            output = {
                "answer": f"Evaluation Execution Error: {e}",
                "sources": [],
                "agents_used": [],
                "tools_used": [],
                "confidence": 0.0,
                "guardrail_blocked": False
            }

        latencies.append(latency)
        actual_answer = output.get("answer", "")
        actual_sources = output.get("sources", [])
        actual_tools = output.get("tools_used", [])
        actual_blocked = output.get("guardrail_blocked", False)

        tokens = int(len(question.split()) * 1.3 + len(actual_answer.split()) * 1.3 + 450)
        total_tokens_estimated += tokens

        guard_score = evaluate_guardrail_decision(expected_blocked, actual_blocked)
        guardrail_accuracy_scores.append(guard_score)

        tool_score = evaluate_tool_selection(expected_tools, actual_tools)
        tool_accuracy_scores.append(tool_score)

        if not expected_blocked and not actual_blocked:
            context_texts = _resolve_context_texts(actual_sources)
            
            faith_score = calculate_faithfulness(actual_answer, context_texts)
            ans_rel_score = calculate_answer_relevance(question, actual_answer)
            ctx_rel_score = calculate_context_relevance(question, context_texts)
            ctx_rec_score = calculate_context_recall(expected_evidence, context_texts)
            cite_score = evaluate_citations_presence(actual_answer)

            rag_faithfulness_scores.append(faith_score)
            rag_answer_rel_scores.append(ans_rel_score)
            rag_context_rel_scores.append(ctx_rel_score)
            rag_context_recall_scores.append(ctx_rec_score)
            citation_scores.append(cite_score)
            
            case_passed = (faith_score >= 0.70 and ans_rel_score >= 0.70 and guard_score == 1.0)
        else:
            faith_score = 1.0
            ans_rel_score = 1.0
            ctx_rel_score = 1.0
            ctx_rec_score = 1.0
            cite_score = 1.0
            case_passed = (guard_score == 1.0)

        case_result = {
            "id": q_id,
            "question": question,
            "test_type": test_type,
            "category": sample.get("category", "general"),
            "passed": case_passed,
            "latency_seconds": latency,
            "estimated_tokens": tokens,
            "guardrail_score": guard_score,
            "tool_selection_score": tool_score,
            "faithfulness": faith_score,
            "answer_relevance": ans_rel_score,
            "context_relevance": ctx_rel_score,
            "context_recall": ctx_rec_score,
            "citation_score": cite_score,
            "actual_answer_snippet": actual_answer[:220] + "..." if len(actual_answer) > 220 else actual_answer,
            "actual_sources": actual_sources,
            "agents_used": output.get("agents_used", []),
            "tools_used": actual_tools
        }
        results.append(case_result)

    avg_faithfulness = round(sum(rag_faithfulness_scores) / max(1, len(rag_faithfulness_scores)), 3)
    avg_answer_relevance = round(sum(rag_answer_rel_scores) / max(1, len(rag_answer_rel_scores)), 3)
    avg_context_relevance = round(sum(rag_context_rel_scores) / max(1, len(rag_context_rel_scores)), 3)
    avg_context_recall = round(sum(rag_context_recall_scores) / max(1, len(rag_context_recall_scores)), 3)
    avg_tool_accuracy = round(sum(tool_accuracy_scores) / max(1, len(tool_accuracy_scores)), 3)
    avg_guardrail_accuracy = round(sum(guardrail_accuracy_scores) / max(1, len(guardrail_accuracy_scores)), 3)
    avg_citation_score = round(sum(citation_scores) / max(1, len(citation_scores)), 3)
    avg_latency = round(sum(latencies) / max(1, len(latencies)), 3)
    
    passed_cases = sum(1 for r in results if r["passed"])
    overall_score = round(
        (avg_faithfulness * 0.25) +
        (avg_answer_relevance * 0.20) +
        (avg_context_relevance * 0.15) +
        (avg_tool_accuracy * 0.15) +
        (avg_guardrail_accuracy * 0.15) +
        (avg_citation_score * 0.10),
        3
    )

    evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total_samples,
        "passed_cases": passed_cases,
        "failed_cases": total_samples - passed_cases,
        "overall_score": overall_score,
        "metrics_summary": {
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_answer_relevance,
            "context_relevance": avg_context_relevance,
            "context_recall": avg_context_recall,
            "tool_selection_accuracy": avg_tool_accuracy,
            "guardrail_accuracy": avg_guardrail_accuracy,
            "citation_compliance": avg_citation_score,
            "average_latency_sec": avg_latency,
            "total_tokens_consumed": total_tokens_estimated
        },
        "case_results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logfire.info(f"📊 Evaluation Complete! Overall Score: {overall_score * 100}% | Passed: {passed_cases}/{total_samples}")
    return evaluation_report


# Backward compatibility aliases
run_full_evaluation = run_evaluations
evaluate_single_sample = evaluate_single_query


if __name__ == "__main__":
    report = run_evaluations()
    print("\n" + "="*50)
    print("       OPSMIND EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Score:            {report['overall_score']*100:.1f}%")
    print(f"Passed Test Cases:        {report['passed_cases']}/{report['total_cases']}")
    print(f"Faithfulness:             {report['metrics_summary']['faithfulness']*100:.1f}%")
    print(f"Answer Relevance:         {report['metrics_summary']['answer_relevance']*100:.1f}%")
    print(f"Context Relevance:        {report['metrics_summary']['context_relevance']*100:.1f}%")
    print(f"Context Recall:           {report['metrics_summary']['context_recall']*100:.1f}%")
    print(f"Tool Accuracy:            {report['metrics_summary']['tool_selection_accuracy']*100:.1f}%")
    print(f"Guardrail Accuracy:       {report['metrics_summary']['guardrail_accuracy']*100:.1f}%")
    print(f"Citation Compliance:      {report['metrics_summary']['citation_compliance']*100:.1f}%")
    print(f"Average Latency:          {report['metrics_summary']['average_latency_sec']}s")
    print("="*50)
