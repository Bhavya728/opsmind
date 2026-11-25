"""
OpsMind Automated Evaluation Pipeline.

Runs the golden dataset through the full system and scores each case on two layers:
A. Deterministic checks from the system's own output (evals/metrics.py):
   guardrail decision, routing recall, critic source recall/precision, citation use.
B. Independent LLM judge (evals/judge.py, JUDGE_MODEL): correctness against the expected
   answer characteristics, faithfulness to sources, and relevance. Requires USE_LIVE_LLM=true.

Failed runs (e.g. rate limits) are recorded as errors and excluded from the averages.

Usage:
    python evals/evaluator.py                 # all cases
    python evals/evaluator.py --limit 5       # first 5 cases
    python evals/evaluator.py --ids 1,4,26    # specific cases
    python evals/evaluator.py --no-judge      # deterministic checks only
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

# Ensure project root is first in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import settings
from app.agents.graph import investigate_query
from app.gateway.client import get_llm_runtime_status
from evals.metrics import guardrail_correct, routing_recall, source_recall, source_precision, cites_sources, mean
from evals.judge import judge_answer, judge_available
from app.utils.logger import logfire

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")

# Thresholds a research case must meet to pass (judge metrics only apply when the judge ran).
PASS_CORRECTNESS = 0.6
PASS_FAITHFULNESS = 0.7

SUMMARY_METRICS = [
    "correctness", "faithfulness", "relevance", "routing_recall",
    "source_recall", "source_precision", "cites_sources", "guardrail_accuracy",
]
# Metrics averaged into the overall score (system confidence is reported, not scored).
SCORED_METRICS = ["correctness", "faithfulness", "relevance", "routing_recall", "source_recall",
                  "cites_sources", "guardrail_accuracy"]


def _config_snapshot(judge_enabled: bool) -> Dict[str, Any]:
    return {
        "mode": settings.MODE,
        "llm_backend": get_llm_runtime_status(),
        "fast_model": settings.GROQ_FAST_MODEL,
        "report_model": settings.GROQ_MODEL,
        "verifier_model": settings.VERIFIER_MODEL,
        "judge_model": settings.JUDGE_MODEL if judge_enabled else None,
        "web_search": settings.ACTIVE_WEB_PROVIDER,
        "embeddings": "deterministic" if settings.USE_DETERMINISTIC_MODE else settings.EMBEDDING_MODEL,
        "min_relevance_score": settings.MIN_RELEVANCE_SCORE,
        "max_research_rounds": settings.MAX_RESEARCH_ROUNDS,
        "claim_verification": settings.CLAIM_VERIFICATION_ENABLED,
    }


def evaluate_case(sample: Dict[str, Any], use_judge: bool) -> Dict[str, Any]:
    """Run one dataset case through the system and score it."""
    question = sample.get("question", "")
    expected_blocked = bool(sample.get("expected_blocked", False))
    case: Dict[str, Any] = {
        "id": sample.get("id"),
        "question": question,
        "test_type": sample.get("test_type", "research"),
        "category": sample.get("category", "general"),
    }

    start = time.time()
    try:
        result = investigate_query(question)
    except Exception as e:
        case.update({"status": "error", "error": f"{type(e).__name__}: {e}"[:300],
                     "latency_seconds": round(time.time() - start, 2), "passed": False})
        logfire.warning(f"⚠️ Eval case #{case['id']} failed: {case['error']}")
        return case

    blocked = bool(result.get("guardrail_blocked"))
    metrics: Dict[str, Optional[float]] = {"guardrail_accuracy": guardrail_correct(expected_blocked, blocked)}
    judge: Dict[str, Any] = {}

    if not expected_blocked and not blocked:
        sources = result.get("sources", [])
        expected_sources = sample.get("expected_relevant_sources", [])
        metrics.update({
            "routing_recall": routing_recall(sample.get("expected_tools", []), result.get("routing")),
            "source_recall": source_recall(expected_sources, sources),
            "source_precision": source_precision(expected_sources, sources),
            "cites_sources": cites_sources(result),
        })
        if use_judge:
            try:
                judge = judge_answer(question, sample.get("expected_answer_characteristics", []),
                                     result.get("answer", ""), sources)
            except Exception as e:
                judge = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            if judge.get("status") == "ok":
                metrics.update({k: judge.get(k) for k in ("correctness", "faithfulness", "relevance")})

    if expected_blocked or blocked:
        passed = metrics["guardrail_accuracy"] == 1.0
    else:
        checks = [
            metrics.get("routing_recall") in (None, 1.0),
            metrics.get("cites_sources") in (None, 1.0),
            metrics.get("correctness") is None or metrics["correctness"] >= PASS_CORRECTNESS,
            metrics.get("faithfulness") is None or metrics["faithfulness"] >= PASS_FAITHFULNESS,
        ]
        passed = all(checks)

    answer = result.get("answer", "")
    case.update({
        "status": "ok",
        "passed": passed,
        "expected_blocked": expected_blocked,
        "actual_blocked": blocked,
        "metrics": metrics,
        "judge": judge,
        "system": {
            "confidence": result.get("confidence"),
            "evidence_quality": result.get("evidence_quality"),
            "claim_support": result.get("claim_support"),
            "routing": result.get("routing"),
            "research_rounds": result.get("research_rounds"),
            "revisions": result.get("revisions"),
            "tokens": (result.get("token_usage") or {}).get("total"),
            "warnings": result.get("warnings", []),
        },
        "latency_seconds": round(time.time() - start, 2),
        "sources": [{"id": s.get("id"), "type": s.get("type"), "score": s.get("score")} for s in result.get("sources", [])],
        "citations": result.get("citations", []),
        "actual_answer_snippet": answer[:300] + ("..." if len(answer) > 300 else ""),
    })
    return case


def evaluate_single_query(question: str, expected_characteristics: Optional[List[str]] = None,
                          use_judge: Optional[bool] = None) -> Dict[str, Any]:
    """Evaluate one ad-hoc research question."""
    use_judge = (judge_available() is None) if use_judge is None else use_judge
    return evaluate_case({"id": "adhoc", "question": question,
                          "expected_answer_characteristics": expected_characteristics or []}, use_judge)


def run_evaluations(dataset_path: str = DATASET_PATH, output_path: str = RESULTS_PATH,
                    limit: Optional[int] = None, ids: Optional[List[int]] = None,
                    use_judge: bool = True) -> Dict[str, Any]:
    """Execute the evaluation suite against the golden dataset and save the report."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f).get("samples", [])
    if ids:
        samples = [s for s in samples if s.get("id") in set(ids)]
    if limit:
        samples = samples[:limit]

    judge_reason = judge_available() if use_judge else "disabled with --no-judge"
    use_judge = judge_reason is None
    if not use_judge:
        logfire.warning(f"⚠️ Judge metrics skipped: {judge_reason}")
    logfire.info(f"🚀 Starting OpsMind evaluation on {len(samples)} cases (judge: {settings.JUDGE_MODEL if use_judge else 'off'})")

    run_start = time.time()
    cases = []
    for idx, sample in enumerate(samples, start=1):
        logfire.info(f"[{idx}/{len(samples)}] Eval case #{sample.get('id')}: '{sample.get('question', '')[:60]}'")
        cases.append(evaluate_case(sample, use_judge))

    completed = [c for c in cases if c["status"] == "ok"]
    summary = {m: mean([c["metrics"].get(m) for c in completed]) for m in SUMMARY_METRICS}
    summary.update({
        "average_latency_sec": mean([c["latency_seconds"] for c in completed]),
        "average_system_confidence": mean([c["system"]["confidence"] for c in completed if not c["actual_blocked"]]),
        "total_tokens": sum(c["system"]["tokens"] or 0 for c in completed),
    })
    scored = [summary[m] for m in SCORED_METRICS if summary.get(m) is not None]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _config_snapshot(use_judge),
        "judge_status": "ok" if use_judge else judge_reason,
        "total_cases": len(cases),
        "completed_cases": len(completed),
        "error_cases": len(cases) - len(completed),
        "passed_cases": sum(1 for c in cases if c["passed"]),
        "overall_score": round(sum(scored) / len(scored), 3) if scored else None,
        "summary": summary,
        "run_seconds": round(time.time() - run_start, 1),
        "case_results": cases,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logfire.info(f"📊 Evaluation complete: {report['passed_cases']}/{len(cases)} passed, overall {report['overall_score']}")
    return report


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the OpsMind evaluation suite.")
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--ids", help="Comma-separated case ids to run, e.g. 1,4,26.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (deterministic checks only).")
    parser.add_argument("--output", default=RESULTS_PATH, help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    report = run_evaluations(
        output_path=args.output, limit=args.limit, use_judge=not args.no_judge,
        ids=[int(i) for i in args.ids.split(",")] if args.ids else None,
    )
    s = report["summary"]
    print("\n" + "=" * 56)
    print("              OPSMIND EVALUATION SUMMARY")
    print("=" * 56)
    print(f"Cases:                 {report['passed_cases']} passed / {report['completed_cases']} completed / {report['error_cases']} errors")
    print(f"Overall score:         {_pct(report['overall_score'])}")
    print(f"Judge ({report['config']['judge_model'] or 'off'}): {report['judge_status']}")
    print(f"  Correctness:         {_pct(s['correctness'])}")
    print(f"  Faithfulness:        {_pct(s['faithfulness'])}")
    print(f"  Relevance:           {_pct(s['relevance'])}")
    print(f"Routing recall:        {_pct(s['routing_recall'])}")
    print(f"Critic source recall:  {_pct(s['source_recall'])}   precision: {_pct(s['source_precision'])}")
    print(f"Cites sources:         {_pct(s['cites_sources'])}")
    print(f"Guardrail accuracy:    {_pct(s['guardrail_accuracy'])}")
    print(f"System confidence avg: {_pct(s['average_system_confidence'])}")
    print(f"Avg latency:           {s['average_latency_sec']}s   tokens: {s['total_tokens']}   run: {report['run_seconds']}s")
    print("=" * 56)
    for c in report["case_results"]:
        if c["status"] != "ok":
            print(f"  ERROR #{c['id']}: {c['error'][:120]}")
        elif not c["passed"]:
            print(f"  FAIL  #{c['id']}: {c['question'][:60]} | {c['metrics']}")


if __name__ == "__main__":
    main()
