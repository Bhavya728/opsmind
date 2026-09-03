"""
Standalone Test Runner for OpsMind Agentic Research & Knowledge System.
Executes all unit, integration, guardrail, gateway, MCP, RAG, evaluation, and API tests.
"""

import sys
import os
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_tests():
    print("=" * 70)
    print("🚀 RUNNING OPSMIND RESEARCH & KNOWLEDGE SYSTEM TEST SUITE")
    print("=" * 70)

    test_modules = [
        ("Runtime Guardrails & Safety", "tests.test_guardrails"),
        ("LLM Gateway & Task-Based Routing", "tests.test_gateway"),
        ("MCP Tools & Protocol Discovery", "tests.test_mcp_tools"),
        ("ChromaDB RAG Pipeline & Embeddings", "tests.test_rag_pipeline"),
        ("LangGraph 5-Agent Research Workflow", "tests.test_agent_graph"),
        ("RAGAS Evaluation & Custom Metrics", "tests.test_evals_ragas"),
        ("FastAPI Backend Endpoints", "tests.test_api"),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for category, mod_name in test_modules:
        print(f"\n▶ Testing Module: [{category}] ({mod_name})")
        try:
            mod = __import__(mod_name, fromlist=["*"])
            test_funcs = [getattr(mod, f) for f in dir(mod) if f.startswith("test_") and callable(getattr(mod, f))]
            
            for tf in test_funcs:
                test_name = tf.__name__
                try:
                    tf()
                    print(f"  ✅ PASS: {test_name}", flush=True)
                    total_passed += 1
                except Exception as e:
                    print(f"  ❌ FAIL: {test_name} -> {e}", flush=True)
                    total_failed += 1
                    failures.append((category, test_name, str(e), traceback.format_exc()))
        except Exception as e:
            print(f"  ❌ MODULE IMPORT/EXEC ERROR: {mod_name} -> {e}", flush=True)
            total_failed += 1
            failures.append((category, mod_name, str(e), traceback.format_exc()))

    print("\n" + "=" * 70, flush=True)
    print(f"📊 TEST SUITE SUMMARY: {total_passed} Passed | {total_failed} Failed | Total: {total_passed + total_failed}", flush=True)
    print("=" * 70, flush=True)

    if failures:
        print("\n❌ FAILURE DETAILS:")
        for cat, name, err, tb in failures:
            print(f"\n--- [{cat}] {name} ---")
            print(f"Error: {err}")
            print(tb)
        sys.exit(1)
    else:
        print("\n🎉 ALL UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
