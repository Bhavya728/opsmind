"""
Unit Tests for LLM Gateway & Task-Based Routing Module.
Tests model tier selection, fallback reasoning, runtime status, and error resilience.
"""

from app.gateway.client import get_llm, DeterministicReasoningLLM, ResilientFallbackLLM, get_llm_runtime_status


def test_gateway_planner_model_tier():
    """Test that planning/classification routes to lightweight fast model tier."""
    llm = get_llm(task_type="planning")
    assert llm is not None
    response = llm.invoke("Plan research for RAG evaluation architectures")
    content = response.content if hasattr(response, "content") else str(response)
    assert "need_rag" in content or "plan" in content


def test_gateway_synthesis_model_tier():
    """Test that report synthesis routes to strong reasoning model tier."""
    llm = get_llm(task_type="synthesis")
    assert llm is not None
    response = llm.invoke("Synthesize report for Agentic RAG comparison")
    content = response.content if hasattr(response, "content") else str(response)
    assert "Executive Summary" in content or "Architecture" in content or "Source" in content


def test_gateway_critic_model_tier():
    """Test that critic task routes properly and produces confidence evaluation."""
    llm = get_llm(task_type="critic")
    assert llm is not None
    response = llm.invoke("Evaluate evidence for vector search HNSW algorithms")
    content = response.content if hasattr(response, "content") else str(response)
    assert "is_grounded" in content or "confidence" in content


def test_deterministic_reasoning_fallback_engine():
    """Test deterministic offline engine execution."""
    engine = DeterministicReasoningLLM(task_type="synthesis")
    res = engine.invoke("What are LLM gateways?")
    assert len(res.content) > 100
    assert "LLM" in res.content


def test_gateway_runtime_status():
    """Test gateway status reports accurate active backend."""
    status = get_llm_runtime_status()
    assert status in ("deterministic", "groq", "portkey")


def test_rate_limited_primary_falls_back_to_next_model():
    """A 429 on the primary model is retried on the next configured model and marked in the response."""
    from app.agents.llm_json import fallback_note

    class RateLimited(Exception):
        status_code = 429

    class AlwaysRateLimited(DeterministicReasoningLLM):
        def _generate(self, messages, stop=None, **kwargs):
            raise RateLimited("tokens per day (TPD) exhausted")

    llm = ResilientFallbackLLM(
        primary_llm=AlwaysRateLimited(task_type="planning"),
        fallback_llm=DeterministicReasoningLLM(task_type="planning"),
        rate_limit_fallbacks=[("backup-a", AlwaysRateLimited(task_type="planning")),
                              ("backup-b", DeterministicReasoningLLM(task_type="planning"))],
        task_type="planning", active_backend="groq",
    )
    response = llm.invoke('Research Query: "What is RAG?"')
    assert "need_rag" in response.content
    assert response.response_metadata["fallback_model"] == "backup-b"
    assert "backup-b" in fallback_note(response, "Query Planner")


def test_non_rate_limit_errors_are_not_retried_on_other_models():
    class Broken(DeterministicReasoningLLM):
        def _generate(self, messages, stop=None, **kwargs):
            raise ValueError("bad request")

    llm = ResilientFallbackLLM(primary_llm=Broken(), fallback_llm=DeterministicReasoningLLM(),
                               rate_limit_fallbacks=[("backup", DeterministicReasoningLLM())], active_backend="groq")
    try:
        llm.invoke("hi")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
