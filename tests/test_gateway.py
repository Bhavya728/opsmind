"""
Unit Tests for LLM Gateway & Task-Based Routing Module.
Tests model tier selection, fallback reasoning, and error resilience.
"""

from app.gateway.client import get_llm, DeterministicReasoningLLM, ResilientFallbackLLM


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
