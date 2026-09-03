"""
RAG Pipeline and Qdrant Retrieval Tests.
"""

try:
    import pytest
except ImportError:
    from tests import compat as pytest
from app.services.retrieval.qdrant_service import (
    initialize_incident_knowledge_base,
    search_incident_knowledge,
    get_qdrant_client
)
from app.services.retrieval.embedding import embed_query


def test_qdrant_initialization():
    initialize_incident_knowledge_base(force_reload=False)
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    assert "incident_knowledge_base" in collections


def test_embedding_generation():
    vec = embed_query("PostgreSQL connection pool exhaustion")
    assert isinstance(vec, list)
    assert len(vec) == 768


def test_search_incident_knowledge_service_filter():
    results = search_incident_knowledge("connection pool timeout Hikari", service="order-service", limit=3)
    assert len(results) > 0
    first = results[0]
    assert "content" in first
    assert "source" in first
    assert "doc_id" in first


def test_search_incident_knowledge_postmortem_retrieval():
    results = search_incident_knowledge("unindexed query audit logs", service="payment-service", limit=3)
    assert len(results) > 0
    assert any("payment" in r["content"].lower() or "audit" in r["content"].lower() for r in results)
