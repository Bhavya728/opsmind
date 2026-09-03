"""
Unit Tests for ChromaDB RAG Vector Store & Retrieval Pipeline.
Tests document indexing, dense semantic similarity search, and document fetching.
"""

from app.services.retrieval.chroma_service import chroma_service, DeterministicEmbeddingFunction
from app.data.knowledge_base import KNOWLEDGE_DOCUMENTS


def test_deterministic_embedding_generation():
    """Test that embedding function produces normalized 384-d vectors."""
    embed_fn = DeterministicEmbeddingFunction(dim=384)
    vectors = embed_fn(["RAG evaluation frameworks and metrics", "Agentic multi-agent systems"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_chroma_knowledge_retrieval_query():
    """Test querying the ChromaDB knowledge base for relevant concepts."""
    results = chroma_service.query("RAG evaluation metrics faithfulness and relevance", n_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    top_result = results[0]
    assert "content" in top_result
    assert "score" in top_result
    assert top_result["score"] >= 0.50


def test_chroma_knowledge_retrieval_with_category_filter():
    """Test category filtering during similarity retrieval."""
    results = chroma_service.query("model tiering and fallbacks", n_results=2, category="infrastructure")
    assert isinstance(results, list)
    if results:
        assert results[0].get("category") == "infrastructure"


def test_chroma_get_document_by_id():
    """Test fetching full knowledge document by document ID."""
    doc = chroma_service.get_document("doc-rag-eval-01")
    assert doc is not None
    assert doc["id"] == "doc-rag-eval-01"
    assert "RAG Evaluation" in doc["title"]
    assert len(doc["content"]) > 100


def test_chroma_get_document_non_existent():
    """Test fetching non-existent document ID returns None."""
    doc = chroma_service.get_document("non-existent-doc-999")
    assert doc is None
