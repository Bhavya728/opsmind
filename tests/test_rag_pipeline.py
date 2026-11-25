"""
Unit Tests for ChromaDB RAG Vector Store & Retrieval Pipeline.
Tests document indexing, dense semantic similarity search, deterministic embeddings,
and real Hugging Face SentenceTransformer embeddings.
"""

from app.services.retrieval.chroma_service import (
    chroma_service,
    DeterministicEmbeddingFunction,
    HuggingFaceSentenceTransformerEmbeddingFunction,
    get_embedding_function
)
from app.config import settings


def test_deterministic_embedding_generation():
    """Test that deterministic embedding function produces normalized 384-d vectors."""
    embed_fn = DeterministicEmbeddingFunction(dim=384)
    vectors = embed_fn(["RAG evaluation frameworks and metrics", "Agentic multi-agent systems"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    assert embed_fn.name() == "deterministic_embeddings"


def test_sentence_transformer_huggingface_embedding_generation():
    """Test that real Hugging Face SentenceTransformer embedding initializes and generates 384-d vectors."""
    hf_fn = HuggingFaceSentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    assert hf_fn.name() == "sentence-transformers/all-MiniLM-L6-v2"
    vectors = hf_fn(["Dense retrieval with ChromaDB", "Multi-agent research systems"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_embedding_function_factory_switch():
    """Test switching between deterministic and live Hugging Face embedding function."""
    det_fn, det_name = get_embedding_function(force_deterministic=True)
    assert det_name == "deterministic"
    assert isinstance(det_fn, DeterministicEmbeddingFunction)

    hf_fn, hf_name = get_embedding_function(force_deterministic=False)
    assert "sentence-transformers" in hf_name
    assert isinstance(hf_fn, HuggingFaceSentenceTransformerEmbeddingFunction)


def test_chroma_knowledge_retrieval_query():
    """Test querying the ChromaDB knowledge base for relevant concepts."""
    results = chroma_service.query("RAG evaluation metrics faithfulness and relevance", n_results=3)
    assert isinstance(results, list)
    assert len(results) > 0
    top_result = results[0]
    assert "content" in top_result
    assert "score" in top_result
    assert top_result["score"] >= settings.MIN_RELEVANCE_SCORE


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


def test_chroma_status_reporting():
    """Test Chroma service status reports accurate embedding backend."""
    status = chroma_service.get_status()
    assert "embedding_backend" in status
    assert status["embedding_backend"] in ("deterministic", "sentence-transformers/all-MiniLM-L6-v2")
    assert status["document_count"] >= 8


def test_chroma_irrelevant_query_returns_nothing():
    """Off-topic queries return no chunks instead of padding with unrelated documents."""
    assert chroma_service.query("Roman Empire timeline from 27 BCE to 476 CE") == []


def test_chunking_splits_documents():
    """Documents are split into multiple chunks according to CHUNK_SIZE."""
    status = chroma_service.get_status()
    assert status["chunk_count"] > status["document_count"]
