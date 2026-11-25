"""
ChromaDB Retrieval Service with Hugging Face Embeddings for OpsMind.
Provides semantic vector search, document indexing, and similarity ranking.
Supports both real Hugging Face SentenceTransformer embeddings and deterministic offline embeddings.

The corpus is built dynamically from the built-in documents plus any .md/.txt files in
settings.KNOWLEDGE_DIR. The index is rebuilt automatically whenever the corpus, the chunking
settings, or the embedding backend change.
"""

import os
import re
import math
import hashlib
import json
from typing import List, Dict, Any, Optional
from app.config import settings
from app.utils.logger import logfire
from app.data.knowledge_base import KNOWLEDGE_DOCUMENTS

STOP_WORDS = {
    "the", "and", "for", "are", "with", "what", "how", "why", "does", "this", "that", "from",
    "into", "about", "between", "which", "when", "where", "explain", "compare", "describe",
    "their", "there", "these", "those", "have", "has", "was", "were", "will", "can", "you",
    "your", "use", "used", "using", "its", "also", "than", "then", "them", "they", "our",
}


def _terms(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-\+\.#]*", text.lower()) if len(w) > 2 and w not in STOP_WORDS}


class DeterministicEmbeddingFunction:
    """
    Deterministic 384-dimensional embedding function for instant offline/local execution.
    Provides fast, reproducible semantic vector generation conforming to ChromaDB EmbeddingFunction protocol.
    """
    def name(self) -> str:
        return "deterministic_embeddings"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            tokens = str(text).lower().split()
            vec = [0.0] * self.dim
            for idx, token in enumerate(tokens):
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                pos = h % self.dim
                val = ((h >> 8) % 1000) / 1000.0 - 0.5
                vec[pos] += val * (1.0 / (1.0 + math.log(idx + 1)))

            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [float(x / norm) for x in vec]
            embeddings.append(vec)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, input: str) -> List[float]:
        return self([input])[0]


class HuggingFaceSentenceTransformerEmbeddingFunction:
    """
    Real Hugging Face SentenceTransformer embedding function (default all-MiniLM-L6-v2).
    Produces dense normalized semantic embeddings.
    """
    def name(self) -> str:
        return self.model_name if "/" in self.model_name else f"sentence-transformers/{self.model_name}"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logfire.info(f"🤗 Hugging Face SentenceTransformer initialized with model: '{self.model_name}'")
        except Exception as e:
            logfire.error(f"❌ Failed to load SentenceTransformer model '{self.model_name}': {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        if self._model is None:
            self._init_model()

        raw_embeddings = self._model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
        return [vec.tolist() for vec in raw_embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, input: str) -> List[float]:
        return self([input])[0]


def get_embedding_function(force_deterministic: Optional[bool] = None):
    """
    Factory to return the active embedding function based on configuration.
    USE_DETERMINISTIC_MODE=true  -> DeterministicEmbeddingFunction
    USE_DETERMINISTIC_MODE=false -> HuggingFaceSentenceTransformerEmbeddingFunction(EMBEDDING_MODEL)
    """
    use_deterministic = force_deterministic if force_deterministic is not None else settings.USE_DETERMINISTIC_MODE

    if use_deterministic:
        return DeterministicEmbeddingFunction(dim=384), "deterministic"

    try:
        embed_fn = HuggingFaceSentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)
        return embed_fn, embed_fn.name()
    except Exception as e:
        logfire.warning(f"⚠️ HuggingFace SentenceTransformer unavailable ({e}). Falling back to deterministic embeddings.")
        return DeterministicEmbeddingFunction(dim=384), "deterministic"


def load_corpus() -> List[Dict[str, Any]]:
    """Build the knowledge corpus from built-in documents plus files in settings.KNOWLEDGE_DIR."""
    docs: List[Dict[str, Any]] = []
    if settings.INCLUDE_BUILTIN_DOCUMENTS:
        docs.extend({**d, "content": d["content"].strip(), "source": "builtin"} for d in KNOWLEDGE_DOCUMENTS)

    kb_dir = settings.KNOWLEDGE_DIR
    if kb_dir and os.path.isdir(kb_dir):
        for root, _, files in os.walk(kb_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".md", ".txt")):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                except OSError as e:
                    logfire.warning(f"⚠️ Could not read knowledge file '{path}': {e}")
                    continue
                if not text:
                    continue
                rel_dir = os.path.relpath(root, kb_dir)
                stem = os.path.splitext(fname)[0]
                heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                docs.append({
                    "id": re.sub(r"[^a-zA-Z0-9_\-]+", "-", stem).strip("-").lower(),
                    "title": heading.group(1).strip() if heading else stem.replace("_", " ").replace("-", " ").title(),
                    "category": rel_dir.split(os.sep)[0].lower() if rel_dir != "." else "general",
                    "tags": [],
                    "summary": text.split("\n\n")[0][:300],
                    "content": text,
                    "source": os.path.relpath(path, kb_dir),
                })
    return docs


def chunk_document(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text on line boundaries into ~chunk_size character windows with ~overlap characters carried over."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) > chunk_size:
            chunks.append("\n".join(current))
            carry: List[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev)
            current, length = carry, carry_len
        current.append(line)
        length += len(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


class ChromaService:
    """ChromaDB Vector Store management for technical knowledge retrieval."""

    _instance = None

    def __init__(self, persist_directory: Optional[str] = None, force_deterministic: Optional[bool] = None):
        self.persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = settings.CHROMA_COLLECTION
        self.embedding_fn, self.embedding_backend = get_embedding_function(force_deterministic)
        self.documents = load_corpus()
        self._doc_index = {d["id"]: d for d in self.documents}
        self.client = None
        self.collection = None
        self._init_client_and_collection()

    @classmethod
    def get_instance(cls, force_deterministic: Optional[bool] = None):
        if cls._instance is None:
            cls._instance = ChromaService(force_deterministic=force_deterministic)
        return cls._instance

    def _fingerprint(self) -> str:
        payload = json.dumps({
            "docs": [[d["id"], d["title"], d.get("category"), d["content"]] for d in self.documents],
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding": self.embedding_backend,
            "index_format": 2,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _init_client_and_collection(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
            )
            fingerprint = self._fingerprint()
            existing = None
            try:
                existing = self.client.get_collection(name=self.collection_name)
            except Exception:
                existing = None

            if existing is not None and (existing.metadata or {}).get("fingerprint") == fingerprint:
                self.collection = existing
            else:
                if existing is not None:
                    logfire.info("♻️ Knowledge corpus, chunking or embedding backend changed. Rebuilding ChromaDB index.")
                    self.client.delete_collection(self.collection_name)
                # Embeddings are computed by this service, so Chroma stores vectors without its own embedding function.
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={
                        "description": "OpsMind Research Knowledge Base",
                        "fingerprint": fingerprint,
                        "embedding_backend": self.embedding_backend,
                        "hnsw:space": "cosine",
                    },
                )
                self._seed_knowledge_base()
            logfire.info(f"📚 ChromaDB ready: {self.collection.count()} chunks from {len(self.documents)} documents. Embeddings: {self.embedding_backend}")
        except Exception as e:
            logfire.warning(f"⚠️ ChromaDB unavailable ({e}). Using in-memory lexical retrieval.")
            self.client = None
            self.collection = None

    def _seed_knowledge_base(self):
        """Chunk, embed and index every corpus document."""
        ids, documents, embed_texts, metadatas = [], [], [], []
        for doc in self.documents:
            category = doc.get("category", "general")
            for idx, chunk in enumerate(chunk_document(doc["content"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
                ids.append(f"{doc['id']}_c{idx + 1}")
                documents.append(chunk)
                # Title and category are embedded with the chunk so each vector carries its document's context.
                embed_texts.append(f"Title: {doc['title']}\nCategory: {category}\n\n{chunk}")
                metadatas.append({
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "category": category,
                    "chunk_index": idx + 1,
                    "tags": ",".join(doc.get("tags", [])),
                })
        if ids:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=self.embedding_fn(embed_texts))
            logfire.info(f"✅ Indexed {len(ids)} chunks from {len(self.documents)} documents into ChromaDB.")

    @staticmethod
    def _coverage(query_text: str, doc_text: str) -> float:
        """Share of the query's content terms that appear in the chunk."""
        q_terms = _terms(query_text)
        return len(q_terms & _terms(doc_text)) / len(q_terms) if q_terms else 0.0

    def _relevance(self, query_text: str, doc_text: str, vector_similarity: float) -> float:
        """Hybrid relevance: weighted vector similarity plus query-term coverage."""
        w = settings.HYBRID_VECTOR_WEIGHT
        score = w * vector_similarity + (1 - w) * self._coverage(query_text, doc_text)
        return round(max(0.0, min(1.0, score)), 3)

    def query(self, query_text: str, n_results: Optional[int] = None, category: Optional[str] = None,
              min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """Semantic search; returns only chunks whose relevance is at least min_score (default MIN_RELEVANCE_SCORE)."""
        n_results = n_results or settings.RAG_TOP_K
        threshold = settings.MIN_RELEVANCE_SCORE if min_score is None else min_score
        if not self.collection:
            return self._in_memory_fallback_query(query_text, n_results, category, threshold)

        total = self.collection.count()
        if total == 0:
            return []
        try:
            results = self.collection.query(
                query_embeddings=self.embedding_fn([query_text]),
                # Over-fetch candidates so hybrid re-ranking can promote lexically strong chunks.
                n_results=min(total, n_results * 3),
                where={"category": category} if category else None,
            )
        except Exception as e:
            logfire.warning(f"⚠️ Chroma query failed ({e}). Using in-memory lexical retrieval.")
            return self._in_memory_fallback_query(query_text, n_results, category, threshold)

        output = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        for doc_text, meta, cid, dist in zip(docs, metas, ids, distances):
            score = self._relevance(query_text, f"{meta.get('title', '')}\n{doc_text}", 1.0 - dist)
            if score < threshold:
                continue
            output.append({
                "id": cid,
                "doc_id": meta.get("doc_id", cid),
                "title": meta.get("title", ""),
                "category": meta.get("category", "general"),
                "content": doc_text,
                "score": score,
            })
        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:n_results]

    def _in_memory_fallback_query(self, query_text: str, n_results: int, category: Optional[str],
                                  threshold: float) -> List[Dict[str, Any]]:
        """Lexical retrieval over the same chunks when ChromaDB is unavailable."""
        scored = []
        for doc in self.documents:
            if category and doc.get("category") != category:
                continue
            for idx, chunk in enumerate(chunk_document(doc["content"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
                score = round(self._coverage(query_text, f"{doc['title']}\n{chunk}"), 3)
                if score >= threshold:
                    scored.append({
                        "id": f"{doc['id']}_c{idx + 1}",
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "category": doc.get("category", "general"),
                        "content": chunk,
                        "score": score,
                    })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n_results]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full document content by document ID."""
        return self._doc_index.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents)

    def categories(self) -> List[str]:
        return sorted({d.get("category", "general") for d in self.documents})

    def describe_corpus(self) -> str:
        """Short description of the knowledge base for planner prompts."""
        return "; ".join(f"{d['title']} ({d.get('category', 'general')})" for d in self.documents)

    def get_status(self) -> Dict[str, Any]:
        """Return runtime ChromaDB and embedding status."""
        return {
            "status": "connected" if self.collection else "fallback",
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "chunk_count": self.collection.count() if self.collection else None,
            "embedding_backend": self.embedding_backend,
            "persist_dir": self.persist_dir,
            "knowledge_dir": settings.KNOWLEDGE_DIR,
        }


# Global singleton instance
chroma_service = ChromaService.get_instance()
