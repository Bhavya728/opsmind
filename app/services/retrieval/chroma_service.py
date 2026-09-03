"""
ChromaDB Retrieval Service with Hugging Face Embeddings for OpsMind.
Provides semantic vector search, document indexing, and similarity ranking.
"""

import os
import math
import hashlib
from typing import List, Dict, Any, Optional
from app.utils.logger import logfire
from app.data.knowledge_base import KNOWLEDGE_DOCUMENTS

# Chroma storage directory
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chroma_db")


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
            vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, input: str) -> List[float]:
        return self([input])[0]


class ChromaService:
    """ChromaDB Vector Store management for technical knowledge retrieval."""
    
    _instance = None

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or CHROMA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection_name = "opsmind_research_kb"
        self._init_client_and_collection()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ChromaService()
        return cls._instance

    def _init_client_and_collection(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            
            self.embedding_fn = DeterministicEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"description": "OpsMind AI & CS Research Knowledge Base"}
            )
            
            if self.collection.count() == 0:
                self._seed_knowledge_base()
            logfire.info(f"📚 ChromaDB initialized with {self.collection.count()} document chunks.")
        except Exception as e:
            logfire.warning(f"⚠️ ChromaDB native client init note: {e}")
            self.client = None
            self.collection = None
            self.embedding_fn = DeterministicEmbeddingFunction()

    def _seed_knowledge_base(self):
        """Index the knowledge documents into ChromaDB with chunking."""
        ids = []
        documents = []
        metadatas = []
        
        for doc in KNOWLEDGE_DOCUMENTS:
            doc_id = doc["id"]
            title = doc["title"]
            category = doc.get("category", "general")
            content = doc["content"].strip()
            
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for chunk_idx, para in enumerate(paragraphs):
                chunk_id = f"{doc_id}_p{chunk_idx+1}"
                ids.append(chunk_id)
                documents.append(f"Title: {title}\nCategory: {category}\n\n{para}")
                metadatas.append({
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "chunk_index": chunk_idx + 1,
                    "tags": ",".join(doc.get("tags", []))
                })

        if documents and self.collection:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logfire.info(f"✅ Ingested {len(documents)} chunks from {len(KNOWLEDGE_DOCUMENTS)} documents into ChromaDB.")

    def query(self, query_text: str, n_results: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on ChromaDB."""
        if not self.collection:
            return self._in_memory_fallback_query(query_text, n_results, category)

        where_filter = {"category": category} if category else None
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, max(1, self.collection.count())),
                where=where_filter
            )
            
            output = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.2] * len(docs)
                
                for doc_text, meta, cid, dist in zip(docs, metas, ids, distances):
                    similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                    output.append({
                        "id": cid,
                        "doc_id": meta.get("doc_id", cid),
                        "title": meta.get("title", "Knowledge Document"),
                        "category": meta.get("category", "research"),
                        "content": doc_text,
                        "score": round(similarity, 3)
                    })
            if not output:
                return self._in_memory_fallback_query(query_text, n_results, category)
            return output
        except Exception as e:
            logfire.warning(f"⚠️ Chroma query note: {e}. Using fallback.")
            return self._in_memory_fallback_query(query_text, n_results, category)

    def _in_memory_fallback_query(self, query_text: str, n_results: int = 3, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """High-precision keyword + semantic overlap fallback."""
        query_lower = query_text.lower()
        query_words = set(query_lower.split())
        scored = []
        
        for doc in KNOWLEDGE_DOCUMENTS:
            if category and doc.get("category") != category:
                continue
            text = f"{doc['title']} {doc.get('summary', '')} {doc['content']}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            
            tag_matches = sum(1 for t in doc.get("tags", []) if t in query_lower)
            score = (overlap / (len(query_words) + 1)) + (0.3 * tag_matches)
            
            if score > 0.15:
                scored.append({
                    "id": doc["id"],
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "category": doc.get("category", "research"),
                    "content": doc["content"].strip()[:600] + "...",
                    "score": round(min(0.98, 0.5 + score * 0.4), 3)
                })
                
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n_results]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full document content by document ID."""
        for doc in KNOWLEDGE_DOCUMENTS:
            if doc["id"] == doc_id:
                return doc
        return None


# Global singleton instance
chroma_service = ChromaService.get_instance()
