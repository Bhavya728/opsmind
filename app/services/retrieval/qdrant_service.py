"""
Qdrant Knowledge Retrieval Service for Incident Investigation.
Stores and searches Runbooks, Postmortems, and SOPs with Qdrant vector database.
Supports:
1. Cloud Qdrant Cluster (when URL & API key are configured)
2. Native In-Memory QdrantClient (when qdrant-client package is installed)
3. High-Performance Embedded In-Memory Vector Store Fallback (zero external dependencies)
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from app.utils.logger import logfire
from app.config import settings
from app.services.retrieval.embedding import embed_query, embed_texts
from app.services.retrieval.ranking_service import rerank_documents

COLLECTION_NAME = "incident_knowledge_base"


# --- Fallback In-Memory Vector Store implementation ---
class FallbackPoint:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = np.array(vector, dtype=np.float32)
        self.payload = payload
        self.score = 0.0


class FallbackQueryResult:
    def __init__(self, points):
        self.points = points


class FallbackVectorDB:
    def __init__(self):
        self.collections = {}

    def get_collections(self):
        class ColObj:
            def __init__(self, name):
                self.name = name
        class CollList:
            def __init__(self, names):
                self.collections = [ColObj(n) for n in names]
        return CollList(list(self.collections.keys()))

    def recreate_collection(self, collection_name, vectors_config=None):
        self.collections[collection_name] = []

    def upsert(self, collection_name, points):
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        self.collections[collection_name].extend(points)

    def query_points(self, collection_name, query, query_filter=None, limit=5, with_payload=True):
        if collection_name not in self.collections:
            return FallbackQueryResult([])
            
        q_vec = np.array(query, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_points = []
        points = self.collections[collection_name]

        for p in points:
            # Check filter if any
            if query_filter and hasattr(query_filter, "should"):
                svc_val = p.payload.get("service", "")
                # Simple filter matching
                # pass all unless restricted
            
            p_vec = p.vector
            p_norm = np.linalg.norm(p_vec)
            if p_norm > 0:
                p_vec = p_vec / p_norm
            score = float(np.dot(q_vec, p_vec))
            
            res_point = FallbackPoint(p.id, p.vector, p.payload)
            res_point.score = score
            scored_points.append(res_point)

        scored_points.sort(key=lambda x: x.score, reverse=True)
        return FallbackQueryResult(scored_points[:limit])


# Try importing real QdrantClient
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    _QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    models = None
    _QDRANT_AVAILABLE = False


_client = None
_is_initialized = False


def get_qdrant_client():
    global _client
    if _client is not None:
        return _client

    if _QDRANT_AVAILABLE and settings.QDRANT_URL and settings.QDRANT_API_KEY:
        try:
            _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            logfire.info("✅ Connected to Cloud Qdrant cluster")
            return _client
        except Exception as e:
            logfire.warning(f"⚠️ Cloud Qdrant unavailable: {e}. Falling back to in-memory.")

    if _QDRANT_AVAILABLE:
        try:
            _client = QdrantClient(":memory:")
            logfire.info("🧠 Initialized in-memory QdrantClient")
            return _client
        except Exception as e:
            logfire.warning(f"⚠️ In-memory QdrantClient error: {e}. Using fallback vector store.")

    _client = FallbackVectorDB()
    logfire.info("🧠 Initialized High-Performance Embedded Vector Store")
    return _client


def initialize_incident_knowledge_base(force_reload: bool = False):
    """
    Ensures the incident knowledge collection is created and populated with:
    - Runbooks
    - Historical postmortems
    """
    global _is_initialized
    if _is_initialized and not force_reload:
        return

    client = get_qdrant_client()
    
    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections or force_reload:
        if _QDRANT_AVAILABLE and hasattr(models, "VectorParams"):
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
        else:
            client.recreate_collection(collection_name=COLLECTION_NAME)

        logfire.info(f"📁 Created knowledge collection '{COLLECTION_NAME}'")

        # Load knowledge items from data storage
        storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "storage")
        
        runbooks = []
        rb_path = os.path.join(storage_dir, "runbooks.json")
        if os.path.exists(rb_path):
            with open(rb_path, "r", encoding="utf-8") as f:
                runbooks = json.load(f)
                
        incidents = []
        inc_path = os.path.join(storage_dir, "historical_incidents.json")
        if os.path.exists(inc_path):
            with open(inc_path, "r", encoding="utf-8") as f:
                incidents = json.load(f)
                
        points = []
        idx = 1
        
        # Ingest Runbooks
        for rb in runbooks:
            doc_text = f"RUNBOOK {rb['id']}: {rb['title']} (Service: {rb.get('service', 'generic')})\nAlert Trigger: {rb.get('alert_trigger', '')}\nSymptoms: {', '.join(rb.get('symptoms', []))}\nInvestigation: {' '.join(rb.get('investigation_steps', []))}\nMitigation: {' '.join(rb.get('mitigation_steps', []))}"
            vec = embed_query(doc_text)
            
            payload = {
                "doc_id": rb["id"],
                "type": "runbook",
                "service": rb.get("service", "generic"),
                "title": rb["title"],
                "text": doc_text,
                "mitigation_steps": rb.get("mitigation_steps", []),
                "source": f"Runbook {rb['id']}"
            }
            
            if _QDRANT_AVAILABLE and hasattr(models, "PointStruct"):
                points.append(models.PointStruct(id=idx, vector=vec, payload=payload))
            else:
                points.append(FallbackPoint(id=idx, vector=vec, payload=payload))
            idx += 1
            
        # Ingest Historical Incidents
        for inc in incidents:
            doc_text = f"POSTMORTEM {inc['incident_id']}: {inc['title']} (Service: {inc.get('service', 'generic')})\nRoot Cause: {inc.get('root_cause', '')}\nResolution: {inc.get('resolution', '')}\nLessons Learned: {inc.get('lessons_learned', '')}"
            vec = embed_query(doc_text)
            
            payload = {
                "doc_id": inc["incident_id"],
                "type": "postmortem",
                "service": inc.get("service", "generic"),
                "title": inc["title"],
                "text": doc_text,
                "resolution": inc.get("resolution", ""),
                "source": f"Postmortem {inc['incident_id']}"
            }
            
            if _QDRANT_AVAILABLE and hasattr(models, "PointStruct"):
                points.append(models.PointStruct(id=idx, vector=vec, payload=payload))
            else:
                points.append(FallbackPoint(id=idx, vector=vec, payload=payload))
            idx += 1
            
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            logfire.info(f"✅ Ingested {len(points)} knowledge documents into collection '{COLLECTION_NAME}'")
            
    _is_initialized = True


def search_incident_knowledge(
    query: str,
    service: Optional[str] = None,
    limit: int = 4,
    use_reranker: bool = True
) -> List[Dict[str, Any]]:
    """
    High precision semantic retrieval for incident runbooks and past postmortems.
    Applies vector cosine similarity + FlashRank cross-encoder reranking.
    """
    initialize_incident_knowledge_base()
    client = get_qdrant_client()
    
    try:
        query_vector = embed_query(query)
        
        query_filter = None
        if service and service.strip() and _QDRANT_AVAILABLE and hasattr(models, "Filter"):
            svc_clean = service.strip().lower()
            query_filter = models.Filter(
                should=[
                    models.FieldCondition(key="service", match=models.MatchValue(value=svc_clean)),
                    models.FieldCondition(key="service", match=models.MatchValue(value="generic"))
                ]
            )
            
        search_res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit * 2,
            with_payload=True
        )
        
        candidates = []
        for p in search_res.points:
            candidates.append({
                "doc_id": p.payload.get("doc_id", "DOC"),
                "type": p.payload.get("type", "knowledge"),
                "title": p.payload.get("title", ""),
                "service": p.payload.get("service", ""),
                "content": p.payload.get("text", ""),
                "source": p.payload.get("source", "Unknown"),
                "score": float(p.score)
            })
            
        if not candidates:
            return []
            
        if use_reranker and len(candidates) > 1:
            passages = [c["content"] for c in candidates]
            reranked_texts = rerank_documents(query=query, documents=passages, top_n=limit)
            
            final_results = []
            for r_text in reranked_texts:
                for c in candidates:
                    if c["content"] == r_text and c not in final_results:
                        final_results.append(c)
                        break
            return final_results[:limit]
            
        return candidates[:limit]

    except Exception as e:
        logfire.error(f"❌ Knowledge Search Failed: {e}")
        return []
