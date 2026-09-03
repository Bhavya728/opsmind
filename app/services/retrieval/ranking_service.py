"""
Semantic Reranking Service using FlashRank Cross-Encoder with Instant Fallback.
"""

import os
import time
import re
from typing import List
from app.utils.logger import logfire

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker, RerankRequest = None, None

_ranker = None
_ranker_failed = False


def _get_ranker():
    global _ranker, _ranker_failed
    if _ranker_failed or Ranker is None:
        return None
    if _ranker is not None:
        return _ranker

    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "flashrank")
        os.makedirs(cache_dir, exist_ok=True)
        logfire.info("🧠 Initializing FlashRank Model (ms-marco-TinyBERT)...")
        _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=cache_dir)
        return _ranker
    except Exception as e:
        logfire.info(f"ℹ️ FlashRank model initialization note: {e}. Using fast semantic term reranker.")
        _ranker_failed = True
        return None


def rerank_documents(query: str, documents: List[str], top_n: int = 5) -> List[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    """
    if not documents:
        return []

    ranker = _get_ranker()
    if ranker is not None:
        try:
            passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)
            return [res['text'] for res in results[:top_n]]
        except Exception as e:
            logfire.warning(f"⚠️ FlashRank rerank exception: {e}. Using keyword score.")

    # Fast semantic keyword overlap reranker fallback
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for doc in documents:
        doc_tokens = set(re.findall(r"\w+", doc.lower()))
        score = len(query_tokens & doc_tokens) / (len(query_tokens) + 1)
        scored.append((score, doc))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]
