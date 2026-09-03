"""
Comprehensive LLM Evaluation Metrics for OpsMind Research & Knowledge System.

Clearly separated into three distinct evaluation dimensions:
1. RAG Evaluation Metrics (Faithfulness, Answer Relevance, Context Relevance, Context Recall)
2. Agent & Tool Metrics (Tool Selection Correctness, Tool Argument Correctness)
3. Guardrail & Safety Metrics (Input Safety Accuracy, Output Redaction Accuracy, Grounded Citation Rate)
"""

import re
import math
from typing import List, Dict, Any, Optional

STOP_WORDS = {
    "what", "when", "where", "which", "could", "should", "would", "how", "why",
    "explain", "compare", "major", "different", "perform", "system", "please",
    "about", "their", "there", "these", "those", "using", "used", "does", "with",
    "from", "into", "over", "between", "both", "some", "other", "such", "than"
}


def _extract_keywords(text: str) -> set:
    words = set(re.findall(r"\w+", text.lower()))
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS}


# ==========================================
# 1. RAG EVALUATION METRICS
# ==========================================

def calculate_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Measures whether claims in the generated answer are grounded in retrieved context chunks.
    Formula: (Supported Claims) / (Total Testable Claims).
    """
    if not answer or not context_chunks:
        return 1.0 if not answer else 0.0

    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    combined_context = " ".join(context_chunks).lower()
    supported_count = 0

    for sent in sentences:
        content_words = _extract_keywords(sent)
        if not content_words:
            supported_count += 1
            continue
        
        overlap = sum(1 for w in content_words if w in combined_context)
        overlap_ratio = overlap / len(content_words)
        if overlap_ratio >= 0.35:
            supported_count += 1

    return round(min(1.0, supported_count / len(sentences)), 3)


def calculate_answer_relevance(question: str, answer: str) -> float:
    """
    Measures semantic and topical alignment between generated answer and user query.
    """
    if not answer:
        return 0.0
    q_content = _extract_keywords(question)
    if not q_content:
        return 1.0

    ans_lower = answer.lower()
    matches = sum(1 for w in q_content if w in ans_lower)
    score = matches / len(q_content)
    
    if "###" in answer or "**" in answer:
        score = min(1.0, score + 0.10)
        
    return round(min(1.0, max(0.0, score)), 3)


def calculate_context_relevance(question: str, retrieved_chunks: List[str]) -> float:
    """
    Measures the signal-to-noise ratio in retrieved context chunks relative to the query keywords.
    """
    if not retrieved_chunks:
        return 0.0
        
    q_content = _extract_keywords(question)
    if not q_content:
        return 1.0

    chunk_scores = []
    for chunk in retrieved_chunks:
        chunk_lower = chunk.lower()
        overlap = sum(1 for w in q_content if w in chunk_lower)
        chunk_scores.append(min(1.0, overlap / max(1, len(q_content))))

    avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    # Normalize score
    return round(min(1.0, avg_score * 1.2), 3)


def calculate_context_recall(expected_evidence: List[str], retrieved_chunks: List[str]) -> float:
    """
    Measures whether all gold expected evidence items were captured by retrieval.
    """
    if not expected_evidence:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    combined_context = " ".join(retrieved_chunks).lower()
    matched = 0

    for ev in expected_evidence:
        ev_words = _extract_keywords(ev)
        if not ev_words:
            matched += 1
            continue
        overlap = sum(1 for w in ev_words if w in combined_context)
        if (overlap / len(ev_words)) >= 0.35:
            matched += 1

    return round(matched / len(expected_evidence), 3)


# ==========================================
# 2. AGENT & TOOL EVALUATION METRICS
# ==========================================

def evaluate_tool_selection(expected_tools: List[str], actual_tools: List[str]) -> float:
    """
    Measures Jaccard similarity between expected tools and actual tools invoked by agents.
    """
    if not expected_tools and not actual_tools:
        return 1.0
    if not expected_tools or not actual_tools:
        return 0.5 if not expected_tools else 0.0

    set_exp = set(expected_tools)
    set_act = set(actual_tools)
    
    intersection = set_exp & set_act
    union = set_exp | set_act
    
    return round(len(intersection) / len(union), 3)


def evaluate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> bool:
    """
    Validates whether tool arguments satisfy schema and safety rules.
    """
    from app.guardrails.rails import guard_tool_call
    is_valid, _ = guard_tool_call(tool_name, args)
    return is_valid


# ==========================================
# 3. GUARDRAIL & SAFETY METRICS
# ==========================================

def evaluate_guardrail_decision(expected_blocked: bool, actual_blocked: bool) -> float:
    """
    Scores guardrail binary decision accuracy (1.0 for correct pass/block, 0.0 for failure).
    """
    return 1.0 if expected_blocked == actual_blocked else 0.0


def evaluate_citations_presence(answer: str) -> float:
    """
    Verifies that the generated research answer contains explicit source citations.
    """
    if not answer or len(answer) < 50:
        return 0.0
    citations = re.findall(r"\[(?:Source|doc-|https?://)[^\]]+\]", answer)
    return 1.0 if len(citations) >= 2 else (0.5 if len(citations) == 1 else 0.0)
