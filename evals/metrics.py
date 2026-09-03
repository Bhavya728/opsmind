"""
Evaluation Metrics for OpsMind Research & Knowledge System.

Clearly separated into:
A. RAGAS Metrics (faithfulness, answer_relevancy, context_precision, context_recall) from official 'ragas' package.
B. Custom Heuristic Metrics (lexical/overlap-based scoring for fast deterministic evaluation & regression tests).
C. Tool Selection Accuracy (Jaccard similarity between expected and invoked tools).
D. Guardrail Decision Accuracy (binary precision on input/tool safety gates).
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


# =====================================================================
# B. CUSTOM HEURISTIC METRICS (Deterministic Lexical / Overlap Scoring)
# =====================================================================

def custom_faithfulness_heuristic(answer: str, context_chunks: List[str]) -> float:
    """
    Custom Heuristic: Measures whether lexical claims in answer overlap with retrieved context chunks.
    Formula: (Supported Sentences) / (Total Testable Sentences).
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


def custom_answer_relevance_heuristic(question: str, answer: str) -> float:
    """
    Custom Heuristic: Measures query-answer keyword alignment and structural completeness.
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


def custom_context_relevance_heuristic(question: str, retrieved_chunks: List[str]) -> float:
    """
    Custom Heuristic: Measures keyword density in retrieved context chunks relative to query.
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
    return round(min(1.0, avg_score * 1.2), 3)


def custom_context_recall_heuristic(expected_evidence: List[str], retrieved_chunks: List[str]) -> float:
    """
    Custom Heuristic: Measures whether gold reference facts appear in retrieved context chunks.
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


# =====================================================================
# C. TOOL SELECTION & ARGUMENT METRICS
# =====================================================================

def custom_tool_selection_accuracy(expected_tools: List[str], actual_tools: List[str]) -> float:
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


# =====================================================================
# D. GUARDRAIL & SAFETY METRICS
# =====================================================================

def custom_guardrail_accuracy(expected_blocked: bool, actual_blocked: bool) -> float:
    """
    Scores guardrail binary decision accuracy (1.0 for correct pass/block, 0.0 for failure).
    """
    return 1.0 if expected_blocked == actual_blocked else 0.0


def custom_citation_compliance(answer: str) -> float:
    """
    Verifies that the generated research answer contains explicit source citations.
    """
    if not answer or len(answer) < 50:
        return 0.0
    citations = re.findall(r"\[(?:Source|doc-|https?://)[^\]]+\]", answer)
    return 1.0 if len(citations) >= 2 else (0.5 if len(citations) == 1 else 0.0)


# Backward compatibility aliases
calculate_faithfulness = custom_faithfulness_heuristic
calculate_answer_relevance = custom_answer_relevance_heuristic
calculate_context_relevance = custom_context_relevance_heuristic
calculate_context_recall = custom_context_recall_heuristic
evaluate_tool_selection = custom_tool_selection_accuracy
evaluate_guardrail_decision = custom_guardrail_accuracy
evaluate_citations_presence = custom_citation_compliance
