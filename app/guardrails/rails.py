"""
Runtime guardrails for OpsMind.

1. Input rail (guard_input): pattern checks for prompt injection, destructive commands and
   obvious chit-chat. The planner's topic check (RESEARCH_SCOPE) runs afterwards in the graph.
2. Tool rail (guard_tool_call): strict identifiers; free-text search arguments reject command
   sequences, path traversal and control characters.
3. Output rails: guard_output redacts secret-like strings; validate_citations normalizes
   citation tags and removes citations to sources that were not verified.
"""

import re
from typing import Tuple, Optional, Dict, Any, List
from app.utils.logger import logfire
from app.config import settings

# High-precision heuristic filters for deterministic instant safety enforcement
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+|prior\s+)?instructions",
    r"disregard\s+(all\s+)?(system\s+|safety\s+)?(prompts|rules|guidelines)",
    r"you\s+are\s+now\s+dan",
    r"jailbreak\s+mode",
    r"developer\s+mode\s+output",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"override\s+your\s+safety\s+filters",
    r"enter\s+developer\s+mode",
    r"bypass\s+(your\s+)?(guardrails|filters|rules)",
    r"override\s+filters"
]

MALICIOUS_PATTERNS = [
    r"rm\s+-rf\s+[/~]",
    r"drop\s+database",
    r"truncate\s+table",
    r"format\s+[c-zC-Z]:",
    r"chmod\s+777\s+/etc",
    r"kill\s+-9\s+1",
    r"curl\s+https?://[^\s]+\s*\|\s*sh",
    r"wget\s+https?://[^\s]+\s*\|\s*sh"
]

OFF_TOPIC_PATTERNS = [
    r"\b(tell\s+(me\s+)?(a\s+)?(funny\s+)?joke|joke|write\s+a\s+poem|poem)\b",
    r"\b(capital\s+of\s+[a-zA-Z\s]+|what\s+should\s+i\s+eat|eat\s+for\s+dinner|dinner\s+tonight)\b",
    r"\b(bake\s+.*cake|bake\s+a\s+cake|baking\s+a\s+cake|recipe\s+for|recommend\s+a\s+movie|weather\s+forecast)\b"
]

SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9_\-]{20,}",                # OpenAI (incl. sk-proj-...) and Anthropic (sk-ant-...)
    r"\bgsk_[A-Za-z0-9]{20,}",                  # Groq
    r"\btvly-[A-Za-z0-9_\-]{20,}",              # Tavily (incl. tvly-dev-...)
    r"\blsv2_[a-z]{2}_[A-Za-z0-9_]{20,}",       # LangSmith
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",           # AWS access key id
    r"(?:ghp_[a-zA-Z0-9]{36})",
    r"(?:Bearer\s+[a-zA-Z0-9_\-\.]{30,})",
    r"(?:password\s*[:=]\s*['\"][^'\"]{6,}['\"])",
    r"(?:API_KEY\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])"
]


def guard_input(message: str) -> Tuple[bool, Optional[str]]:
    """
    Gate 1: Input Guardrails.
    Evaluates incoming query for prompt injections, malicious commands, and off-topic requests.
    """
    clean_msg = message.strip()
    msg_lower = clean_msg.lower()

    if not settings.GUARDRAILS_ENABLED:
        return False, None

    # 1. Fast Heuristic Guard: Prompt Injections
    for pat in INJECTION_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.warning(f"🛡️ [INPUT RAIL] Prompt injection blocked: '{clean_msg[:50]}'")
            return True, "🛡️ [SECURITY VIOLATION] Prompt injection or unauthorized instruction override attempt detected. Research assistant safety policy strictly forbids guideline manipulation."

    # 2. Fast Heuristic Guard: Destructive Commands
    for pat in MALICIOUS_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.warning(f"🛡️ [INPUT RAIL] Malicious destructive command blocked: '{clean_msg[:50]}'")
            return True, "🛡️ [SECURITY VIOLATION] Destructive or unauthorized execution commands are blocked by system safety guardrails."

    # 3. Fast Heuristic Guard: Out-of-Scope Queries
    for pat in OFF_TOPIC_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.info(f"🛡️ [INPUT RAIL] Off-topic request blocked: '{clean_msg[:50]}'")
            return True, "🛡️ I am OpsMind — an Agentic AI Research & Knowledge System specializing in computer science, machine learning, RAG architectures, LLM systems, and technical knowledge synthesis. I cannot assist with non-technical or general chit-chat requests."

    logfire.info("✅ Input guardrails passed.")
    return False, None


IDENTIFIER_ARGS = {"doc_id", "category"}
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_\-.]{1,128}")
# Non-printing control bytes; tab, newline and carriage return are ordinary whitespace and allowed.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Shell chaining / substitution plus the destructive commands from the input rail.
COMMAND_PATTERNS = [r"&&", r"\|\|", r"`", r"\$\("] + MALICIOUS_PATTERNS


def guard_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Gate 2: Tool Call Guardrails.
    Identifier arguments (doc_id, category) must be plain ids. Free-text search arguments may use
    normal punctuation (; | < > etc.), but command sequences, path traversal and control
    characters are rejected.
    """
    for k, v in args.items():
        if not isinstance(v, str):
            continue
        if re.search(r"\.\./|\.\.\\|/etc/|\\windows\\", v, re.IGNORECASE):
            logfire.warning(f"🛡️ [TOOL RAIL] Path traversal detected in arg '{k}': {v[:80]}")
            return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains prohibited file traversal sequence."
        if k in IDENTIFIER_ARGS:
            if not IDENTIFIER_PATTERN.fullmatch(v):
                logfire.warning(f"🛡️ [TOOL RAIL] Invalid identifier in arg '{k}': {v[:80]}")
                return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains prohibited characters for an identifier."
            continue
        if CONTROL_CHARS.search(v):
            logfire.warning(f"🛡️ [TOOL RAIL] Control character detected in arg '{k}': {v[:80]!r}")
            return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains a non-printing control character."
        if any(re.search(p, v, re.IGNORECASE) for p in COMMAND_PATTERNS):
            logfire.warning(f"🛡️ [TOOL RAIL] Command sequence detected in arg '{k}': {v[:80]}")
            return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains a prohibited command sequence."

    logfire.info(f"✅ Tool call '{tool_name}' verified by guardrails.")
    return True, None


CITATION_PATTERN = re.compile(r"\[Sources?:\s*([^\]]+?)\s*\]")
# Bracketed groups in other styles models use for citations: [id], 【id】, (Source: id).
ALT_CITATION_PATTERN = re.compile(r"(?<!\])[\[【(]([^\[\]【】()]{1,200})[\]】)]")
# LLMs often emit typographic hyphens (U+2010-U+2015, U+2212) inside ids.
_HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


def validate_citations(text: str, allowed_ids: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Gate 3b: Citation grounding check.
    Normalizes [Source: X] and [Source: X, Y] tags to one tag per verified id and removes
    ids that are not among the verified sources.
    Returns (cleaned_text, cited_valid_ids, invalid_ids).
    """
    allowed = set(allowed_ids)
    used: List[str] = []
    invalid: List[str] = []

    def _canonicalize(match: "re.Match") -> str:
        # Rewrite [web-1], 【web‑1】, (Source: web-1, doc-x) ... to [Source: id] when every id is verified.
        body = re.sub(r"^\s*Sources?:\s*", "", match.group(1))
        ids = [part.strip().translate(_HYPHENS) for part in re.split(r"[,;]", body) if part.strip()]
        if ids and all(i in allowed for i in ids):
            return " ".join(f"[Source: {i}]" for i in ids)
        return match.group(0)

    if allowed:
        text = ALT_CITATION_PATTERN.sub(_canonicalize, text)

    def _check(match: "re.Match") -> str:
        kept = []
        for raw in re.split(r"[,;]", match.group(1)):
            cid = raw.strip().translate(_HYPHENS)
            if not cid:
                continue
            if cid in allowed:
                kept.append(cid)
                if cid not in used:
                    used.append(cid)
            elif cid not in invalid:
                invalid.append(cid)
        return " ".join(f"[Source: {cid}]" for cid in kept)

    cleaned = CITATION_PATTERN.sub(_check, text)
    if invalid:
        logfire.warning(f"🛡️ [OUTPUT RAIL] Removed citations to unverified sources: {invalid}")
    return cleaned, used, invalid


def guard_output(text: str, evidence: Optional[List[str]] = None) -> str:
    """
    Gate 3: Output Guardrails.
    Scans for leaked credentials/tokens and redacts them before returning to user.
    """
    if not settings.GUARDRAILS_ENABLED:
        return text
    redacted_text = text
    for pat in SECRET_PATTERNS:
        redacted_text = re.sub(pat, "[REDACTED_SECRET]", redacted_text)

    if redacted_text != text:
        logfire.warning("🛡️ [OUTPUT RAIL] Redacted sensitive credential pattern from output.")

    return redacted_text
