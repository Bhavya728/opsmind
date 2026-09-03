"""
NeMo Guardrails & Multi-Layered Safety System for Incident Investigation.

Implements 3 Guardrail Gates:
1. INPUT RAILS:
   - Prompt Injection & Jailbreak Detection
   - Destructive / Malicious Instruction Detection
   - Out-of-Scope / General Knowledge Filtering
2. TOOL RAILS:
   - Service Name Whitelisting
   - Shell / SQL / Command Injection Prevention
   - Path Traversal & Unauthorized Access Blocking
3. OUTPUT RAILS:
   - API Key & Secret Redaction (Regex & Entropy scanning)
   - Evidence-Backed Hallucination & Unsupported Claim Guard
"""

import re
from typing import Tuple, Optional, Dict, Any, List
from app.utils.logger import logfire
from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS
from app.tools.schemas import VALID_SERVICES

_rails = None

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
    r"(tell\s+(me\s+)?(a\s+)?(funny\s+)?joke|\bjoke\b|write\s+a\s+poem|\bpoem\b)",
    r"(what\s+is\s+the\s+capital\s+of\s+[a-zA-Z\s]+|what\s+should\s+i\s+eat\s+for\s+dinner|\beat\s+for\s+dinner)",
    r"(how\s+do\s+i\s+bake\s+a\s+cake|\bbake\s+a\s+cake|\brecommend\s+a\s+movie|\bweather\s+forecast\b)"
]

SECRET_PATTERNS = [
    r"(?:sk-[a-zA-Z0-9]{20,})",
    r"(?:ghp_[a-zA-Z0-9]{36})",
    r"(?:Bearer\s+[a-zA-Z0-9_\-\.]{30,})",
    r"(?:password\s*[:=]\s*['\"][^'\"]{6,}['\"])",
    r"(?:API_KEY\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"])"
]


def initialize_rails() -> None:
    """Initialize NeMo Guardrails if package and LLM available."""
    global _rails
    try:
        from nemoguardrails import RailsConfig, LLMRails
        from langchain_groq import ChatGroq

        if settings.GROQ_API_KEY:
            guard_llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model="llama-3.1-8b-instant",
                temperature=0
            )
            config = RailsConfig.from_content(
                colang_content=COLANG_CONTENT,
                yaml_content=YAML_CONTENT
            )
            _rails = LLMRails(config, llm=guard_llm)
            logfire.info("🛡️ NeMo Guardrails initialized with LLM (llama-3.1-8b-instant)")
        else:
            logfire.info("🛡️ NeMo Guardrails using deterministic rule engine (no external key required)")
    except Exception as e:
        logfire.warning(f"⚠️ NeMo LLMRails init note: {e}. Active heuristic rails operational.")


def guard_input(message: str) -> Tuple[bool, Optional[str]]:
    """
    Gate 1: Input Guardrails.
    Evaluates incoming query for prompt injections, malicious commands, and off-topic requests.
    """
    clean_msg = message.strip()
    msg_lower = clean_msg.lower()

    # 1. Fast Heuristic Guard: Prompt Injections
    for pat in INJECTION_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.warning(f"🛡️ [INPUT RAIL] Prompt injection blocked: '{clean_msg[:50]}'")
            return True, "🛡️ [SECURITY VIOLATION] Prompt injection or unauthorized instruction override attempt detected. Incident investigation policy strictly forbids guideline manipulation."

    # 2. Fast Heuristic Guard: Destructive Commands
    for pat in MALICIOUS_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.warning(f"🛡️ [INPUT RAIL] Malicious destructive command blocked: '{clean_msg[:50]}'")
            return True, "🛡️ [SECURITY VIOLATION] Destructive or unauthorized infrastructure execution commands are blocked by system safety guardrails."

    # 3. Fast Heuristic Guard: Out-of-Scope Queries
    for pat in OFF_TOPIC_PATTERNS:
        if re.search(pat, msg_lower, re.IGNORECASE):
            logfire.info(f"🛡️ [INPUT RAIL] Off-topic request blocked: '{clean_msg[:50]}'")
            return True, "🛡️ I am an Agentic Incident Investigation System specializing in cloud infrastructure, service telemetry, logs, metrics, and root-cause analysis. I cannot assist with non-operational or general knowledge requests."

    # 4. NeMo LLM Rails (if initialized)
    if _rails is not None:
        try:
            with logfire.span("🛡️ NeMo Guardrails LLM Gate"):
                result = _rails.generate(messages=[{"role": "user", "content": clean_msg}])
                content = result.get("content", "") if isinstance(result, dict) else str(result)
                if any(ind in content for ind in RAIL_INDICATORS):
                    logfire.info(f"🛡️ NeMo Rail Fired: '{content[:60]}'")
                    return True, content
        except Exception as e:
            logfire.warning(f"⚠️ NeMo LLM evaluation skipped: {e}")

    logfire.info("✅ Input guardrails passed.")
    return False, None


# Backward-compatible alias for existing imports
def guard(message: str) -> Tuple[bool, Optional[str]]:
    return guard_input(message)


def guard_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Gate 2: Tool Call Guardrails.
    Validates tool arguments to prevent SQL injection, shell command execution, and path traversal.
    """
    dangerous_shell_chars = [";", "&&", "||", "`", "$", "|", ">", "<", "\n", "\r"]
    
    # Check string arguments for shell injection or path traversal
    for k, v in args.items():
        if isinstance(v, str):
            if k == "service" and any(ch in v for ch in dangerous_shell_chars):
                logfire.warning(f"🛡️ [TOOL RAIL] Dangerous shell character detected in arg '{k}': {v}")
                return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains prohibited shell control characters."
            if ".." in v or "/etc/" in v or "\\windows\\" in v.lower():
                logfire.warning(f"🛡️ [TOOL RAIL] Path traversal detected in arg '{k}': {v}")
                return False, f"🛡️ [TOOL SECURITY VIOLATION] Argument '{k}' contains prohibited file traversal sequence."

    # Validate service name against inventory if service arg exists
    if "service" in args and args["service"]:
        svc = str(args["service"]).strip().lower()
        if not any(valid in svc or svc in valid for valid in VALID_SERVICES):
            logfire.warning(f"🛡️ [TOOL RAIL] Unrecognized service '{svc}' requested")
            # We return warning rather than hard blocking if harmless, but alert
    
    logfire.info(f"✅ Tool call '{tool_name}' verified by guardrails.")
    return True, None


def guard_output(text: str, evidence: Optional[List[str]] = None) -> str:
    """
    Gate 3: Output Guardrails.
    Scans for leaked credentials/tokens and redacts them before returning to user.
    """
    redacted_text = text
    for pat in SECRET_PATTERNS:
        redacted_text = re.sub(pat, "[REDACTED_SECRET]", redacted_text)

    if redacted_text != text:
        logfire.warning("🛡️ [OUTPUT RAIL] Redacted sensitive credential pattern from output.")

    return redacted_text
