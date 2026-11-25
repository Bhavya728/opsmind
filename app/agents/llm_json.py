"""Helpers for extracting text, structured JSON and token usage from LLM responses."""

import json
import re
from typing import Any, Dict, Optional

# Reasoning models (e.g. Qwen) may prepend their chain of thought in <think> tags.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def response_text(response: Any) -> str:
    text = response.content if hasattr(response, "content") else str(response)
    return _THINK_BLOCK.sub("", text).strip()


def token_usage(response: Any) -> int:
    """Total tokens reported by the provider for this call (0 when unavailable, e.g. offline)."""
    usage = getattr(response, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0) or 0)


def fallback_note(response: Any, agent: str) -> Optional[str]:
    """Warning text when the gateway answered with a rate-limit fallback model, else None."""
    model = (getattr(response, "response_metadata", None) or {}).get("fallback_model")
    return f"{agent} used fallback model {model} because the primary model was rate-limited." if model else None


def hit_output_limit(response: Any) -> bool:
    """True when the provider stopped generating because the max output tokens were reached."""
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("finish_reason") == "length"


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the outermost JSON object in text (tolerating code fences and prose), or None."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
