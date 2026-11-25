"""
LLM Gateway Module with Portkey Integration and Multi-Tier Task-Based Routing.

- Task tiers: planning / critic / verification -> GROQ_FAST_MODEL (verification: VERIFIER_MODEL);
  synthesis -> GROQ_MODEL
- Backends: Portkey Gateway (if PORTKEY_API_KEY) -> direct Groq; offline engine when USE_LIVE_LLM=false
- get_judge_llm(): independent evaluation judge (JUDGE_MODEL, optional OpenAI-compatible JUDGE_BASE_URL)
"""

import re
import json
from typing import List, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import settings
from app.utils.logger import logfire

# Gateway routing configuration
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503, 500]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_FAST_MODEL}"}},
    ]
}

class DeterministicReasoningLLM(BaseChatModel):
    """
    Deterministic offline LLM for unit tests and local runs without API keys.
    It derives its output from the prompt it receives (the query and the evidence blocks),
    so offline results reflect what was actually retrieved.
    """
    model_name: str = "offline-deterministic"
    task_type: str = "general"

    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs) -> Any:
        parts = []
        for m in messages:
            if hasattr(m, "content"):
                parts.append(str(m.content))
            elif isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        response_text = self._synthesize_response("\n".join(parts))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    @staticmethod
    def _quoted_query(prompt: str) -> str:
        match = re.search(r'(?:Research Query|Query|USER QUERY):?\s*\n?\s*"(.+?)"', prompt)
        return match.group(1) if match else prompt.strip().splitlines()[0][:200]

    def _synthesize_response(self, prompt: str) -> str:
        query = self._quoted_query(prompt)
        query_lower = query.lower()

        if self.task_type in ("planning", "classification", "routing"):
            return json.dumps({
                "in_scope": True,
                "scope_reason": "Offline mode does not judge scope; only pattern guardrails apply.",
                "need_rag": "latest" not in query_lower and "news" not in query_lower,
                "need_web": "internal only" not in query_lower,
                "search_queries": [query],
                "focus_areas": [],
                "plan": [
                    f"1. Retrieve internal knowledge relevant to '{query}'",
                    f"2. Search external sources for '{query}'",
                    "3. Review retrieved evidence for relevance",
                    "4. Write a cited report from the accepted evidence",
                ],
            })

        if self.task_type in ("critic", "evaluation"):
            refs = re.findall(r"^\[(S\d+)\]", prompt, re.MULTILINE)
            return json.dumps({
                "sources": [{"ref": r, "relevant": True, "reason": "Accepted by offline critic (retrieval threshold already applied)."} for r in refs],
                "is_grounded": bool(refs),
                "identified_gaps": [],
                "critic_notes": f"Offline critic accepted {len(refs)} retrieved sources without semantic review.",
            })

        if self.task_type == "verification":
            ids_line = re.search(r"^Source ids: (.*)$", prompt, re.MULTILINE)
            allowed = {i.strip() for i in ids_line.group(1).split(",")} if ids_line else set()
            report = prompt.split("REPORT:", 1)[-1]
            claims = [
                {"claim": line.strip()[:200], "source": cid, "verdict": "supported" if cid in allowed else "unsupported"}
                for line in report.splitlines()
                for cid in re.findall(r"\[Source: ([^\]]+)\]", line)[:1]
            ]
            return json.dumps({"claims": claims})

        evidence = re.findall(r"^\[Source: ([^\]]+)\] (.+)\n(.+)", prompt, re.MULTILINE)
        evidence = list({cid: (cid, title, line) for cid, title, line in evidence}.values())
        if not evidence:
            return (
                "### Executive Summary\n"
                f"No verified evidence was found for \"{query}\". OpsMind is running in offline mode, "
                "so no answer can be generated without retrieved sources."
            )
        findings = "\n".join(
            f"- **{title.strip()}**: {first_line.strip()[:240]} [Source: {cid}]"
            for cid, title, first_line in evidence
        )
        return (
            "### Executive Summary\n"
            f"Offline summary of the {len(evidence)} verified sources retrieved for \"{query}\".\n\n"
            "### Key Findings\n"
            f"{findings}\n\n"
            "### Note\n"
            "Generated by the offline deterministic engine; set USE_LIVE_LLM=true for full synthesis."
        )

    @property
    def _llm_type(self) -> str:
        return "deterministic-research"


class ResilientFallbackLLM(BaseChatModel):
    """
    Wraps the primary LLM. On failure it raises, unless ALLOW_OFFLINE_FALLBACK=true,
    in which case it answers with the offline engine and logs a warning.
    """
    primary_llm: Optional[BaseChatModel] = None
    fallback_llm: BaseChatModel
    # Other models tried in order when the primary is rate-limited (e.g. its daily token quota is spent).
    rate_limit_fallbacks: List[Any] = []
    task_type: str = "general"
    active_backend: str = "deterministic"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Any:
        if self.primary_llm is None:
            return self.fallback_llm._generate(messages, stop=stop, **kwargs)
        try:
            return self.primary_llm._generate(messages, stop=stop, **kwargs)
        except Exception as e:
            if _is_rate_limit(e) and self.rate_limit_fallbacks:
                return self._generate_with_fallback_models(messages, stop, e, **kwargs)
            if not settings.ALLOW_OFFLINE_FALLBACK:
                logfire.error(f"❌ {self.active_backend} LLM call failed for task '{self.task_type}': {e}")
                raise
            logfire.warning(f"⚠️ {self.active_backend} LLM call failed ({e}). ALLOW_OFFLINE_FALLBACK=true, using offline engine.")
            return self.fallback_llm._generate(messages, stop=stop, **kwargs)

    def _generate_with_fallback_models(self, messages, stop, first_error: Exception, **kwargs) -> Any:
        last_error = first_error
        for name, llm in self.rate_limit_fallbacks:
            logfire.warning(f"⚠️ Rate limit on the primary model for '{self.task_type}'; retrying with {name}.")
            try:
                result = llm._generate(messages, stop=stop, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                last_error = e
                continue
            for generation in result.generations:
                generation.message.response_metadata["fallback_model"] = name
            return result
        logfire.error(f"❌ All models are rate-limited for task '{self.task_type}': {last_error}")
        raise last_error

    @property
    def _llm_type(self) -> str:
        return f"resilient-fallback({self.active_backend})"


def _has_key(key: Optional[str]) -> bool:
    return bool(key and key.strip() and not key.startswith("mock"))


def _is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError"


def get_llm_runtime_status() -> str:
    """Return name of active LLM backend."""
    if not settings.USE_LIVE_LLM:
        return "deterministic"
    if _has_key(settings.PORTKEY_API_KEY):
        return "portkey"
    if _has_key(settings.GROQ_API_KEY):
        return "groq"
    return "unconfigured"


FAST_TASKS = ("planning", "classification", "critic", "routing", "verification")
# Tasks with their own configurable model.
TASK_MODELS = {"verification": settings.VERIFIER_MODEL}


def _groq_chat(api_key: str, model: str, max_tokens: int, reasoning_effort: str) -> BaseChatModel:
    from langchain_groq import ChatGroq
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=settings.LLM_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        max_tokens=max_tokens,
        **kwargs,
    )


def get_llm(task_type: str = "general") -> BaseChatModel:
    """
    Task-based LLM Router:
    - "planning" / "classification" / "critic" -> Cheaper fast model (GROQ_FAST_MODEL)
    - "synthesis" / "report" -> Stronger model (GROQ_MODEL)
    - Backend order: Portkey Gateway -> Direct Groq. Offline engine when USE_LIVE_LLM=false
      (or on failure only if ALLOW_OFFLINE_FALLBACK=true).
    """
    fast = task_type in FAST_TASKS
    model_name = TASK_MODELS.get(task_type) or (settings.GROQ_FAST_MODEL if fast else settings.GROQ_MODEL)
    max_tokens = settings.LLM_MAX_OUTPUT_TOKENS_FAST if fast else settings.LLM_MAX_OUTPUT_TOKENS_REPORT
    fallback_engine = DeterministicReasoningLLM(task_type=task_type)
    
    if not settings.USE_LIVE_LLM:
        return fallback_engine

    # 1. Try Portkey Gateway if key is present
    if _has_key(settings.PORTKEY_API_KEY):
        try:
            from langchain_openai import ChatOpenAI
            from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
            
            slug = settings.GROQ_SLUG_2 if fast else settings.GROQ_SLUG
            
            llm = ChatOpenAI(
                api_key=settings.PORTKEY_API_KEY,
                base_url=PORTKEY_GATEWAY_URL,
                model=f"@{slug}/{model_name}",
                temperature=settings.LLM_TEMPERATURE,
                timeout=settings.LLM_TIMEOUT_SECONDS,
                max_retries=settings.LLM_MAX_RETRIES,
                max_tokens=max_tokens,
                default_headers=createHeaders(
                    api_key=settings.PORTKEY_API_KEY,
                    config=GATEWAY_CONFIG,
                    metadata={
                        "task_type": task_type,
                        "environment": settings.ENVIRONMENT
                    }
                )
            )
            logfire.info(f"🌐 LLM Gateway: Routing task '{task_type}' via Portkey Gateway (@{slug}/{model_name})")
            return ResilientFallbackLLM(primary_llm=llm, fallback_llm=fallback_engine, task_type=task_type, active_backend="portkey")
        except Exception as e:
            logfire.warning(f"⚠️ Portkey Gateway init note: {e}. Trying direct Groq client.")

    # 2. Try direct ChatGroq
    if _has_key(settings.GROQ_API_KEY):
        try:
            llm = _groq_chat(settings.GROQ_API_KEY, model_name, max_tokens, settings.LLM_REASONING_EFFORT)
            # Fallback models get no reasoning_effort: valid values differ between model families.
            fallbacks = [(m, _groq_chat(settings.GROQ_API_KEY, m, max_tokens, ""))
                         for m in settings.LLM_FALLBACK_MODELS if m != model_name]
            logfire.info(f"⚡ LLM Gateway: Direct Groq client for '{task_type}' ({model_name})")
            return ResilientFallbackLLM(primary_llm=llm, fallback_llm=fallback_engine, rate_limit_fallbacks=fallbacks,
                                        task_type=task_type, active_backend="groq")
        except Exception as e:
            logfire.warning(f"⚠️ Groq direct client note: {e}. Trying fallback.")

    # 3. Live mode without a usable backend
    if not settings.ALLOW_OFFLINE_FALLBACK:
        raise RuntimeError("USE_LIVE_LLM=true but no usable GROQ_API_KEY or PORTKEY_API_KEY is configured.")
    logfire.warning(f"⚠️ LLM Gateway: no live backend available, using offline engine for '{task_type}'")
    return fallback_engine


def get_judge_llm() -> BaseChatModel:
    """
    Independent evaluation judge: JUDGE_MODEL with JUDGE_API_KEY (defaults to GROQ_API_KEY).
    JUDGE_BASE_URL points it at any OpenAI-compatible provider instead of Groq.
    """
    if not settings.USE_LIVE_LLM:
        raise RuntimeError("The evaluation judge needs USE_LIVE_LLM=true.")
    if not _has_key(settings.JUDGE_API_KEY):
        raise RuntimeError("No JUDGE_API_KEY or GROQ_API_KEY configured for the evaluation judge.")
    if settings.JUDGE_BASE_URL:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.JUDGE_API_KEY,
            base_url=settings.JUDGE_BASE_URL,
            model=settings.JUDGE_MODEL,
            temperature=0.0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            max_tokens=settings.JUDGE_MAX_OUTPUT_TOKENS,
        )
    return _groq_chat(settings.JUDGE_API_KEY, settings.JUDGE_MODEL, settings.JUDGE_MAX_OUTPUT_TOKENS,
                      settings.JUDGE_REASONING_EFFORT)
