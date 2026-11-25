import os
from dotenv import load_dotenv

# Disable telemetry and network calls globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OPTOUT"] = "True"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _list(name: str, default: str) -> list:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


class Settings:
    VERSION = "2.1.0"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

    # --- System Mode ---
    # USE_LIVE_LLM=true calls Groq/Portkey; false uses the offline deterministic engine (tests/CI).
    USE_LIVE_LLM = _bool("USE_LIVE_LLM", False)
    # In live mode, a failed LLM call raises instead of silently returning offline template output.
    ALLOW_OFFLINE_FALLBACK = _bool("ALLOW_OFFLINE_FALLBACK", False)

    # --- LLM (Groq) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
    # Model for the post-report claim verifier (defaults to the fast model).
    VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", GROQ_FAST_MODEL)
    LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)
    LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 60)
    # Retries cover connection errors and rate limits; on a 429 the client waits the time Groq asks for.
    # The free tier allows ~8,000 tokens/minute per model, so back-to-back questions need several waits.
    LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 6)
    # When a model is still rate-limited after retries (e.g. its 200k tokens/day free quota is spent),
    # the same request is retried on these models in order. Empty = no model fallback.
    LLM_FALLBACK_MODELS = _list("LLM_FALLBACK_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b")
    # Output caps also lower how many tokens each request reserves against the per-minute limit.
    LLM_MAX_OUTPUT_TOKENS_FAST = _int("LLM_MAX_OUTPUT_TOKENS_FAST", 1500)
    LLM_MAX_OUTPUT_TOKENS_REPORT = _int("LLM_MAX_OUTPUT_TOKENS_REPORT", 3000)
    # gpt-oss / qwen reasoning effort: low, medium or high. Empty = provider default.
    LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

    # --- Evaluation judge (independent model that grades answers) ---
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or GROQ_API_KEY
    # Any OpenAI-compatible endpoint (e.g. Gemini, OpenRouter, a local server). Empty = Groq.
    JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "")
    JUDGE_MAX_OUTPUT_TOKENS = _int("JUDGE_MAX_OUTPUT_TOKENS", 2000)
    JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "")

    # --- Input limits ---
    MAX_QUERY_CHARS = _int("MAX_QUERY_CHARS", 2000)

    # --- LLM Gateway (Portkey) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "research")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "fast")

    # --- Research scope (used by the planner's topic check and prompts) ---
    RESEARCH_SCOPE = os.getenv(
        "RESEARCH_SCOPE",
        "computer science, software engineering, machine learning, AI and LLM systems, "
        "data engineering, databases, cloud infrastructure, DevOps and security",
    )
    TOPIC_GUARD_ENABLED = _bool("TOPIC_GUARD_ENABLED", True)

    # --- Knowledge base, chunking & embeddings ---
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "opsmind_research_kb")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(APP_DIR, "data", "chroma_index"))
    # Extra .md/.txt files in this folder (sub-folder name = category) are indexed alongside the built-in corpus.
    KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", os.path.join(APP_DIR, "data", "documents"))
    INCLUDE_BUILTIN_DOCUMENTS = _bool("INCLUDE_BUILTIN_DOCUMENTS", True)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Defaults to real embeddings in live mode and hash embeddings offline.
    USE_DETERMINISTIC_MODE = _bool("USE_DETERMINISTIC_MODE", not USE_LIVE_LLM)
    CHUNK_SIZE = _int("CHUNK_SIZE", 800)
    CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

    # --- Retrieval ---
    MAX_SEARCH_QUERIES = _int("MAX_SEARCH_QUERIES", 3)
    RAG_TOP_K = _int("RAG_TOP_K", 4)
    WEB_TOP_K = _int("WEB_TOP_K", 3)
    MAX_TOOL_RESULTS = _int("MAX_TOOL_RESULTS", 10)
    MIN_RELEVANCE_SCORE = _float("MIN_RELEVANCE_SCORE", 0.3)
    # Weight of vector similarity vs. query-term coverage in the hybrid relevance score.
    HYBRID_VECTOR_WEIGHT = _float("HYBRID_VECTOR_WEIGHT", 0.6)
    # Live web search; offline mode searches the small built-in web index instead.
    USE_LIVE_WEB = _bool("USE_LIVE_WEB", USE_LIVE_LLM)
    # "tavily" (official API), "ddgs" (scrapes search engines, often blocked), or "auto" = tavily when a key is set.
    WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "auto").strip().lower()
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")  # "basic" (1 credit) or "advanced" (2 credits)
    # ddgs engines tried in order, e.g. "auto" or "duckduckgo,brave,mojeek"; timeout is per request.
    WEB_SEARCH_BACKEND = os.getenv("WEB_SEARCH_BACKEND", "auto")
    WEB_SEARCH_TIMEOUT = _int("WEB_SEARCH_TIMEOUT", 15)
    # Recency filter for questions the planner marks time-sensitive: day, week, month or year.
    WEB_RECENT_TIME_RANGE = os.getenv("WEB_RECENT_TIME_RANGE", "year")

    @property
    def ACTIVE_WEB_PROVIDER(self) -> str:
        if not self.USE_LIVE_WEB:
            return "offline_index"
        if self.WEB_SEARCH_PROVIDER == "auto":
            return "tavily" if (self.TAVILY_API_KEY or "").strip() else "ddgs"
        return self.WEB_SEARCH_PROVIDER
    # Per source type (knowledge base and web), across all research rounds.
    MAX_CONTEXT_SOURCES = _int("MAX_CONTEXT_SOURCES", 6)

    # --- Critic, verification & revision ---
    CRITIC_SNIPPET_CHARS = _int("CRITIC_SNIPPET_CHARS", 700)
    # Characters of each accepted source passed to the report writer and claim verifier.
    REPORT_SOURCE_CHARS = _int("REPORT_SOURCE_CHARS", 1200)
    # 2 = the critic may request one extra retrieval round for evidence gaps; 1 disables it.
    MAX_RESEARCH_ROUNDS = _int("MAX_RESEARCH_ROUNDS", 2)
    CLAIM_VERIFICATION_ENABLED = _bool("CLAIM_VERIFICATION_ENABLED", True)
    MAX_VERIFIED_CLAIMS = _int("MAX_VERIFIED_CLAIMS", 10)
    # Rewrite the report once when this share of checked claims is unsupported, or when it cites nothing.
    REVISION_UNSUPPORTED_THRESHOLD = _float("REVISION_UNSUPPORTED_THRESHOLD", 0.3)
    MAX_REPORT_REVISIONS = _int("MAX_REPORT_REVISIONS", 1)

    # --- MCP Tool Layer ---
    USE_MCP_SERVER = _bool("USE_MCP_SERVER", True)

    # --- Guardrails ---
    GUARDRAILS_ENABLED = _bool("GUARDRAILS_ENABLED", True)

    # --- API ---
    CORS_ORIGINS = _list("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

    # --- Observability (LangSmith) ---
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "opsmind_research")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_TRACING = bool(
        (os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" or os.getenv("LANGSMITH_API_KEY"))
        and os.getenv("LANGSMITH_API_KEY")
        and not os.getenv("LANGSMITH_API_KEY").startswith("mock")
    )

    @property
    def MODE(self) -> str:
        return "live" if self.USE_LIVE_LLM else "offline"


# Apply LangSmith env vars only if a valid key is provided
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_API_KEY", "").startswith("mock"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "opsmind_research")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

settings = Settings()
