import os
from dotenv import load_dotenv

# Disable telemetry and network calls globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_OPTOUT"] = "True"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()


class Settings:
    # --- Vector DB (ChromaDB) & Embeddings ---
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "opsmind_research_kb")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    USE_DETERMINISTIC_MODE = os.getenv("USE_DETERMINISTIC_MODE", "true").lower() == "true"

    # --- LLM (Groq) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

    # --- LLM Gateway (Portkey) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "research")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "fast")

    # --- MCP Tool Layer ---
    USE_MCP_SERVER = os.getenv("USE_MCP_SERVER", "true").lower() == "true"

    # --- Guardrails ---
    GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"

    # --- System Mode ---
    LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"
    USE_LIVE_LLM = os.getenv("USE_LIVE_LLM", "false").lower() == "true"

    # --- Observability (LangSmith) ---
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "opsmind_research")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_TRACING = bool(
        (os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" or os.getenv("LANGSMITH_API_KEY"))
        and os.getenv("LANGSMITH_API_KEY")
        and not os.getenv("LANGSMITH_API_KEY").startswith("mock")
    )


# Apply LangSmith env vars only if a valid key is provided
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_API_KEY", "").startswith("mock"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "opsmind_research")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

settings = Settings()

