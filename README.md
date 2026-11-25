# OpsMind — Multi-Agent Research & Technical Knowledge System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-6366F1.svg)](https://www.trychroma.com/)
[![MCP](https://img.shields.io/badge/Protocol-Model_Context_Protocol-8B5CF6.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpsMind** is a stateful Multi-Agent AI Research and Knowledge System designed to investigate complex computer science, machine learning, and software architecture questions. It coordinates dynamic query planning, web research, dense vector search across a ChromaDB private knowledge base, factual evidence critiquing, runtime safety guardrails, and grounded report generation with source citations.

---

## 🎯 Motivation & Problem Scope

Single-shot Retrieval-Augmented Generation (RAG) pipelines (`Query → Embed → Top-k Search → Augment → Response`) encounter severe limitations on multi-step technical investigations:
1. **Multi-Hop Synthesis**: Complex questions span multiple disciplines (e.g., comparing agentic workflows with traditional RAG while analyzing evaluation metrics).
2. **Context Contamination**: Irrelevant or low-quality vector chunks pollute the context window, inducing hallucinations.
3. **Information Freshness**: Internal knowledge bases lack recent framework releases and industry benchmarks.
4. **Safety & Compliance**: Unguarded LLMs are susceptible to prompt injection, unauthorized command execution, and accidental credential leakage.

**OpsMind addresses these challenges** using a 6-agent LangGraph state machine with planner-driven routing, an evidence critic, post-report claim verification, Model Context Protocol (FastMCP) tools, task-based model tiering, layered runtime guardrails, and an evaluation suite with an independent LLM judge.

---

## 🏗️ System Architecture

```
 Question ─► Input guardrail (patterns) ──blocked──► refusal
                 │
                 ▼
          Query Planner ──out of scope──► refusal
     (scope, routing, sub-queries)
          │             │            (one, both in parallel, or neither)
          ▼             ▼
    Web Research   Knowledge / RAG
    (Tavily/ddgs)  (ChromaDB, hybrid score ≥ MIN_RELEVANCE_SCORE)
          └──────┬──────┘
                 ▼
          Evidence Critic ──gaps──► one follow-up retrieval round (MAX_RESEARCH_ROUNDS)
     (accept/reject each source, evidence quality)
                 ▼
          Report Generator ◄──────── one revision (MAX_REPORT_REVISIONS)
     (accepted sources only, [Source: id])          ▲
                 ▼                                   │ too many unsupported
          Output guardrail                           │ claims, or no citations
     (secret masking, citation validation)           │
                 ▼                                   │
          Claim Verifier ────────────────────────────┘
     (checks claims vs cited sources, final confidence)
                 ▼
     Report + sources + claims + confidence + warnings
```

---

## 🤖 Agents

The workflow is a compiled LangGraph state graph (`ResearchState`) with conditional routing and two bounded loops.

| Agent | Responsibility | Model (default) |
|---|---|---|
| **Query Planner** | Scope check against `RESEARCH_SCOPE`, routing (knowledge base / web), up to `MAX_SEARCH_QUERIES` sub-queries | `GROQ_FAST_MODEL` (`openai/gpt-oss-20b`) |
| **Web Research** | Runs each sub-query through Tavily (or ddgs), de-duplicates by URL | — |
| **Knowledge / RAG** | Hybrid vector + keyword retrieval over the chunked corpus; drops chunks below `MIN_RELEVANCE_SCORE` | — |
| **Evidence Critic** | Reads each new source, accepts or rejects it, estimates coverage, flags contradictions, may request one follow-up round for gaps | `GROQ_FAST_MODEL` |
| **Report Generator** | Writes the report from accepted sources only, citing `[Source: id]`; rewrites once on verifier feedback | `GROQ_MODEL` (`openai/gpt-oss-120b`) |
| **Claim Verifier** | Extracts up to `MAX_VERIFIED_CLAIMS` claims and checks each against its cited source | `VERIFIER_MODEL` (defaults to the fast model) |

**Confidence** is computed, not self-reported:
`confidence = evidence_quality × claim_support`, where
- `evidence_quality` = average of the accepted sources' retrieval relevance and the critic's coverage estimate
- `claim_support` = (supported + ½·partial + ½·uncited) / claims checked

With no accepted evidence, confidence is 0 and the report says so.

---

## 📚 Knowledge Base & Retrieval

- **Corpus**: built-in documents plus any `.md`/`.txt` files in `KNOWLEDGE_DIR` (`app/data/documents/`; sub-folder name = category).
- **Chunking**: line-aware windows of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (live) or deterministic hash embeddings (offline/tests).
- **Scoring**: `HYBRID_VECTOR_WEIGHT` × cosine similarity + the rest × query-term coverage.
- **Auto re-index**: a fingerprint of documents, chunk settings and embedding model is stored with the collection; any change rebuilds it.

---

## 🛠️ Model Context Protocol (FastMCP) Tools

`app/tools/mcp_server.py` registers three FastMCP tools, called through `app/tools/mcp_client.py` after the tool guardrail:
1. `web_search(query, num_results)`: Tavily API when `TAVILY_API_KEY` is set, otherwise ddgs scraping; offline mode uses a small built-in index
2. `knowledge_search(query, category, limit)`
3. `document_lookup(doc_id)`

---

## 🌐 LLM Gateway

`app/gateway/client.py` routes each task to a model tier (fast: planner, critic, verifier; strong: report) via Portkey (if `PORTKEY_API_KEY`) or Groq directly.
- **Failures surface as errors**: a failed live call raises, and the API returns HTTP 502. Set `ALLOW_OFFLINE_FALLBACK=true` only if you want offline template answers instead.
- **Rate limits**: Groq's free tier allows about 8,000 tokens/minute and 200,000 tokens/day per model. Output caps (`LLM_MAX_OUTPUT_TOKENS_*`) shrink each request, and `LLM_MAX_RETRIES` retries wait the time Groq asks for. If a model is still rate-limited (typically its daily quota is spent), the request is retried on the next model in `LLM_FALLBACK_MODELS`, and the answer's warnings name the model used.
- **Offline engine** (`USE_LIVE_LLM=false`): deterministic, derives its output from the actual prompt; used by the test suite.

---

## 🛡️ Guardrails

1. **Input**: pattern checks for prompt injection, destructive commands and obvious chit-chat, plus a length limit (`MAX_QUERY_CHARS`). The planner then applies the topic check (`TOPIC_GUARD_ENABLED`).
2. **Tool**: identifier arguments must be plain ids. Search text may use normal punctuation, but command sequences (`&&`, backticks, `$(`, `rm -rf` …), path traversal and control characters are rejected.
3. **Output**: secret-pattern redaction; citation validation normalizes `[Source: a, b]`, `[a]`, `【a】`, `(Source: a)` and removes citations to unverified sources.
4. **Untrusted evidence**: critic, report and verifier prompts treat retrieved text as data and ignore instructions inside it.

---

## 📊 Evaluation

`python evals/evaluator.py` runs the golden dataset (`evals/golden_dataset.json`, 30 cases: research, freshness, not-in-knowledge-base, special characters, critic distractors, injection, malicious, off-topic, scope) and scores two layers:

**A. Deterministic checks** (`evals/metrics.py`, from the system's own output):
| Metric | Meaning |
|---|---|
| Guardrail accuracy | Blocked/refused exactly when expected |
| Routing recall | The planner used every retrieval path the case requires (extra paths are fine) |
| Critic source recall / precision | Accepted knowledge-base documents vs. the case's `expected_relevant_sources` |
| Cites sources | The report cites at least one verified source |

**B. Independent judge** (`evals/judge.py`, `JUDGE_MODEL`, default `qwen/qwen3.8-27b`, a different model family from the system):
| Metric | Meaning |
|---|---|
| Correctness | Coverage of the case's `expected_answer_characteristics` (yes / partial / no) |
| Faithfulness | 1 − unsupported claims / total claims, judged against the sources |
| Relevance | 1–5 rating, scaled to 0–1 |

Runs that fail (e.g. rate limits) are recorded as errors and excluded from averages. Each report stores the configuration used. Results are written to `evals/evaluation_results.json` and shown on the UI's Evaluation Dashboard.

```bash
python evals/evaluator.py --limit 5        # quick run
python evals/evaluator.py --ids 1,26,30    # specific cases
python evals/evaluator.py --no-judge       # deterministic checks only
```
The judge can use another account or provider: set `JUDGE_API_KEY` (and `JUDGE_BASE_URL` for any OpenAI-compatible endpoint). Groq limits each model separately, so a different `JUDGE_MODEL` on the same key already gets its own quota.

---

## ⏱️ Latency & Cost

Measured with Groq's free tier and Tavily: a typical research question takes **20–90 s** and **10k–30k tokens**.
Groq's free tier also caps each model at **200,000 tokens per day**, and most calls (planner, critic, verifier) use the fast model, so expect roughly 10–20 questions per day before waiting for the rolling window. For heavier use, upgrade to Groq's Dev tier or spread tasks across models (`VERIFIER_MODEL`, `GROQ_FAST_MODEL`). Time depends on whether the critic requests a follow-up round, whether the report needs a revision, and on rate-limit waits. Offline mode (tests) runs in well under a second per question. Per-agent latency and token usage are returned with every answer.

---

## 📁 Project Layout

```
app/
  main.py                  FastAPI backend (/research, /status, /mcp/tools) and web UI host
  config.py                every setting, overridable in .env
  agents/
    graph.py               LangGraph workflow, routing, investigate_query()
    state.py               ResearchState and its reducers
    llm_json.py            response text / JSON / token-usage helpers
    nodes/                 planner, web, RAG, critic, report, claim verifier
  gateway/client.py        model routing (Groq / Portkey), offline engine, evaluation judge
  guardrails/rails.py      input, tool and output guardrails, citation validation
  services/retrieval/      ChromaDB service: corpus loading, chunking, hybrid scoring
  tools/                   FastMCP server, client and tool schemas
  data/knowledge_base.py   built-in documents; add your own under app/data/documents/
evals/                     evaluator, deterministic metrics, LLM judge, golden dataset
ui/index.html              web UI (static page served by FastAPI at /)
tests/                     offline test suite (python run_tests.py)
DOCS/, notebooks/          background guides and experiments (not OpsMind's implementation)
```

---

## 🚀 Quickstart & Installation

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/Akshat191005/OpsMind.git
cd OpsMind
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root. Only the first two lines are needed for live use; everything else has a default in `app/config.py`.
```env
USE_LIVE_LLM=true                # false = offline deterministic engine (what the test suite uses)
GROQ_API_KEY=gsk_your_groq_api_key

# Models (Groq retires models; check console.groq.com/docs/models)
GROQ_MODEL=openai/gpt-oss-120b          # report synthesis
GROQ_FAST_MODEL=openai/gpt-oss-20b      # planner and critic
VERIFIER_MODEL=openai/gpt-oss-20b       # claim verifier (defaults to GROQ_FAST_MODEL)
ALLOW_OFFLINE_FALLBACK=false     # true = on LLM failure, answer with the offline engine instead of an error
LLM_MAX_RETRIES=6                # retries wait the time Groq asks for on rate limits
LLM_FALLBACK_MODELS=openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b  # tried in order when a model stays rate-limited
LLM_MAX_OUTPUT_TOKENS_FAST=1500  # planner/critic/verifier output cap (also lowers per-minute token reservations)
LLM_MAX_OUTPUT_TOKENS_REPORT=3000
LLM_REASONING_EFFORT=low         # gpt-oss reasoning effort: low, medium, high

# Critic, verification & revision
MAX_RESEARCH_ROUNDS=2            # 2 = critic may request one follow-up retrieval round; 1 disables
CLAIM_VERIFICATION_ENABLED=true
MAX_VERIFIED_CLAIMS=10
REVISION_UNSUPPORTED_THRESHOLD=0.3   # rewrite the report once if more claims than this are unsupported
MAX_REPORT_REVISIONS=1
MAX_QUERY_CHARS=2000

# Evaluation judge
JUDGE_MODEL=qwen/qwen3.8-27b     # different model family from the system
JUDGE_API_KEY=                   # empty = GROQ_API_KEY
JUDGE_BASE_URL=                  # any OpenAI-compatible endpoint (Gemini, OpenRouter, local server)

# Scope enforced by the planner's topic check
RESEARCH_SCOPE=computer science, software engineering, machine learning, AI and LLM systems, ...
TOPIC_GUARD_ENABLED=true

# Knowledge base: built-in documents plus .md/.txt files in KNOWLEDGE_DIR (sub-folder = category).
# The index rebuilds automatically when documents, chunk settings or the embedding model change.
KNOWLEDGE_DIR=app/data/documents
USE_DETERMINISTIC_MODE=false     # false = sentence-transformers embeddings (default when live)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Retrieval
MAX_SEARCH_QUERIES=3             # sub-queries the planner may generate
RAG_TOP_K=4
WEB_TOP_K=3
MIN_RELEVANCE_SCORE=0.3          # chunks below this hybrid score are dropped
MAX_CONTEXT_SOURCES=6            # per source type (knowledge base, web), across rounds
HYBRID_VECTOR_WEIGHT=0.6         # vector similarity vs. query-term coverage
USE_LIVE_WEB=true                # live web search (default when live); false = small built-in index
TAVILY_API_KEY=tvly-...          # official search API (app.tavily.com, free tier); used automatically when set
WEB_SEARCH_PROVIDER=auto         # auto = tavily if key set, else ddgs; or force tavily / ddgs
TAVILY_SEARCH_DEPTH=basic        # basic (1 credit/search) or advanced (2 credits)
WEB_RECENT_TIME_RANGE=year       # filter for questions the planner marks time-sensitive: day/week/month/year
WEB_SEARCH_BACKEND=auto          # ddgs only: engines tried in order, e.g. duckduckgo,brave,mojeek
WEB_SEARCH_TIMEOUT=15

# Optional integrations
PORTKEY_API_KEY=
LANGSMITH_API_KEY=
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

---

## 🧪 Testing & Verification

### Run Full Test Suite (offline: no API calls, throwaway vector index)
```bash
python run_tests.py
```
Test modules include:
- `tests.test_guardrails` (Input, tool, and output safety layers)
- `tests.test_gateway` (Task routing, model tiering, and fallback engine)
- `tests.test_mcp_tools` (FastMCP server, schema introspection, client execution)
- `tests.test_rag_pipeline` (SentenceTransformer all-MiniLM-L6-v2 + ChromaDB retrieval)
- `tests.test_agent_graph` (5-agent LangGraph workflow execution)
- `tests.test_evals` (evaluation metrics, judge scoring, error handling)
- `tests.test_api` (FastAPI `/research`, `/status`, `/health`, and `/mcp/tools` endpoints)

### Run the Evaluation Suite (live; see Evaluation above)
```bash
python evals/evaluator.py --limit 5
```

---

## 🖥️ Running the Application

One uvicorn process serves both the API and the web UI:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### API
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **System Diagnostics**: `GET http://localhost:8000/status`
- **Research Investigation**: `POST http://localhost:8000/research` (or `POST /research/stream` for Server-Sent Events with per-agent progress)
- **Evaluation Results / Knowledge Base**: `GET /evals/results`, `GET /knowledge/documents`

### Web UI: `http://localhost:8000/`
- **Research**: ask a question, watch each agent finish, then read the report with numbered citations linked to its sources. Claim checks, the plan, timing and token use sit under *Details*.
- **Knowledge base**: searchable list of indexed documents.
- **Evaluations**: scores from the latest `evals/evaluator.py` run; failing cases are listed first.

Add `--reload-include .env` to the uvicorn command to restart automatically when `.env` changes (settings are read at startup).

---

## 📄 License
MIT License. Built for rigorous technical evaluation, pair programming, and production LLM engineering demonstrations.
