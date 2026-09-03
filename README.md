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

**OpsMind addresses these challenges** using a 5-agent LangGraph state machine, Model Context Protocol (FastMCP) tool execution, task-based model tiering, 3-tier runtime guardrails, and dual-layer evaluation (official RAGAS + deterministic heuristics).

---

## 🏗️ System Architecture

```
                                  +-----------------------+
                                  |     User Question     |
                                  +-----------+-----------+
                                              |
                                              v
                              +-------------------------------+
                              |   GATE 1: Input Guardrails    |
                              | (Injection / Malicious / Chit)|
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |      1. Query Planner         |
                              |  (Deconstruction & Routing)   |
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
    +--------------------------------+               +----------------------------------+
    |     2. Web Research Agent      |               |     3. Knowledge / RAG Agent     |
    | (FastMCP: web_search)          |               | (FastMCP: knowledge_search)      |
    +----------------+---------------+               +-----------------+----------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    4. Evidence / Critic       |
                              | (Factual Check & Citations)   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |   5. Report Generator Agent   |
                              | (Synthesis & Structured Report)
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |   GATE 3: Output Guardrails   |
                              |  (Secret Masking & Grounding) |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |  Structured Research Report   |
                              +-------------------------------+
```

---

## 🤖 5-Agent LangGraph Workflow

The workflow executes as a compiled LangGraph state graph (`ResearchState`) tracking node outputs, citations, and per-node latencies:

| Agent Node | Responsibility | Task Tier & Model Selection | MCP Tools Invoked |
|---|---|---|---|
| **Query Planner** | Deconstructs user queries into structured search tasks and routing decisions. | Fast Tier (`llama-3.1-8b-instant`) | None (Reasoning) |
| **Web Research Agent** | Gathers external technical literature and industry benchmarks. | Fast Tier (`llama-3.1-8b-instant`) | `web_search` |
| **Knowledge / RAG Agent** | Performs dense semantic retrieval over ChromaDB collections. | Fast Tier (`llama-3.1-8b-instant`) | `knowledge_search`, `document_lookup` |
| **Evidence Critic Agent** | Evaluates retrieved chunks, checks factual alignment, and scores confidence. | Fast Tier (`llama-3.1-8b-instant`) | None (Critic Verification) |
| **Report Generator Agent** | Synthesizes verified evidence into an executive report with explicit `[Source: ...]` tags. | Strong Tier (`llama-3.3-70b-versatile`) | None (Synthesis) |

---

## 📚 ChromaDB Vector Retrieval & Embeddings

- **Vector Database**: Embedded [ChromaDB](https://www.trychroma.com/) with native HNSW index.
- **Embedding Models**:
  - **Live Mode**: Real Hugging Face `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional dense vectors.
  - **Deterministic Mode** (`USE_DETERMINISTIC_MODE=true`): Fast, normalized 384-d deterministic embeddings for local testing and CI/CD without network overhead.
- **Corpus**: Curated technical documents covering RAG evaluation, agentic architectures, LLM gateways, vector indexing algorithms, prompt engineering, and safety guardrails.

---

## 🛠️ Model Context Protocol (FastMCP) Integration

OpsMind implements the official Anthropic Model Context Protocol using `mcp.server.fastmcp.FastMCP`:
- **Server**: `app/tools/mcp_server.py` hosts the `OpsMind-Research-Tools` FastMCP server.
- **Client**: `app/tools/mcp_client.py` discovers registered tools and invokes them with strict schema validation.
- **Registered Tools**:
  1. `web_search(query: str, num_results: int = 3)`
  2. `knowledge_search(query: str, category: str = None, limit: int = 4)`
  3. `document_lookup(doc_id: str)`

---

## 🌐 LLM Gateway & Task-Based Routing

The LLM Gateway (`app/gateway/client.py`) provides intelligent routing and resilient fallback:
- **Task Routing**:
  - `planning`, `classification`, `critic` → Fast, lower-cost model (`llama-3.1-8b-instant`).
  - `synthesis`, `report` → High-reasoning model (`llama-3.3-70b-versatile`).
- **Resilient Fallback Hierarchy**:
  1. **Portkey AI Gateway** (if `PORTKEY_API_KEY` is configured)
  2. **Direct ChatGroq Client** (if `GROQ_API_KEY` is configured)
  3. **Deterministic Reasoning Engine** (offline local fallback ensuring 100% test pass rate without external credentials).

---

## 🛡️ 3-Tier Runtime Guardrails

Safety policies are enforced at three distinct stages without pretending inactive frameworks are running:
1. **Input Guardrails**:
   - Blocks prompt injection / jailbreak patterns (`ignore previous instructions`, `DAN mode`, `developer mode`).
   - Blocks malicious system commands (`rm -rf`, `DROP TABLE`, `format`).
   - Filters non-technical chit-chat queries to preserve system scope.
2. **Tool Execution Guardrails**:
   - Sanitizes arguments against shell injection (`;`, `&&`, `|`, `` ` ``) and path traversal (`../`, `/etc/`).
   - Validates parameters against tool schemas.
3. **Output Guardrails**:
   - Scans responses for sensitive credentials (`sk-`, `ghp_`, `Bearer tokens`), redacting them to `[REDACTED_SECRET]`.
   - Validates grounded source citations.

---

## 📊 Dual-Layer Evaluation Methodology

OpsMind separates automated evaluation into two explicit layers to maintain technical integrity:

### 1. Official RAGAS Framework Evaluation
- Integrates `ragas.metrics`: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- Executed via `ragas.evaluate` when `USE_LIVE_LLM=true` and valid LLM API keys are provided.
- When offline or keys are absent, accurately reports `status: unavailable_offline (requires live LLM API key)` rather than fabricating scores.

### 2. Custom Deterministic Heuristic Metrics
- Evaluates lexical and overlap grounding across the **25-case Golden Dataset** (`evals/golden_dataset.json`):
  - **Faithfulness Heuristic**: Measures sentence-level semantic support from retrieved context.
  - **Answer Relevance Heuristic**: Measures keyword alignment and structural completeness.
  - **Context Relevance & Recall Heuristics**: Validates retrieval density against reference evidence.
  - **Tool Selection Accuracy**: Jaccard index between expected and actual tools.
  - **Guardrail Accuracy**: Binary precision on security and safety test cases.

### Baseline Benchmark Results (Deterministic Heuristics)
| Metric Dimension | Evaluation Metric | Score |
|---|---|---|
| **Groundedness** | Faithfulness (Custom Heuristic) | **86.8%** |
| **Relevance** | Answer Relevance (Custom Heuristic) | **65.6%** |
| **Retrieval** | Context Recall (Custom Heuristic) | **95.0%** |
| | Context Relevance (Custom Heuristic) | **35.9%** |
| **Agent / Tooling** | Tool Selection Accuracy | **76.0%** |
| **Safety** | Guardrail Defense Accuracy | **100.0%** |
| | Citation Compliance | **100.0%** |
| **Composite** | Overall Benchmark Score | **76.6%** |

---

## ⏱️ Honest Latency Profile

Execution latency depends entirely on the active runtime mode:
- **Offline / Deterministic Mode**: **15ms – 30ms** total workflow latency (local CPU memory execution, ideal for unit testing and CI).
- **Live Groq API Inference**: **1.2s – 3.0s** total workflow latency (network roundtrip + Groq LPUs for 5-agent sequential execution).
- **Per-Node Latency Tracking**: Exposed directly in `ResearchState.latencies` and returned in the `/research` API response.

---

## 🚀 Quickstart & Installation

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/Akshat191005/OpsMind.git
cd OpsMind
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Create a `.env` file in the project root:
```env
# Optional: Live LLM Inference
USE_LIVE_LLM=false
GROQ_API_KEY=gsk_your_groq_api_key

# Optional: Portkey Gateway
PORTKEY_API_KEY=your_portkey_api_key

# Optional: LangSmith Tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=lsv2_your_key
LANGSMITH_PROJECT=opsmind_research

# Deterministic Embedding Mode (true = fast offline, false = HuggingFace SentenceTransformer)
USE_DETERMINISTIC_MODE=true
```

---

## 🧪 Testing & Verification

### Run Full Test Suite (47 Tests across 7 Modules)
```bash
python run_tests.py
```
Test modules include:
- `tests.test_guardrails` (Input, tool, and output safety layers)
- `tests.test_gateway` (Task routing, model tiering, and fallback engine)
- `tests.test_mcp_tools` (FastMCP server, schema introspection, client execution)
- `tests.test_rag_pipeline` (SentenceTransformer all-MiniLM-L6-v2 + ChromaDB retrieval)
- `tests.test_agent_graph` (5-agent LangGraph workflow execution)
- `tests.test_evals_ragas` (RAGAS metric imports and custom heuristic scoring)
- `tests.test_api` (FastAPI `/research`, `/status`, `/health`, and `/mcp/tools` endpoints)

### Run Automated 25-Case Evaluation Suite
```bash
python evals/evaluator.py
```

---

## 🖥️ Running the Application

### 1. Launch FastAPI Backend
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **System Diagnostics**: `GET http://localhost:8000/status`
- **Research Investigation**: `POST http://localhost:8000/research`

### 2. Launch Streamlit Web UI
```bash
streamlit run ui/app.py
```
- **Research Assistant**: Interactive query formulation, live per-node latency telemetry, and verified source citations.
- **Evaluation Dashboard**: Automated scorecards, RAGAS execution status, and per-test failure drill-downs.
- **MCP Tool Registry**: Live FastMCP JSON schemas and parameter documentation.
- **Knowledge Base Inspector**: Curated technical documents and metadata explorer.

---

## 📄 License
MIT License. Built for rigorous technical evaluation, pair programming, and production LLM engineering demonstrations.
