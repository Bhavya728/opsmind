# OpsMind — Agentic Research & Knowledge System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-6366F1.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpsMind** is a production-oriented Multi-Agent AI Research and Knowledge System designed to investigate complex computer science, machine learning, and software architecture questions. It coordinates dynamic query planning, external web research, dense vector search across a ChromaDB private knowledge base, factual evidence critiquing, runtime safety guardrails, and grounded research report synthesis with verifiable source citations.

---

## 🎯 Problem Statement & Objective

Traditional Retrieval-Augmented Generation (RAG) pipelines follow a rigid, single-shot execution: `Query → Embed → Top-k Vector Search → Prompt Augmentation → Response`. This approach fails when:
1. **Multi-Hop Synthesis is Required**: Complex research questions span multiple concepts (e.g., comparing agentic workflows with traditional RAG while analyzing evaluation metrics).
2. **Retrieval Noise Exists**: Irrelevant vector chunks pollute context windows, inducing hallucinations.
3. **External Freshness is Needed**: Internal knowledge bases lack recent industry framework releases or external benchmarks.
4. **Safety & Compliance are Missing**: LLM applications without strict input/output guardrails are vulnerable to prompt injections, command injections, and credential leakage.

**OpsMind solves these challenges** by implementing a modular 5-agent LangGraph workflow backed by an LLM Gateway with task-based routing, Model Context Protocol (MCP) tool integration, three-tier runtime guardrails, and automated evaluation metrics.

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
    | (MCP Tool: web_search)         |               | (MCP Tool: knowledge_search)     |
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

## 🤖 Multi-Agent Workflow & Responsibilities

The system orchestrates **5 specialized agents** within a stateful LangGraph execution graph:

| Agent | Responsibility | Task-Tier LLM | Tools Used |
|---|---|---|---|
| **Query Planner** | Deconstructs research queries into sub-questions, identifies focus areas, and determines retrieval routing. | Fast / Tier-1 (`llama-3.1-8b-instant`) | None (Reasoning) |
| **Web Research Agent** | Gathers recent external technical literature, release notes, and industry benchmarks. | Fast / Tier-1 (`llama-3.1-8b-instant`) | MCP: `web_search` |
| **Knowledge / RAG Agent** | Performs dense semantic retrieval across ChromaDB vector collections using HuggingFace embeddings. | Fast / Tier-1 (`llama-3.1-8b-instant`) | MCP: `knowledge_search`, `document_lookup` |
| **Evidence Critic Agent** | Evaluates retrieved chunks, detects factual contradictions, scores confidence, and compiles citations. | Fast / Tier-1 (`llama-3.1-8b-instant`) | None (Verification) |
| **Report Generator** | Synthesizes verified evidence into an authoritative research report with explicit markdown citations `[Source: ...]`. | Strong / Tier-2 (`llama-3.3-70b-versatile`) | None (Synthesis) |

---

## 📚 ChromaDB RAG Pipeline

- **Vector Database**: Embedded [ChromaDB](https://www.trychroma.com/) with native HNSW indexing.
- **Embeddings**: 384-dimensional dense vector representations (`sentence-transformers/all-MiniLM-L6-v2` with deterministic local fallback).
- **Corpus**: Technical knowledge documents covering RAG evaluation, agentic architectures, LLM gateways, vector indexing algorithms, prompt engineering, and guardrails.
- **Chunking Strategy**: Hierarchical paragraph-level chunking with document metadata (doc_id, title, category, tags).
- **Retrieval Output**: Document chunk content, relevance distance score, and citation metadata for source attribution.

---

## 🛠️ Model Context Protocol (MCP) Tool Layer

OpsMind implements a standardized MCP tool layer exposing explicit JSON schemas for tool discovery and parameter validation:

1. **`web_search`**:
   - Schema: `{"query": string, "num_results": integer}`
   - Action: Queries external web documentation or indexed research databases.
2. **`knowledge_search`**:
   - Schema: `{"query": string, "category": string | null, "limit": integer}`
   - Action: Executes vector similarity queries against ChromaDB.
3. **`document_lookup`**:
   - Schema: `{"doc_id": string}`
   - Action: Retrieves complete document sections and metadata by unique ID.

---

## 🌐 LLM Gateway & Task-Based Routing

The LLM Gateway centralizes model access with production reliability patterns:
- **Dynamic Task Routing**:
  - `planning`, `classification`, `critic` → Fast, cost-effective model (`llama-3.1-8b-instant`).
  - `synthesis`, `report_generation` → High-capacity frontier model (`llama-3.3-70b-versatile`).
- **Resilient Fallback Hierarchy**:
  1. Primary: Portkey Gateway / Direct Groq API
  2. Fallback: Secondary Groq key / Direct OpenAI
  3. Offline: High-fidelity Deterministic Reasoning Engine (guaranteeing 100% test and offline execution reliability).

---

## 🛡️ Runtime Guardrails System

Safety policies are enforced at three distinct gates:

1. **Gate 1: Input Guardrails**
   - **Prompt Injection Defense**: Detects adversarial jailbreaks (`Ignore previous instructions`, `DAN mode`, `developer mode override`).
   - **Destructive Command Defense**: Blocks execution commands (`rm -rf`, `DROP TABLE`, `format`).
   - **Out-of-Scope Filter**: Politely filters non-technical queries (jokes, recipes, chit-chat) to save token consumption.
2. **Gate 2: Tool Execution Guardrails**
   - **Parameter Sanitization**: Restricts shell control characters (`;`, `&&`, `|`, `` ` ``) and path traversal (`../`, `/etc/`).
   - **Tool Authorization**: Ensures agents only invoke authorized tools from the registry.
3. **Gate 3: Output Guardrails**
   - **Secret Masking**: Scans responses for API keys (`sk-`, `ghp_`, `Bearer`), redacting them to `[REDACTED_SECRET]`.
   - **Citation Enforcement**: Verifies that every technical assertion is backed by retrieved source tags.

---

## 📊 Evaluation Methodology & Benchmark Results

Evaluations are conducted against a curated **25-case Golden Dataset** (`evals/golden_dataset.json`) containing technical questions, expected answer characteristics, expected evidence, and adversarial security tests.

### Benchmark Results (from `evals/evaluation_results.json`)

| Evaluation Dimension | Metric | Score |
|---|---|---|
| **RAG Retrieval Quality** | **Faithfulness (Groundedness)** | **86.8%** |
| | **Context Recall** | **95.0%** |
| | **Context Relevance** | **35.9%** |
| | **Answer Relevance** | **65.6%** |
| **Agent / Tool Precision** | **Tool Selection Accuracy** | **76.0%** |
| **Safety & Guardrails** | **Guardrail Defense Accuracy** | **100.0%** |
| | **Citation Compliance** | **100.0%** |
| **Performance** | **Average Execution Latency** | **0.02s** |
| **Composite Score** | **Overall Benchmark Score** | **76.6%** |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/Akshat191005/OpsMind.git
cd OpsMind
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)
Create a `.env` file in the project root:
```env
# Optional: Groq API Key for live LLM inference
GROQ_API_KEY=your_groq_api_key_here

# Optional: Portkey Gateway
PORTKEY_API_KEY=your_portkey_api_key_here

# Optional: LangSmith Observability
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=opsmind_research
```
*(Note: OpsMind runs completely offline and locally without requiring any paid API keys using its built-in deterministic reasoning engine and ChromaDB vector store).*

---

## 🧪 Running Tests & Evaluations

### Run All 33 Unit & Integration Tests
```bash
python run_tests.py
```

### Run the Automated 25-Case Evaluation Suite
```bash
python evals/evaluator.py
```

---

## 🖥️ User Interfaces & API

### 1. Launch FastAPI Backend
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Check: `GET http://localhost:8000/health`
- Research Endpoint:
```bash
curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the major approaches to RAG evaluation?"}'
```

### 2. Launch Streamlit Web UI & Evaluation Dashboard
```bash
streamlit run ui/app.py
```
- Research Assistant tab with live execution tracking and verified source inspector.
- Evaluation Dashboard tab with aggregate scorecards, latency distribution, and failure drill-downs.

---

## 📝 Example Research Queries

1. *"What are the major approaches and metrics used for RAG evaluation?"*
2. *"Compare agentic RAG with traditional naive RAG architectures."*
3. *"Explain the key architectural components and advantages of production LLM gateways."*
4. *"How do vector databases perform approximate nearest neighbor search using HNSW?"*
5. *"When should an enterprise choose Fine-Tuning versus Retrieval-Augmented Generation (RAG)?"*

---

## ⚠️ Limitations & Future Work

1. **Dense Retrieval Scope**: Current internal knowledge documents are focused on AI/CS engineering; scaling to thousands of enterprise PDFs requires background batch indexing pipelines.
2. **Re-Ranker Overhead**: Adding a heavy cross-encoder (e.g. `bge-reranker-large`) increases inference latency by ~150ms per query.
3. **Live Web Dependency**: Live web searches require outbound internet connectivity; offline environments automatically utilize the indexed technical database.

---

## 📄 License

MIT License. Designed for academic research, pair programming, and production LLM engineering demonstrations.
