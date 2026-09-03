# OpsMind — Architecture Design & Technical Specifications

This document outlines the complete architectural design, data flow, multi-agent coordination, and engineering specifications behind **OpsMind — Multi-Agent Research & Knowledge System**.

---

## 1. Architectural Principles

OpsMind implements a production-oriented multi-agent architecture designed to replace brittle single-shot RAG with a verifiable, multi-step investigation pipeline:
1. **Dynamic Query Decomposition**: Breaking down multi-faceted research topics into targeted sub-queries.
2. **Dense Vector Retrieval (RAG)**: Indexing technical computer science and AI documents into ChromaDB with 384-d Hugging Face SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
3. **Model Context Protocol (FastMCP)**: Standardizing agent tool execution via the official `mcp.server.fastmcp.FastMCP` protocol.
4. **Evidence Evaluation & Critic**: Verifying factual alignment, detecting contradictions, and scoring confidence before report generation.
5. **Runtime Safety Guardrails**: Deterministic 3-tier validation across Input, Tool execution, and Output phases.
6. **Task-Based LLM Gateway**: Routing fast classification and critic tasks to `llama-3.1-8b-instant` and report synthesis to `llama-3.3-70b-versatile`, with graceful local fallback.
7. **Dual-Layer Evaluation**: Supporting both official RAGAS framework metrics (`ragas.metrics`) and fast deterministic heuristic benchmarks.

---

## 2. Multi-Agent Topology (LangGraph)

```
                            [ User Query ]
                                  |
                                  v
                     [ Gate 1: Input Guardrail ]
                      /                       \
             (Violations)                   (Passed)
                  v                            v
       [ Refusal Response ]           [ Query Planner ]
                                        /           \
                                       v             v
                           [ Web Research ]   [ RAG Knowledge ]
                           (FastMCP: web_search) (FastMCP: knowledge_search)
                                       \             /
                                        v           v
                                    [ Evidence Critic ]
                                              |
                                              v
                                  [ Report Generator ]
                                              |
                                              v
                                [ Gate 3: Output Guardrail ]
                                              |
                                              v
                                 [ Structured Report ]
```

### State Definition (`ResearchState`)
The LangGraph state is tracked in `app/agents/state.py` using Python `TypedDict` with explicit merge operators and per-node latency telemetry:
```python
class ResearchState(TypedDict):
    query: str
    thread_id: Optional[str]
    plan: List[str]
    need_rag: bool
    need_web: bool
    search_queries: List[str]
    focus_areas: List[str]
    current_step: str
    status: str
    rag_findings: List[Dict[str, Any]]
    web_findings: List[Dict[str, Any]]
    observed_evidence: Annotated[List[str], operator.add]
    verified_sources: List[Dict[str, Any]]
    critic_score: float
    confidence: float
    critic_notes: str
    report: str
    agents_used: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    latencies: Dict[str, float]
    guardrail_blocked: bool
    guardrail_response: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]
    final_output: Dict[str, Any]
```

---

## 3. Component Specifications

### 3.1 Query Planner Node (`app/agents/nodes/query_planner.py`)
- **LLM Tier**: `llama-3.1-8b-instant` (Fast routing)
- **Role**: Deconstructs research queries into sub-questions, identifies focus areas, and determines whether RAG, Web, or both retrieval mechanisms are necessary.
- **Latency Tracking**: Records duration under `latencies["query_planner"]`.

### 3.2 Web Research Node (`app/agents/nodes/web_research_agent.py`)
- **Tool Protocol**: FastMCP `web_search`
- **Role**: Executes web search across planned sub-queries to acquire fresh external literature, benchmarks, and release notes.
- **Latency Tracking**: Records duration under `latencies["web_research"]`.

### 3.3 Knowledge / RAG Node (`app/agents/nodes/rag_knowledge_agent.py`)
- **Tool Protocol**: FastMCP `knowledge_search` & `document_lookup`
- **Vector DB**: ChromaDB embedded collection (`opsmind_knowledge_base`)
- **Role**: Queries vector database with dense embeddings to fetch relevant internal technical documents with relevance scores.
- **Latency Tracking**: Records duration under `latencies["rag_knowledge"]`.

### 3.4 Evidence Critic Node (`app/agents/nodes/evidence_critic_agent.py`)
- **LLM Tier**: `llama-3.1-8b-instant` (Evaluation)
- **Role**: Cross-references RAG chunks and Web snippets, flags contradictory assertions, computes confidence score (0.0 to 1.0), and compiles source references.
- **Latency Tracking**: Records duration under `latencies["evidence_critic"]`.

### 3.5 Report Generator Node (`app/agents/nodes/report_generator_agent.py`)
- **LLM Tier**: `llama-3.3-70b-versatile` (Frontier Synthesis)
- **Role**: Synthesizes verified evidence into an Executive Summary, Technical Architecture Analysis, Comparative Trade-offs, and explicit source citations (`[Source: doc-id]`).
- **Latency Tracking**: Records duration under `latencies["report_generator"]`.

---

## 4. Model Context Protocol (FastMCP) Layer

Tool discovery and invocation are managed through the official FastMCP SDK:
- **Server** (`app/tools/mcp_server.py`):
  Initializes `FastMCP("OpsMind-Research-Tools")` and registers tools with type annotations and docstrings.
- **Client** (`app/tools/mcp_client.py`):
  Introspects registered tools and executes calls synchronously with parameter schema verification.
- **Exported Tools**:
  1. `web_search(query: str, num_results: int = 3)`
  2. `knowledge_search(query: str, category: str = None, limit: int = 4)`
  3. `document_lookup(doc_id: str)`

---

## 5. ChromaDB & SentenceTransformer Pipeline

- **Vector Database**: Embedded [ChromaDB](https://www.trychroma.com/) with native HNSW approximate nearest neighbor indexing.
- **Embeddings**:
  - `HuggingFaceSentenceTransformerEmbeddingFunction`: Real `sentence-transformers/all-MiniLM-L6-v2` producing normalized 384-dimensional dense vectors.
  - `DeterministicEmbeddingFunction`: Normalized 384-d deterministic hash embeddings for fast local tests when `USE_DETERMINISTIC_MODE=true`.
- **Status Introspection**: Exposes `chroma_service.get_status()` returning active embedding backend and document count.

---

## 6. Three-Tier Runtime Safety Guardrails

Safety policies are enforced at three distinct gates:
1. **Input Rails** (`app/guardrails/rails.py`):
   - Regex and heuristic classifiers for adversarial prompt injections (`Ignore all instructions`, `DAN mode`, `developer mode override`).
   - Destructive command filters (`rm -rf`, `DROP TABLE`).
   - Out-of-scope chit-chat refusal (recipes, jokes, general chat) to maintain domain focus.
2. **Tool Rails**:
   - Parameter sanitization against shell control characters (`;`, `&&`, `|`, `` ` ``) and directory traversal (`../`).
3. **Output Rails**:
   - Token pattern scanner to mask sensitive credentials (`sk-`, `ghp_`, `Bearer`) into `[REDACTED_SECRET]`.
   - Groundedness check verifying cited source tags.

---

## 7. Dual Evaluation Framework

- **Golden Dataset**: 25 curated test cases in `evals/golden_dataset.json`.
- **RAGAS Evaluation Layer**:
  - Official `ragas.metrics`: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
  - Executed via `ragas.evaluate` when `USE_LIVE_LLM=true` and LLM credentials are provided.
- **Deterministic Heuristic Layer**:
  - `custom_faithfulness_heuristic`: Sentence-level context overlap support ratio.
  - `custom_answer_relevance_heuristic`: Query-answer keyword alignment and structure completeness.
  - `custom_context_relevance_heuristic`: Retrieval chunk density.
  - `custom_context_recall_heuristic`: Ground truth evidence matching.
  - `custom_tool_selection_accuracy`: Jaccard index between expected and actual tools.
  - `custom_guardrail_accuracy`: Binary precision on security test cases.
  - `custom_citation_compliance`: Verifies presence of explicit source citations.
