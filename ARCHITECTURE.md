# OpsMind — Architecture Design & Technical Specifications

This document outlines the complete architectural design, data flow, multi-agent coordination, and engineering decisions behind **OpsMind — Agentic Research & Knowledge System**.

---

## 1. System Overview & Core Tenets

OpsMind is an Agentic AI Research and Knowledge Assistant built on production-grade LLM engineering patterns. It replaces fragile single-shot RAG with a stateful multi-agent system orchestrating:
1. **Query Planning & Deconstruction**: Breaking down complex research topics into targeted sub-queries.
2. **Dense Vector Retrieval (RAG)**: Indexing technical computer science and AI documents into ChromaDB using HuggingFace embeddings.
3. **External Web Research**: Augmenting internal documents with real-time web literature via Model Context Protocol (MCP) tools.
4. **Evidence Evaluation & Critic**: Verifying factual alignment, detecting contradictions, and scoring grounded confidence.
5. **Runtime Guardrails**: Deterministic safety checks across Input, Tool execution, and Output phases.
6. **Task-Based LLM Routing**: Optimizing cost and latency by routing routine tasks to lightweight models and reserving high-capacity models for synthesis.

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
                           (MCP: web_search)  (MCP: knowledge_search)
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
The LangGraph state is tracked using Python `TypedDict` with explicit merge operators:
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
    guardrail_blocked: bool
    guardrail_response: Optional[str]
    messages: Annotated[List[Dict[str, str]], operator.add]
    final_output: Dict[str, Any]
```

---

## 3. Component Breakdown

### 3.1 Query Planner Node
- **File**: `app/agents/nodes/query_planner.py`
- **Role**: Analyzes the query, formulates sub-queries, determines whether RAG, Web, or both are needed, and generates an execution plan.

### 3.2 Web Research Node
- **File**: `app/agents/nodes/web_research_agent.py`
- **Role**: Invokes the `web_search` MCP tool to gather external literature, benchmarks, and release notes.

### 3.3 Knowledge / RAG Node
- **File**: `app/agents/nodes/rag_knowledge_agent.py`
- **Role**: Executes dense vector similarity search against ChromaDB using the `knowledge_search` MCP tool and retrieves document chunks with source metadata.

### 3.4 Evidence Critic Node
- **File**: `app/agents/nodes/evidence_critic_agent.py`
- **Role**: Evaluates factual consistency across RAG and web findings, calculates confidence score (0.0 to 1.0), filters ungrounded claims, and prepares citation records.

### 3.5 Report Generator Node
- **File**: `app/agents/nodes/report_generator_agent.py`
- **Role**: Synthesizes verified evidence into an Executive Summary, Technical Deep Dive, Comparative Trade-offs, and explicit source citations `[Source: ...]`.

---

## 4. MCP (Model Context Protocol) Layer

All external tool invocations conform to the Model Context Protocol standard with JSON Schema validation:
```json
{
  "name": "knowledge_search",
  "description": "Performs dense semantic vector similarity search in ChromaDB.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "category": {"type": "string", "nullable": true},
      "limit": {"type": "integer", "default": 3}
    },
    "required": ["query"]
  }
}
```

---

## 5. Three-Tier Runtime Guardrails

1. **Input Rails** (`app/guardrails/rails.py`):
   - Regex and heuristic classifiers for adversarial prompt injections (`Ignore all instructions`, `DAN mode`).
   - Destructive command filters (`rm -rf`, `DROP TABLE`).
   - Out-of-scope chit-chat refusal (recipes, jokes, weather).
2. **Tool Rails**:
   - Parameter sanitization against shell control characters (`;`, `&&`, `|`) and path traversal (`../`).
3. **Output Rails**:
   - Pattern scanning to redact exposed API tokens (`sk-`, `ghp_`, `Bearer`).
   - Groundedness check ensuring cited evidence presence.

---

## 6. Evaluation Framework

- **Golden Dataset**: 25 curated test cases in `evals/golden_dataset.json`.
- **Metrics**:
  - **Faithfulness**: Proportion of generated claims grounded in retrieved text.
  - **Context Recall**: Coverage of gold expected evidence items.
  - **Answer Relevance**: Alignment with user prompt.
  - **Tool Selection Accuracy**: Correctness of tools invoked by agents.
  - **Guardrail Accuracy**: Precision and recall of security block decisions.
- **Evaluation Runner**: `python evals/evaluator.py` exports results directly to `evals/evaluation_results.json` for visualization in the Streamlit Evaluation Dashboard.
