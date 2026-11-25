# OpsMind — Architecture

This document describes how a research question flows through OpsMind and where each part lives in the code. Configuration options are listed in the README; every value below marked `LIKE_THIS` is a setting in `app/config.py` that can be overridden in `.env`.

---

## 1. Request flow

```
POST /research  (app/main.py)          or the web UI (ui/index.html, served at /)
        │
        ▼
investigate_query()  (app/agents/graph.py)
  ├─ length check (MAX_QUERY_CHARS) ─────────────► HTTP 400
  ├─ input guardrail (app/guardrails/rails.py) ──► refusal
  ▼
LangGraph state machine
  query_planner ──out of scope──► END (refusal)
      │ need_web / need_rag (fan-out; parallel when both)
      ├──► web_research ──┐
      └──► rag_knowledge ─┤
                          ▼
                   evidence_critic ──follow_up_queries──► web_research / rag_knowledge  (≤ MAX_RESEARCH_ROUNDS)
                          ▼
                   report_generator ◄──revision_pending──┐  (≤ MAX_REPORT_REVISIONS)
                          ▼                              │
                   claim_verifier ───────────────────────┘
                          ▼
                         END
```

Routing functions: `route_after_planning`, `route_after_critic`, `route_after_report`, `route_after_verification` in `app/agents/graph.py`.

## 2. State

`ResearchState` (`app/agents/state.py`) is a `TypedDict`. Fields written by the parallel retrieval branches declare reducers:
- `latencies`: merged dicts (per node; later rounds use suffixed keys such as `rag_r2`)
- `token_usage`: summed per agent
- `agents_used`, `tools_used`, `warnings`, `messages`, `observed_evidence`: appended lists
- `source_verdicts`: merged map of source key → accepted, kept across critic rounds so sources are judged once
- `current_step`: last value wins

## 3. Agents

| Node | File | Model | Output |
|---|---|---|---|
| Query Planner | `nodes/query_planner.py` | `GROQ_FAST_MODEL` | `in_scope`, `need_rag`, `need_web`, `needs_recent`, `search_queries` (≤ `MAX_SEARCH_QUERIES`), plan |
| Web Research | `nodes/web_research_agent.py` | — | `web_findings` (deduped by URL, ≤ `MAX_CONTEXT_SOURCES`) |
| Knowledge / RAG | `nodes/rag_knowledge_agent.py` | — | `rag_findings` (≥ `MIN_RELEVANCE_SCORE`, ≤ `MAX_CONTEXT_SOURCES`) |
| Evidence Critic | `nodes/evidence_critic_agent.py` | `GROQ_FAST_MODEL` | per-source verdicts, `evidence_quality`, gaps, contradictions, optional `followup_queries` |
| Report Generator | `nodes/report_generator_agent.py` | `GROQ_MODEL` | `report`, validated `citations` |
| Claim Verifier | `nodes/claim_verifier_agent.py` | `VERIFIER_MODEL` | `claims`, `claim_support`, final `confidence`, `revision_pending` |

Prompts for the critic, report generator and verifier mark retrieved text as untrusted data. The planner and report prompts include today's date; the planner flags time-sensitive questions (`needs_recent`), which restricts web search to `WEB_RECENT_TIME_RANGE`.

**Confidence** = `evidence_quality × claim_support`:
- `evidence_quality`: mean retrieval relevance of accepted sources, averaged with the critic's coverage estimate
- `claim_support`: (supported + ½·partial + ½·uncited) / claims checked
- With no accepted evidence, confidence is 0.

## 4. Retrieval

- **Knowledge base** (`app/services/retrieval/chroma_service.py`): built-in documents (`app/data/knowledge_base.py`) plus `.md`/`.txt` files in `KNOWLEDGE_DIR`. Documents are chunked (`CHUNK_SIZE`, `CHUNK_OVERLAP`) and embedded with `EMBEDDING_MODEL` (or hash embeddings when `USE_DETERMINISTIC_MODE=true`). They are stored in ChromaDB under `CHROMA_PERSIST_DIR` with cosine distance. A fingerprint of the corpus, chunk settings and embedding backend triggers an automatic rebuild when any of them change.
- **Scoring**: `HYBRID_VECTOR_WEIGHT × cosine + (1 − weight) × query-term coverage`; candidates are over-fetched and re-ranked.
- **Web** (`app/tools/mcp_server.py`): `ACTIVE_WEB_PROVIDER` is `tavily` (API, when `TAVILY_API_KEY` is set), `ddgs` (scraping fallback), or `offline_index` (tests). Failures return an error result and never fabricated content.

## 5. Tools (MCP)

FastMCP server `OpsMind-Research-Tools` (`app/tools/mcp_server.py`) exposes `web_search`, `knowledge_search` and `document_lookup`. `app/tools/mcp_client.py` applies the tool guardrail before calling the registered function.

## 6. LLM gateway

`app/gateway/client.py`:
- `get_llm(task_type)` picks the model tier (fast: planning, critic, verification; strong: synthesis) and the backend (Portkey if `PORTKEY_API_KEY`, else Groq). It applies `LLM_MAX_OUTPUT_TOKENS_*`, `LLM_REASONING_EFFORT` and `LLM_MAX_RETRIES`; rate-limited calls wait the time Groq asks for.
- In live mode a failed call raises; the API returns 502. `ALLOW_OFFLINE_FALLBACK=true` substitutes the offline engine instead.
- With `USE_LIVE_LLM=false`, `DeterministicReasoningLLM` derives planner, critic, report and verifier output from the prompt itself (used by the test suite).
- `get_judge_llm()` returns the evaluation judge (`JUDGE_MODEL`, `JUDGE_API_KEY`, optional `JUDGE_BASE_URL`).

## 7. Guardrails

`app/guardrails/rails.py`:
1. **Input**: injection, destructive-command and chit-chat patterns (`GUARDRAILS_ENABLED`).
2. **Topic**: the planner's `in_scope` decision against `RESEARCH_SCOPE` (`TOPIC_GUARD_ENABLED`).
3. **Tool**: strict identifiers (`doc_id`, `category`); free text rejects command sequences, path traversal and control characters.
4. **Output**: secret redaction, then `validate_citations`. This normalizes citation styles and drops ids that are not verified sources.

## 8. Evaluation

`evals/evaluator.py` runs `evals/golden_dataset.json` through `investigate_query` and scores each case:
- **Deterministic** (`evals/metrics.py`): guardrail accuracy, routing recall, critic source recall/precision, citation use.
- **Judge** (`evals/judge.py`): correctness against `expected_answer_characteristics`, faithfulness to sources, relevance.

Errors are recorded separately from scores; each report stores the configuration used (`evals/evaluation_results.json`).

## 9. Limits to know

- Groq free tier: ~8,000 tokens/minute and 200,000 tokens/day per model. One question uses ~10k–30k tokens, mostly on the fast model.
- `thread_id` is accepted but there is no conversation memory between questions.
- The API has no authentication; bind it to `127.0.0.1` unless you add one.
