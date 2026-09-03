"""
Knowledge Base Corpus for OpsMind Research & Knowledge System.
Contains comprehensive, technical documents on production LLM engineering topics.
"""

from typing import List, Dict, Any

KNOWLEDGE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc-rag-eval-01",
        "title": "RAG Evaluation Methodologies and Frameworks",
        "category": "evaluation",
        "tags": ["rag", "evaluation", "ragas", "faithfulness", "llm-judge"],
        "summary": "Overview of RAG evaluation metrics including Faithfulness, Answer Relevance, Context Precision, and Context Recall using RAGAS and LLM-as-a-Judge paradigms.",
        "content": """
RAG (Retrieval-Augmented Generation) systems require multi-component evaluation separating retrieval quality from generation quality.
1. The RAG Triad & RAGAS Metrics:
   - Faithfulness (Groundedness): Measures whether the generated answer is strictly derived from the retrieved context without hallucinating external ungrounded facts. Formula: (Number of factual claims in answer supported by context) / (Total claims in answer).
   - Answer Relevance: Measures whether the generated response directly and concisely addresses the user's initial prompt, penalizing incomplete, redundant, or tangential responses.
   - Context Relevance (Context Precision): Evaluates signal-to-noise ratio in retrieved chunks. Verifies whether all retrieved chunks are relevant to the query and whether relevant chunks are ranked higher.
   - Context Recall: Evaluates whether the retrieval mechanism captured all ground-truth facts necessary to construct the complete gold answer.
2. Evaluation Frameworks:
   - RAGAS (Retrieval Augmented Generation Assessment): Automated evaluation library using LLMs to score component metrics without requiring expensive human ground-truth labels for every test sample.
   - TruLens: Focuses on the RAG Triad (Context Relevance, Groundedness, Answer Relevance) with real-time feedback functions and tracing.
   - DeepEval / G-Eval: Uses chain-of-thought prompting for LLM-as-a-judge evaluation with customizable rubrics.
3. Best Practices in Production:
   - Separate retrieval evaluation (NDCG@k, MRR, Context Recall) from generation evaluation (Faithfulness, LLM-as-judge correctness).
   - Curate a golden dataset of 25-100 high-quality query-context-answer triples with domain experts.
   - Automate regression testing in CI/CD pipelines to catch retrieval or prompt degradations prior to deployment.
"""
    },
    {
        "id": "doc-agentic-rag-02",
        "title": "Agentic RAG vs Traditional Naive RAG Architectures",
        "category": "architecture",
        "tags": ["agentic-rag", "langgraph", "query-planning", "multi-agent"],
        "summary": "Deep architectural comparison between static single-shot Traditional RAG and iterative, multi-hop Agentic RAG workflows.",
        "content": """
Traditional RAG follows a rigid, linear pipeline: User Query -> Embed -> Top-k Vector Search -> Prompt Augmentation -> Single LLM Generation.
While effective for simple fact-retrieval, Traditional RAG suffers from severe limitations:
1. Limitations of Traditional RAG:
   - Single-shot retrieval fails when a complex query requires multi-hop reasoning or multiple distinct data sources.
   - Semantic vector similarity does not guarantee factual relevance (retrieval noise).
   - Inability to self-correct if the retrieved documents do not contain the answer.
2. What is Agentic RAG?
   - Agentic RAG embeds LLMs with autonomous control loops (ReAct, Planning, Reflection) to dynamically orchestrate retrieval.
   - Query Deconstruction / Planning: Breaks complex research questions into sub-queries.
   - Routing: Dynamically decides whether to query a vector database, perform web search, or execute structured SQL queries.
   - Iterative Retrieval: If initial context is insufficient or contradictory, the agent reformulates the query and retrieves additional context.
   - Self-Correction & Evidence Evaluation: Evaluates retrieved chunks before generation. If retrieved chunks fail relevance checks, the agent rejects them.
3. Implementation Frameworks:
   - LangGraph provides stateful, graph-based multi-agent orchestration with conditional routing, cyclical execution loops, and checkpointing.
   - LlamaIndex Workflows / Agentic Strategies (Sub-question query engine, Router Query Engine).
"""
    },
    {
        "id": "doc-llm-gateway-03",
        "title": "Production LLM Gateway Architectures & Design Patterns",
        "category": "infrastructure",
        "tags": ["gateway", "portkey", "litellm", "routing", "fallbacks", "caching"],
        "summary": "Engineering principles of LLM Gateway layers: model tiering, resilient fallbacks, semantic caching, rate limiting, and observability.",
        "content": """
An LLM Gateway serves as a centralized reverse proxy and orchestration layer between client applications and downstream LLM model providers (OpenAI, Anthropic, Groq, local models).
Key Core Capabilities of an LLM Gateway:
1. Dynamic Task-Based Routing & Model Tiering:
   - Routes simple tasks (query classification, planning, sentiment analysis, entity extraction) to fast, cost-effective models (e.g., Llama-3.1-8B, Claude 3 Haiku, GPT-4o-mini).
   - Routes high-complexity tasks (deep synthesis, multi-agent report generation, complex code generation) to frontier reasoning models (e.g., Llama-3.3-70B, GPT-4o, Claude 3.5 Sonnet).
2. Resilient Fallbacks & Circuit Breaking:
   - Automatically catches 429 (Rate Limit), 500/503 (Server Error), or network timeout exceptions from the primary provider and seamlessly retries with a designated fallback model/provider without user interruption.
3. Semantic Caching:
   - Computes embedding similarity on incoming prompts. If cosine similarity to a cached query exceeds a threshold (e.g., 0.95), the gateway returns the cached response instantly, reducing latency from 2000ms to <50ms and cutting inference costs.
4. Load Balancing & Rate Limiting:
   - Distributes requests across multiple API keys or provider regions using round-robin or least-latency balancing. Enforces token-bucket rate limits per user/organization.
5. Unified Observability:
   - Standardizes telemetry, logging token consumption, prompt latency, cache hit/miss ratio, and cost tracking across all upstream model providers.
"""
    },
    {
        "id": "doc-vector-search-04",
        "title": "Vector Databases, Embeddings, and Indexing Algorithms",
        "category": "retrieval",
        "tags": ["chromadb", "pgvector", "faiss", "hnsw", "embeddings", "dense-retrieval"],
        "summary": "Technical deep dive into vector databases (ChromaDB, PGVector, FAISS), dense embeddings, and approximate nearest neighbor (ANN) search algorithms.",
        "content": """
Vector search enables semantic information retrieval by representing unstructured text documents as high-dimensional dense numerical vectors (embeddings).
1. Dense Embedding Models:
   - Hugging Face sentence-transformers (e.g., all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5) map text into a dense 384 or 768-dimensional space where semantic similarity corresponds to geometric proximity.
2. Distance Metrics:
   - Cosine Similarity: Measures the cosine of the angle between two normalized vectors, independent of vector magnitude. Ideal for text retrieval.
   - Dot Product: Equivalent to cosine similarity when vectors are unit-normalized; computationally faster.
   - Euclidean Distance (L2): Measures straight-line geometric distance; sensitive to vector magnitude.
3. Approximate Nearest Neighbor (ANN) Indexing:
   - Exact k-NN (brute force) computes distance to every vector in the dataset (O(N*D)), which becomes prohibitively slow at scale.
   - HNSW (Hierarchical Navigable Small World): Graph-based multi-layer indexing providing logarithmic search time (O(log N)) with high recall accuracy (>95%).
   - IVF (Inverted File Index): Partitions vector space into Voronoi cells using k-means clustering to search only a fraction of clusters.
4. Comparison of Vector Stores:
   - ChromaDB: Lightweight, embedded, developer-friendly vector database with native Python support, SQLite/duckdb storage, and built-in HNSW indexing. Excellent for local development and modular applications.
   - FAISS (Facebook AI Similarity Search): High-performance C++/Python library optimized for GPU/CPU in-memory vector indexing.
   - PGVector: PostgreSQL extension enabling vector similarity search alongside relational data, transactional ACID guarantees, and SQL joins.
"""
    },
    {
        "id": "doc-llm-guardrails-05",
        "title": "LLM Safety Guardrails: Input, Tool, and Output Defenses",
        "category": "security",
        "tags": ["guardrails", "nemo", "prompt-injection", "pii-masking", "tool-safety"],
        "summary": "Three-tier runtime guardrails architecture covering input sanitization, tool call validation, and output grounding verification.",
        "content": """
Production LLM applications must enforce deterministic safety policies at runtime through multi-tiered guardrail layers.
1. Layer 1: Input Guardrails (Pre-Execution):
   - Prompt Injection & Jailbreak Defense: Detects adversarial attempts to override system prompts (e.g., 'Ignore previous instructions', 'DAN mode', 'Developer mode override') using regex heuristics and specialized classifier models.
   - Malicious Instruction Detection: Blocks requests to execute destructive shell commands (rm -rf, format, DROP TABLE) or malicious exploits.
   - Scope / Domain Alignment: Filters out-of-scope or irrelevant queries before spending expensive LLM tokens.
2. Layer 2: Tool Execution Guardrails (In-Execution):
   - Parameter & Argument Validation: Uses strict Pydantic schemas to validate types, ranges, and formats of tool inputs.
   - Injection Prevention: Sanitizes tool arguments to prevent SQL injection, shell command chaining (;, &&, |), and path traversal (../).
   - Tool Authorization Whitelisting: Restricts which tools an agent can invoke based on agent role permissions.
3. Layer 3: Output Guardrails (Post-Execution):
   - Sensitive Data / PII Redaction: Regex and entropy scanning to detect and redact API keys, JWT tokens, passwords, and personally identifiable information.
   - Groundedness & Hallucination Check: Verifies that factual assertions in the generated response correspond to evidence gathered from retrieval.
   - Citation Enforcement: Ensures that technical reports include explicit source citations for all claims.
"""
    },
    {
        "id": "doc-multi-agent-06",
        "title": "Multi-Agent System Orchestration Patterns in LangGraph",
        "category": "architecture",
        "tags": ["multi-agent", "langgraph", "state-machine", "critic", "planner"],
        "summary": "Patterns for multi-agent collaboration, state management, cyclical graph topologies, and supervisor vs peer communication.",
        "content": """
Multi-agent systems deconstruct complex knowledge workflows into specialized agents with explicit boundaries and responsibilities.
1. Multi-Agent Design Patterns:
   - Supervisor Pattern: A central coordinator agent receives user input, creates a task plan, delegates sub-tasks to specialized worker agents, aggregates results, and decides termination.
   - Plan-and-Execute (Planner -> Workers -> Critic -> Synthesizer): Decouples goal planning from tool execution and evaluation.
   - Reflection & Critic Pattern: A dedicated critic agent evaluates outputs produced by upstream worker agents, identifies flaws, missing citations, or contradictions, and routes feedback back for refinement.
2. LangGraph Architecture:
   - Stateful Graph: Graph execution state is defined via a TypedDict with explicit merge operators (e.g., operator.add for message/evidence aggregation).
   - Nodes: Python functions executing agent logic, tool calls, or gateway invocations.
   - Conditional Edges: Decision functions that inspect state and route execution to appropriate subsequent nodes.
   - Checkpointing & Persistence: Enables resuming workflows, human-in-the-loop validation, and replayable execution history.
3. Benefits over Monolithic Prompts:
   - Modularity: Each agent prompt is focused on one specialized task (planning, searching, critiquing, report writing), reducing cognitive load and prompt degradation.
   - Traceability: Distinct graph transitions make debugging, evaluation, and latency profiling straightforward.
"""
    },
    {
        "id": "doc-prompt-engineering-07",
        "title": "Advanced Prompt Engineering and Context Management",
        "category": "prompting",
        "tags": ["prompt-engineering", "cot", "react", "context-compression"],
        "summary": "Techniques for structuring LLM prompts: Chain-of-Thought, ReAct, system role definition, few-shot formatting, and context optimization.",
        "content": """
High-performing LLM applications rely on structured prompt engineering paradigms:
1. ReAct (Reasoning + Acting):
   - Interleaves step-by-step reasoning ('Thought') with tool execution ('Action' and 'Observation') allowing agents to formulate dynamic plans and adjust based on intermediate tool outputs.
2. Chain-of-Thought (CoT) Prompting:
   - Encourages LLMs to generate intermediate reasoning steps before arriving at a final conclusion, significantly boosting accuracy on multi-step logic, mathematical reasoning, and synthesis.
3. System Role & Delimiter Formatting:
   - Explicit persona definition with clear XML/Markdown delimiters (`### Task`, `### Rules`, `### Context`) prevents instruction leakage and improves structural adherence.
4. Context Window Management:
   - Re-ranking & Context Selection: Applying Cross-Encoder re-rankers (e.g., bge-reranker-large) to filter top-k chunks down to the top-3 most relevant chunks before prompt injection.
   - Context Compression: Removing boilerplate and whitespace to fit maximum information within model context limits.
"""
    },
    {
        "id": "doc-finetuning-vs-rag-08",
        "title": "Fine-Tuning vs RAG: Trade-offs, Decision Matrix, and Hybrid Approaches",
        "category": "architecture",
        "tags": ["rag", "finetuning", "tradeoffs", "knowledge-freshness", "catastrophic-forgetting"],
        "summary": "Comprehensive trade-off analysis between Fine-Tuning and Retrieval-Augmented Generation for enterprise AI applications.",
        "content": """
When building enterprise AI systems, choosing between RAG and Fine-Tuning depends on specific operational requirements:
1. Retrieval-Augmented Generation (RAG):
   - Best For: Dynamic knowledge bases, frequently updated documents, strict factual groundedness, traceable citations, and low initial compute investment.
   - Strengths: Real-time data freshness, zero retraining cost when documents change, transparent auditability via source citations, strong hallucination reduction.
   - Weaknesses: Latency overhead from vector search and multi-step retrieval, dependency on retrieval quality.
2. Fine-Tuning (SFT / LoRA / QLoRA):
   - Best For: Teaching specific output formats, specialized domain tone/style, domain syntax (e.g., code/SQL dialects), and reducing prompt token overhead.
   - Strengths: Fast inference latency, baked-in stylistic alignment, lower per-request prompt token usage.
   - Weaknesses: Knowledge cutoff is static; fine-tuning does not reliably inject new facts without catastrophic forgetting; expensive GPU training pipelines.
3. Hybrid Architecture (RAG + Fine-Tuning):
   - Fine-tune a smaller model (e.g., Llama-3.1-8B) on domain formatting, tool calling schemas, and citation generation, while using RAG to dynamically provide fresh knowledge context.
"""
    }
]

# Web research synthetic knowledge index for external topics and real-time facts
WEB_KNOWLEDGE_INDEX: List[Dict[str, Any]] = [
    {
        "id": "web-ragas-latest",
        "title": "RAGAS 0.2 Framework Release & Multi-Turn Metrics",
        "category": "web_article",
        "url": "https://docs.ragas.io/en/latest/concepts/metrics/",
        "snippet": "RAGAS 0.2 introduces enhanced LLM-based evaluation metrics including Aspect Critique, Multi-turn Conversation Faithfulness, Semantic Similarity, and Agent Goal Accuracy with LangSmith and OpenTelemetry integrations.",
        "keywords": ["ragas", "evaluation", "metrics", "rag", "faithfulness", "aspect critique"]
    },
    {
        "id": "web-langgraph-features",
        "title": "LangGraph: Multi-Agent Orchestration with Cyclic Graphs",
        "category": "web_article",
        "url": "https://langchain-ai.github.io/langgraph/",
        "snippet": "LangGraph enables stateful multi-agent workflows with built-in cycles, conditional branching, time-travel debugging, human-in-the-loop state review, and streaming token support for production applications.",
        "keywords": ["langgraph", "agent", "workflow", "multi-agent", "state", "orchestration"]
    },
    {
        "id": "web-portkey-gateway",
        "title": "AI Gateway Architecture: Portkey LLM Routing & Guardrails",
        "category": "web_article",
        "url": "https://docs.portkey.ai/docs/product/ai-gateway",
        "snippet": "Portkey AI Gateway provides a unified API for 200+ LLMs with automatic fallbacks, load balancing, semantic caching, budget limits, runtime guardrails, and full request-response logging.",
        "keywords": ["portkey", "gateway", "llm gateway", "routing", "fallback", "caching", "rate limiting"]
    },
    {
        "id": "web-vector-benchmarks",
        "title": "Vector Database Benchmarks: ChromaDB, Qdrant, PGVector, FAISS",
        "category": "web_article",
        "url": "https://ann-benchmarks.com/vector-db-comparison",
        "snippet": "ANN benchmarks reveal HNSW indexing delivers sub-10ms p99 query latency with >96% recall across ChromaDB, FAISS, and PGVector. ChromaDB excels at embedded developer ergonomics and modular integration.",
        "keywords": ["vector database", "chromadb", "pgvector", "faiss", "hnsw", "indexing", "benchmark"]
    },
    {
        "id": "web-guardrails-nemoguard",
        "title": "NeMo Guardrails: Programmable Rails for LLM Safety",
        "category": "web_article",
        "url": "https://github.com/NVIDIA/NeMo-Guardrails",
        "snippet": "NeMo Guardrails enables developers to define Colang and Python programmable rails to enforce conversational flow, topical boundaries, jailbreak prevention, and output credential masking.",
        "keywords": ["guardrails", "nemo", "safety", "injection", "colang", "pii", "security"]
    }
]
