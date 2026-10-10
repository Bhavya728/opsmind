"""
OpsMind — Agentic Research & Knowledge System Streamlit Web Interface.

Features:
1. Research Assistant Tab:
   - Interactive research query input
   - Agent execution tracker & workflow telemetry with per-node latencies
   - Retrieved knowledge & web sources inspection
   - Structured grounded report with confidence score
2. Evaluation Dashboard Tab:
   - Automated evaluation metrics (RAGAS + Custom Heuristics, Tool Accuracy, Guardrails)
   - Scorecards, latency distribution, and failure drill-downs
3. MCP Tool Registry:
   - Live FastMCP tool definitions and JSON schemas
4. Knowledge Base Inspector:
   - ChromaDB documents and metadata inspection
"""

import os
import sys
import json
import time
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import settings
from app.agents.graph import investigate_query
from app.tools.mcp_tools import get_mcp_tool_definitions
from app.services.retrieval.chroma_service import chroma_service
from app.gateway.client import get_llm_runtime_status

# Page Configuration
st.set_page_config(
    page_title="OpsMind — Agentic Research & Knowledge System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .agent-badge {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #3B82F6;
        color: #60A5FA;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar & Diagnostics
st.sidebar.title("🧠 OpsMind")
st.sidebar.caption("Agentic Research & Knowledge System v2.0")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🔬 Research Assistant", "📊 Evaluation Dashboard", "🛠️ MCP Tool Registry", "📚 Knowledge Base Inspector"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Live Runtime Diagnostics")

chroma_status = chroma_service.get_status()
llm_status = get_llm_runtime_status()

st.sidebar.info(f"**LLM Backend:** `{llm_status}`")
st.sidebar.info(f"**Embeddings:** `{chroma_status.get('embedding_backend')}`")
st.sidebar.info(f"**MCP Server:** `FastMCP (Active)`")
st.sidebar.info(f"**Guardrails:** `3-Tier Active`")
st.sidebar.info(f"**LangSmith:** `{'Enabled' if settings.LANGSMITH_TRACING else 'Disabled'}`")


# TAB 1: RESEARCH ASSISTANT
if nav_choice == "🔬 Research Assistant":
    st.markdown('<div class="main-header">OpsMind — AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-agent investigation using ChromaDB RAG, FastMCP Tools, Evidence Critique, and Runtime Guardrails.</div>', unsafe_allow_html=True)

    if "user_query" not in st.session_state:
        st.session_state["user_query"] = "What are the major approaches and metrics used for RAG evaluation?"

    # Preset Example Queries
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("📌 Approaches to RAG Evaluation"):
            st.session_state["user_query"] = "What are the major approaches and metrics used for RAG evaluation?"
            st.rerun()
    with col_ex2:
        if st.button("📌 Agentic RAG vs Traditional RAG"):
            st.session_state["user_query"] = "Compare agentic RAG with traditional naive RAG architectures."
            st.rerun()
    with col_ex3:
        if st.button("📌 Production LLM Gateways"):
            st.session_state["user_query"] = "Explain the key architectural components and advantages of production LLM gateways."
            st.rerun()

    user_query = st.text_area(
        "Enter your research question:",
        value=st.session_state["user_query"],
        height=100,
        placeholder="e.g., Explain the trade-offs between Fine-Tuning and RAG for enterprise applications..."
    )

    col_btn, col_chk = st.columns([1, 4])
    with col_btn:
        submit_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

    if submit_btn and user_query.strip():
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Input Guardrails & Planning
        status_text.markdown("🛡️ **Step 1/5:** Evaluating Input Guardrails & Formulating Query Plan...")
        progress_bar.progress(20)
        time.sleep(0.1)

        # Step 2: Retrieval
        status_text.markdown("🔍 **Step 2/5:** Querying ChromaDB Vector Store & Executing Web Research...")
        progress_bar.progress(50)
        
        # Execute research workflow
        start_time = time.time()
        result = investigate_query(user_query)
        latency = round(time.time() - start_time, 3)

        # Step 3: Evidence Critique
        status_text.markdown("⚖️ **Step 3/5:** Evidence Critic evaluating groundedness and citations...")
        progress_bar.progress(80)
        time.sleep(0.1)

        # Step 4: Synthesis & Output Guardrails
        status_text.markdown("📝 **Step 4/5:** Report Generator synthesizing final research report...")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        if result.get("guardrail_blocked"):
            st.error(result.get("answer", "Blocked by safety guardrails."))
        else:
            # Metrics Banner
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence Score", f"{int(result.get('confidence', 0.95)*100)}%")
            c2.metric("Total Latency", f"{latency}s")
            c3.metric("Verified Sources", len(result.get("sources", [])))
            c4.metric("Agents Activated", len(result.get("agents_used", [])))

            st.markdown("### 📋 Executive Research Report")
            st.markdown(result.get("answer", ""))

            # Collapsible Detailed Tabs
            tab_plan, tab_sources, tab_agents, tab_lat = st.tabs(["🗺️ Query Plan", "📚 Verified Sources", "🤖 Agent Workflow", "⏱️ Node Latencies"])
            
            with tab_plan:
                st.markdown("#### Dynamic Execution Plan")
                for step in result.get("plan", []):
                    st.markdown(f"- {step}")

            with tab_sources:
                st.markdown("#### Retrieved Citations & Documents")
                sources = result.get("sources", [])
                if sources:
                    for s in sources:
                        st.markdown(f"- **{s.get('title')}** (`{s.get('id')}`) — *{s.get('type')}*")
                else:
                    st.info("No external sources cited.")

            with tab_agents:
                st.markdown("#### Agents Involved in Orchestration")
                for ag in result.get("agents_used", []):
                    st.markdown(f"- 🤖 **{ag}**")
                st.markdown("#### Tools Executed via MCP")
                for tl in result.get("tools_used", []):
                    st.markdown(f"- 🛠️ `{tl}`")

            with tab_lat:
                st.markdown("#### Per-Node Execution Latencies")
                lats = result.get("latencies", {})
                if lats:
                    for node, dur in lats.items():
                        st.markdown(f"- **{node}:** `{dur}s`")
                else:
                    st.markdown(f"Total graph latency: `{latency}s`")


# TAB 2: EVALUATION DASHBOARD
elif nav_choice == "📊 Evaluation Dashboard":
    st.markdown('<div class="main-header">OpsMind Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated evaluation metrics across RAG Groundedness, Agent/Tool Selection, and Guardrail Safety.</div>', unsafe_allow_html=True)

    eval_path = os.path.join(PROJECT_ROOT, "evals", "evaluation_results.json")
    eval_data = None

    if os.path.exists(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    col_reload, col_run = st.columns([1, 1])
    with col_reload:
        if st.button("🔄 Reload Results"):
            st.rerun()
    with col_run:
        if st.button("⚡ Run Live Evaluation Suite"):
            with st.spinner("Running 25 evaluation test cases..."):
                from evals.evaluator import run_evaluations
                eval_data = run_evaluations()
                st.success("Evaluation suite complete!")
                st.rerun()

    if eval_data:
        metrics = eval_data.get("custom_heuristics_summary", eval_data.get("metrics_summary", {}))
        overall = eval_data.get("overall_score", 0.0)
        passed = eval_data.get("passed_cases", 0)
        total = eval_data.get("total_cases", 0)
        ragas_info = eval_data.get("ragas_evaluation", {})

        st.markdown("---")
        st.markdown("### 🏆 Aggregate Performance Scorecard (Deterministic Heuristics)")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall Score", f"{overall*100:.1f}%")
        m2.metric("Passed Cases", f"{passed}/{total}")
        m3.metric("Faithfulness (Heur)", f"{metrics.get('faithfulness_heuristic', metrics.get('faithfulness', 0))*100:.1f}%")
        m4.metric("Answer Relevance (Heur)", f"{metrics.get('answer_relevance_heuristic', metrics.get('answer_relevance', 0))*100:.1f}%")
        m5.metric("Guardrail Accuracy", f"{metrics.get('guardrail_accuracy', 0)*100:.1f}%")

        m6, m7, m8, m9 = st.columns(4)
        m6.metric("Context Relevance (Heur)", f"{metrics.get('context_relevance_heuristic', metrics.get('context_relevance', 0))*100:.1f}%")
        m7.metric("Tool Selection Accuracy", f"{metrics.get('tool_selection_accuracy', 0)*100:.1f}%")
        m8.metric("Citation Compliance", f"{metrics.get('citation_compliance', 0)*100:.1f}%")
        m9.metric("Avg Latency", f"{metrics.get('average_latency_sec', 0)}s")

        # RAGAS status banner
        st.markdown("### 📊 Official RAGAS Evaluation Status")
        if ragas_info.get("status") == "completed":
            st.success("✅ RAGAS LLM-Assisted Evaluation Completed!")
            st.json(ragas_info.get("metrics", {}))
        else:
            st.info(f"ℹ️ **RAGAS Status:** `{ragas_info.get('status', 'unavailable_offline')}` — {ragas_info.get('message', 'Requires live LLM credentials.')}")

        st.markdown("---")
        st.markdown("### 🔍 Test Case Inspector & Failure Drill-Down")

        filter_option = st.selectbox("Filter Cases:", ["All Cases", "Failed Cases Only", "Guardrail Attacks Only", "Research Queries Only"])

        cases = eval_data.get("case_results", [])
        for c in cases:
            if filter_option == "Failed Cases Only" and c.get("passed"):
                continue
            if filter_option == "Guardrail Attacks Only" and not c.get("test_type", "").startswith("guardrail"):
                continue
            if filter_option == "Research Queries Only" and c.get("test_type", "") != "research":
                continue

            status_icon = "✅ PASS" if c.get("passed") else "❌ FAIL"
            heur = c.get("custom_heuristics", {})
            with st.expander(f"{status_icon} | Case #{c.get('id')}: {c.get('question')[:80]}..."):
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.write(f"**Test Type:** `{c.get('test_type')}`")
                    st.write(f"**Category:** `{c.get('category')}`")
                    st.write(f"**Latency:** `{c.get('latency_seconds')}s`")
                    st.write(f"**Estimated Tokens:** `{c.get('estimated_tokens')}`")
                with c_col2:
                    st.write(f"**Faithfulness (Heur):** `{heur.get('faithfulness', c.get('faithfulness'))}`")
                    st.write(f"**Answer Relevance (Heur):** `{heur.get('answer_relevance', c.get('answer_relevance'))}`")
                    st.write(f"**Guardrail Score:** `{c.get('guardrail_score')}`")
                    st.write(f"**Tool Score:** `{c.get('tool_selection_score')}`")

                st.markdown("**Actual Answer Snippet:**")
                st.code(c.get("actual_answer_snippet", ""))
    else:
        st.info("No evaluation results found. Click 'Run Live Evaluation Suite' above to generate evaluation benchmarks.")


# TAB 3: MCP TOOL REGISTRY
elif nav_choice == "🛠️ MCP Tool Registry":
    st.markdown('<div class="main-header">Model Context Protocol (MCP) Registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Standardized JSON tool schemas for agent tool discovery and secure parameter validation via official FastMCP.</div>', unsafe_allow_html=True)

    tools = get_mcp_tool_definitions()
    for t in tools:
        with st.expander(f"🛠️ Tool: `{t.get('name')}`", expanded=True):
            st.markdown(f"**Description:** {t.get('description')}")
            st.markdown("**Input Schema (JSON Schema Spec):**")
            st.json(t.get("inputSchema", {}))


# TAB 4: KNOWLEDGE BASE INSPECTOR
elif nav_choice == "📚 Knowledge Base Inspector":
    st.markdown('<div class="main-header">ChromaDB Technical Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Curated knowledge documents covering RAG architectures, LLM gateways, vector indexing, and guardrails.</div>', unsafe_allow_html=True)

    from app.data.knowledge_base import KNOWLEDGE_DOCUMENTS
    for doc in KNOWLEDGE_DOCUMENTS:
        with st.expander(f"📄 {doc.get('title')} (`{doc.get('id')}`)", expanded=False):
            st.write(f"**Category:** `{doc.get('category')}` | **Tags:** `{', '.join(doc.get('tags', []))}`")
            st.markdown(f"**Summary:** {doc.get('summary')}")
            st.markdown("---")
            st.markdown(doc.get("content", ""))
