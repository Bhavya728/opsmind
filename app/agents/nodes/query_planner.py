"""
Query Planner Node for OpsMind Research & Knowledge System.
Deconstructs research questions, plans investigation strategy, and determines retrieval routing.
"""

import json
import time
from typing import Dict, Any
from app.agents.state import ResearchState
from app.gateway.client import get_llm
from app.utils.logger import logfire


def query_planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Analyzes the user's research query, breaks it down into sub-queries,
    and decides whether to invoke Web Research, RAG Knowledge Search, or both.
    """
    start_time = time.time()
    query = state.get("query", "")
    logfire.info(f"🧠 [Query Planner] Analyzing research request: '{query[:60]}'")
    
    planner_prompt = f"""You are the Query Planner Agent in OpsMind research assistant.
Deconstruct this user research query into a structured research plan.

Research Query: "{query}"

Output valid JSON with the following schema:
{{
    "need_rag": true,
    "need_web": true,
    "search_queries": [
        "sub query 1",
        "sub query 2"
    ],
    "focus_areas": [
        "key aspect 1",
        "key aspect 2"
    ],
    "plan": [
        "1. Step description",
        "2. Step description"
    ]
}}
"""
    llm = get_llm(task_type="planning")
    existing_latencies = dict(state.get("latencies", {}))
    
    try:
        response = llm.invoke(planner_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Parse JSON
        parsed = None
        if "{" in content and "}" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
            try:
                parsed = json.loads(json_str)
            except Exception:
                pass
                
        if not parsed:
            parsed = {
                "need_rag": True,
                "need_web": True,
                "search_queries": [query],
                "focus_areas": ["Core concepts", "Architectural trade-offs", "Evaluation"],
                "plan": [
                    f"1. Query ChromaDB internal knowledge for '{query}'",
                    f"2. Execute web research for latest benchmarks on '{query}'",
                    "3. Evaluate factual evidence and filter ungrounded claims",
                    "4. Synthesize comprehensive structured research report with citations"
                ]
            }

        elapsed = round(time.time() - start_time, 4)
        existing_latencies["planner"] = elapsed

        return {
            "need_rag": parsed.get("need_rag", True),
            "need_web": parsed.get("need_web", True),
            "search_queries": parsed.get("search_queries", [query]),
            "focus_areas": parsed.get("focus_areas", []),
            "plan": parsed.get("plan", []),
            "current_step": "Planning complete",
            "latencies": existing_latencies,
            "agents_used": ["Query Planner"],
            "messages": [{"role": "assistant", "content": f"Query Planner created a {len(parsed.get('plan', []))}-step research strategy in {elapsed}s."}]
        }
    except Exception as e:
        logfire.error(f"❌ Planner node error: {e}")
        elapsed = round(time.time() - start_time, 4)
        existing_latencies["planner"] = elapsed
        return {
            "need_rag": True,
            "need_web": True,
            "search_queries": [query],
            "focus_areas": ["System analysis"],
            "plan": ["1. Search RAG", "2. Search Web", "3. Critique", "4. Synthesize"],
            "current_step": "Planning fallback",
            "latencies": existing_latencies,
            "agents_used": ["Query Planner"],
            "messages": [{"role": "assistant", "content": f"Planner error: {e}"}]
        }
