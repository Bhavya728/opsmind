"""
Web Research Agent Node for OpsMind.
Executes external technical web searches via MCP tool and extracts relevant snippets.
"""

import time
from typing import Dict, Any, List
from app.agents.state import ResearchState
from app.tools.mcp_tools import execute_mcp_tool
from app.utils.logger import logfire


def web_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes web searches for each sub-query determined by the Planner,
    aggregating external literature, benchmarks, and release notes.
    """
    start_time = time.time()
    existing_latencies = dict(state.get("latencies", {}))

    if not state.get("need_web", True):
        logfire.info("🌐 [Web Research Agent] Skipped by Planner decision.")
        existing_latencies["web_research"] = 0.0
        return {
            "web_findings": [],
            "latencies": existing_latencies,
            "current_step": "Web research skipped"
        }

    search_queries = state.get("search_queries", [state.get("query", "")])
    logfire.info(f"🌐 [Web Research Agent] Executing web research across {len(search_queries)} queries.")
    
    findings: List[Dict[str, Any]] = []
    tools_used: List[str] = []
    evidence_extracted: List[str] = []

    for sq in search_queries[:2]:
        result = execute_mcp_tool("web_search", {"query": sq, "num_results": 3})
        tools_used.append("web_search")
        
        if result.get("status") == "success" and result.get("results"):
            for res in result["results"]:
                findings.append(res)
                evidence_extracted.append(f"Web Source [{res.get('title')}]: {res.get('snippet')[:180]}")

    elapsed = round(time.time() - start_time, 4)
    existing_latencies["web_research"] = elapsed

    return {
        "web_findings": findings,
        "observed_evidence": evidence_extracted,
        "latencies": existing_latencies,
        "current_step": "Web research completed",
        "agents_used": ["Web Research Agent"],
        "tools_used": ["web_search"] if tools_used else [],
        "messages": [{"role": "assistant", "content": f"Web Research Agent gathered {len(findings)} external web articles in {elapsed}s."}]
    }
