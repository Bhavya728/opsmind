"""
Web Research Agent Node for OpsMind.
Executes external technical web searches via MCP tool and extracts relevant snippets.
"""

import time
from typing import Dict, Any, List
from app.agents.state import ResearchState
from app.config import settings
from app.tools.mcp_tools import execute_mcp_tool
from app.utils.logger import logfire


def web_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes a web search for each pending sub-query (from the Planner, or the Critic's follow-up
    round), de-duplicating results by URL and keeping findings from earlier rounds.
    Only invoked when the Planner routes the query to the web.
    """
    start_time = time.time()
    research_round = state.get("research_round", 1)
    search_queries = state.get("search_queries") or [state.get("query", "")]
    logfire.info(f"🌐 [Web Research Agent] Executing web research across {len(search_queries)} queries.")

    findings: List[Dict[str, Any]] = list(state.get("web_findings") or [])
    evidence_extracted: List[str] = []
    warnings: List[str] = []
    seen_urls = {f.get("url") for f in findings}
    previous_count = len(findings)

    for sq in search_queries:
        if len(findings) >= settings.MAX_CONTEXT_SOURCES:
            break
        result = execute_mcp_tool("web_search", {"query": sq, "num_results": settings.WEB_TOP_K,
                                                 "recent": bool(state.get("needs_recent"))})
        if result.get("status") != "success":
            warnings.append(f"Web search failed for '{sq}': {result.get('error', result.get('status'))}")
            continue
        for res in result.get("results", []):
            url = res.get("url")
            if not url or url in seen_urls or len(findings) >= settings.MAX_CONTEXT_SOURCES:
                continue
            seen_urls.add(url)
            findings.append(res)
            evidence_extracted.append(f"Web Source [{res.get('title')}]: {res.get('snippet', '')[:180]}")

    elapsed = round(time.time() - start_time, 4)

    return {
        "web_findings": findings,
        "observed_evidence": evidence_extracted,
        "latencies": {"web_research" if research_round == 1 else f"web_research_r{research_round}": elapsed},
        "current_step": "Web research completed",
        "agents_used": ["Web Research Agent"],
        "tools_used": ["web_search"],
        "warnings": warnings,
        "messages": [{"role": "assistant", "content": f"Web Research Agent gathered {len(findings) - previous_count} new web results (round {research_round}) in {elapsed}s."}]
    }
