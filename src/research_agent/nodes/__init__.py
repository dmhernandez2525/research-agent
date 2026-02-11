"""Research pipeline graph nodes.

Each node is a callable ``(ResearchState) -> dict`` that returns partial
state updates to be merged by LangGraph.
"""

from __future__ import annotations

from research_agent.nodes.planner import plan_node
from research_agent.nodes.scraper import scrape_node
from research_agent.nodes.searcher import search_node
from research_agent.nodes.summarizer import summarize_node
from research_agent.nodes.synthesizer import synthesize_node

__all__ = [
    "plan_node",
    "scrape_node",
    "search_node",
    "summarize_node",
    "synthesize_node",
]
