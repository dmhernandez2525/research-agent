"""LangGraph StateGraph definition for the research pipeline.

Defines a 5-node graph: plan -> search -> scrape -> summarize -> synthesize
with conditional edges and SqliteSaver checkpointing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from research_agent.nodes.planner import plan_node
from research_agent.nodes.scraper import scrape_node
from research_agent.nodes.searcher import search_node
from research_agent.nodes.summarizer import summarize_node
from research_agent.nodes.synthesizer import synthesize_node
from research_agent.state import ResearchState

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.config import Settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


_MAX_SEARCH_RETRIES = 3


def _should_continue_search(state: ResearchState) -> str:
    """Decide whether to proceed to scraping or retry search.

    Returns ``"scrape"`` if we have enough search results, otherwise
    ``"search"`` for a retry (up to ``_MAX_SEARCH_RETRIES``).

    Args:
        state: Current research state.

    Returns:
        Next node name.
    """
    min_results = 3
    results = state.get("search_results", [])
    retries = state.get("search_retry_count", 0)
    if len(results) >= min_results or retries >= _MAX_SEARCH_RETRIES:
        return "scrape"
    return "search"


def _should_continue_scrape(state: ResearchState) -> str:
    """Decide whether to proceed to summarization or end early.

    Returns ``"summarize"`` if we have scraped content, otherwise routes
    to ``END`` with a warning.

    Args:
        state: Current research state.

    Returns:
        Next node name or END.
    """
    scraped = state.get("scraped_content", [])
    if scraped:
        return "summarize"

    logger.warning("no_scraped_content", msg="No content was scraped; ending early.")
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(settings: Settings) -> StateGraph[Any]:
    """Construct the research StateGraph (uncompiled).

    Args:
        settings: Application settings.

    Returns:
        An uncompiled StateGraph ready for ``.compile()``.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("synthesize", synthesize_node)

    # Set entry point
    graph.set_entry_point("plan")

    # Edges
    graph.add_edge("plan", "search")
    graph.add_conditional_edges(
        "search",
        _should_continue_search,
        {"scrape": "scrape", "search": "search"},
    )
    graph.add_conditional_edges(
        "scrape",
        _should_continue_scrape,
        {"summarize": "summarize", END: END},
    )
    graph.add_edge("summarize", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def compile_graph(
    settings: Settings,
    checkpoint_db: Path | None = None,
) -> Any:
    """Build and compile the research graph with optional checkpointing.

    Args:
        settings: Application settings.
        checkpoint_db: Path to SQLite database for LangGraph checkpointing.
            If ``None`` and checkpoints are enabled, uses the configured
            checkpoint directory.

    Returns:
        A compiled, runnable LangGraph graph.
    """
    graph = build_graph(settings)

    checkpointer: AsyncSqliteSaver | None = None
    if settings.checkpoints.enabled:
        db_path = checkpoint_db or (settings.checkpoints.directory / "langgraph.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = AsyncSqliteSaver.from_conn_string(str(db_path))
        logger.info("checkpointer_enabled", db_path=str(db_path))

    return graph.compile(checkpointer=checkpointer)
