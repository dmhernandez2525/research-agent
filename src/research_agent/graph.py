"""LangGraph StateGraph definition for the research pipeline.

Defines a 5-node graph with fan-out/fan-in subtopic iteration:
    plan -> search -> [should_continue_search] -> scrape -> summarize
                                                      -> [all_subtopics_done] -> search (loop) | synthesize -> END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langgraph.graph import END, StateGraph

from research_agent.nodes.planner import plan_node
from research_agent.nodes.scraper import scrape_node
from research_agent.nodes.searcher import search_node
from research_agent.nodes.summarizer import summarize_node
from research_agent.nodes.synthesizer import synthesize_node
from research_agent.recovery import RecoveryOrchestrator
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
    """
    scraped = state.get("scraped_pages", [])
    if scraped:
        return "summarize"

    logger.warning("no_scraped_content", msg="No content was scraped; ending early.")
    return END


def _all_subtopics_done(state: ResearchState) -> str:
    """Decide whether to loop back to search or proceed to synthesis.

    After summarizing the current subtopic, checks if more sub-questions
    remain. If so, loops back to ``"search"`` for the next subtopic.
    Otherwise, routes to ``"synthesize"`` for final report generation.
    """
    subtopics = state.get("subtopics", [])
    current_idx = state.get("current_subtopic_index", 0)

    if current_idx < len(subtopics):
        logger.info(
            "next_subtopic",
            index=current_idx,
            remaining=len(subtopics) - current_idx,
        )
        return "search"

    logger.info("all_subtopics_complete", total=len(subtopics))
    return "synthesize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(settings: Settings) -> StateGraph[Any]:
    """Construct the research StateGraph (uncompiled).

    Graph topology:
        plan -> search -> [should_continue_search] -> scrape | search (retry)
                           scrape -> [should_continue_scrape] -> summarize | END
                                      summarize -> [all_subtopics_done] -> search (loop) | synthesize
                                                    synthesize -> END

    Args:
        settings: Application settings.

    Returns:
        An uncompiled StateGraph ready for ``.compile()``.
    """
    graph = StateGraph(ResearchState)

    orchestrator: RecoveryOrchestrator | None = None
    if settings.recovery.enabled:
        orchestrator = RecoveryOrchestrator.from_settings(settings.recovery)
        logger.info(
            "recovery_orchestrator_enabled",
            circuit_breaker_threshold=settings.recovery.circuit_breaker_threshold,
        )

    def wrap_node(node_name: str, node_fn: Any) -> Any:
        if orchestrator is None:
            return node_fn
        return orchestrator.wrap(node_name, node_fn)

    # Add nodes
    graph.add_node("plan", wrap_node("plan", plan_node))
    graph.add_node("search", wrap_node("search", search_node))
    graph.add_node("scrape", wrap_node("scrape", scrape_node))
    graph.add_node("summarize", wrap_node("summarize", summarize_node))
    graph.add_node("synthesize", wrap_node("synthesize", synthesize_node))

    # Set entry point
    graph.set_entry_point("plan")

    # plan -> search (always)
    graph.add_edge("plan", "search")

    # search -> should_continue_search -> scrape or search (retry)
    graph.add_conditional_edges(
        "search",
        _should_continue_search,
        {"scrape": "scrape", "search": "search"},
    )

    # scrape -> should_continue_scrape -> summarize or END (no content)
    graph.add_conditional_edges(
        "scrape",
        _should_continue_scrape,
        {"summarize": "summarize", END: END},
    )

    # summarize -> all_subtopics_done -> search (loop) or synthesize
    graph.add_conditional_edges(
        "summarize",
        _all_subtopics_done,
        {"search": "search", "synthesize": "synthesize"},
    )

    # synthesize -> END
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
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    graph = build_graph(settings)

    checkpointer: AsyncSqliteSaver | None = None
    if settings.checkpoints.enabled:
        db_path = checkpoint_db or (settings.checkpoints.directory / "langgraph.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = AsyncSqliteSaver.from_conn_string(str(db_path))
        logger.info("checkpointer_enabled", db_path=str(db_path))

    return graph.compile(checkpointer=checkpointer)
