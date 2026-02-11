"""Tavily search node.

Generates 3 query variations per sub-question (ExpandSearch pattern) and
executes searches, accumulating results into the graph state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from research_agent.state import ResearchState, SearchResult, SubQuestion

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured output for query expansion
# ---------------------------------------------------------------------------


class ExpandedQueries(BaseModel):
    """Three query variations for a single sub-question."""

    original: str = Field(description="The original sub-question.")
    variations: list[str] = Field(
        description="Three diverse search query reformulations.",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_queries(sub_question: SubQuestion) -> list[str]:
    """Generate 3 diverse search query variations for a sub-question.

    Uses LLM structured output to produce varied phrasings that improve
    search recall.

    Args:
        sub_question: The sub-question to expand.

    Returns:
        List of 3 query strings.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_expand_queries is not yet implemented")


def _execute_search(query: str, max_results: int = 10) -> list[SearchResult]:
    """Execute a single Tavily search and return parsed results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of ``SearchResult`` models.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_execute_search is not yet implemented")


def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate search results by URL.

    Args:
        results: Raw list of search results (may contain duplicates).

    Returns:
        Deduplicated list preserving first occurrence order.
    """
    seen_urls: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            unique.append(result)
    return unique


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def search_node(state: ResearchState) -> dict[str, Any]:
    """Search the web for each sub-question using the ExpandSearch pattern.

    For each sub-question, generates 3 query variations and runs searches,
    deduplicating the combined results.

    Args:
        state: Current research state with ``sub_questions`` populated.

    Returns:
        Partial state update with ``search_results``, ``step``, and
        ``step_index``.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    sub_questions = state.get("sub_questions", [])
    logger.info("search_start", num_sub_questions=len(sub_questions))

    raise NotImplementedError("search_node is not yet implemented")
