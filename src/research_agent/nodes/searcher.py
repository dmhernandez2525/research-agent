"""Tavily search node.

Generates 3 query variations per sub-question (ExpandSearch pattern) and
executes searches, accumulating results into the graph state.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.state import SearchResult

if TYPE_CHECKING:
    from research_agent.state import ResearchState

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONCURRENT_SEARCHES = 3
_MIN_RELEVANCE_SCORE = 0.3


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
# Tavily client helpers
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((OSError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
async def _tavily_search_with_retry(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """Execute a single Tavily search with retry and exponential backoff.

    Args:
        query: The search query string.
        max_results: Maximum number of results.
        search_depth: Tavily search depth ("basic" or "advanced").

    Returns:
        List of raw result dicts from Tavily.
    """
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient()
    response = await client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
    )
    return response.get("results", [])


def _parse_results(
    raw_results: list[dict[str, Any]],
    sub_question_id: int,
    query: str,
) -> list[SearchResult]:
    """Convert raw Tavily results to SearchResult models.

    Filters out results below the minimum relevance score and sorts
    by score descending.

    Args:
        raw_results: Raw dicts from Tavily API.
        sub_question_id: The sub-question these results belong to.
        query: The query that produced these results.

    Returns:
        Sorted, filtered list of SearchResult models.
    """
    results: list[SearchResult] = []
    for item in raw_results:
        score = float(item.get("score", 0.0))
        if score < _MIN_RELEVANCE_SCORE:
            continue
        results.append(
            SearchResult(
                sub_question_id=sub_question_id,
                query=query,
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=score,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results


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
# Core search function
# ---------------------------------------------------------------------------


async def execute_search(
    query: str,
    sub_question_id: int,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[SearchResult]:
    """Execute a Tavily search and return parsed, filtered results.

    Args:
        query: The search query string.
        sub_question_id: ID of the originating sub-question.
        max_results: Maximum number of results to return.
        search_depth: Tavily search depth ("basic" or "advanced").

    Returns:
        List of SearchResult models, filtered and sorted by score.
    """
    raw = await _tavily_search_with_retry(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
    )
    return _parse_results(raw, sub_question_id, query)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Search the web for the current subtopic's query.

    For the current subtopic (indexed by current_subtopic_index), executes
    the sub-question as a search query with rate limiting and deduplication.

    Args:
        state: Current research state with ``sub_questions`` populated.

    Returns:
        Partial state update with ``search_results``, ``seen_urls``,
        ``step``, ``step_index``, and ``search_retry_count``.
    """
    sub_questions = state.get("sub_questions", [])
    current_idx = state.get("current_subtopic_index", 0)
    seen_urls = state.get("seen_urls", [])

    if current_idx >= len(sub_questions):
        logger.info("search_skip", reason="no more sub-questions")
        return {"search_results": [], "step": "search", "step_index": 1}

    sub_q = sub_questions[current_idx]
    sub_q_id = sub_q.get("id", current_idx + 1) if isinstance(sub_q, dict) else sub_q.id
    question = sub_q.get("question", "") if isinstance(sub_q, dict) else sub_q.question

    logger.info(
        "search_start",
        sub_question_id=sub_q_id,
        question=question,
    )

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
    all_results: list[SearchResult] = []

    async def _search_one(q: str) -> list[SearchResult]:
        async with semaphore:
            try:
                return await execute_search(
                    query=q,
                    sub_question_id=sub_q_id,
                )
            except Exception as exc:
                logger.warning("search_query_failed", query=q, error=str(exc))
                return []

    # For now, search with the original question directly.
    # ExpandSearch (F2.2) will add query expansion later.
    results = await _search_one(question)
    all_results.extend(results)

    # Deduplicate within this batch
    unique = _deduplicate_results(all_results)

    # Filter out already-seen URLs (cross-subtopic dedup)
    seen_set = set(seen_urls)
    new_results = [r for r in unique if r.url not in seen_set]
    new_urls = [r.url for r in new_results]

    logger.info(
        "search_complete",
        sub_question_id=sub_q_id,
        total_raw=len(all_results),
        unique=len(unique),
        new=len(new_results),
    )

    return {
        "search_results": new_results,
        "seen_urls": new_urls,
        "step": "search",
        "step_index": 1,
        "search_retry_count": 0,
    }
