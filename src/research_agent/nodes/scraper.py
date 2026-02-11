"""Trafilatura content extraction node.

Fetches URLs from search results, extracts clean text content, and scores
the quality of each extracted page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from research_agent.state import ResearchState, ScrapedContent, SearchResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fetch_and_extract(
    result: SearchResult,
    timeout: int = 30,
    max_content_length: int = 500_000,
) -> ScrapedContent | None:
    """Fetch a URL and extract content using Trafilatura.

    Args:
        result: The search result to scrape.
        timeout: HTTP request timeout in seconds.
        max_content_length: Maximum characters to retain.

    Returns:
        A ``ScrapedContent`` model, or ``None`` if extraction failed.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_fetch_and_extract is not yet implemented")


async def _scrape_batch(
    results: list[SearchResult],
    max_concurrent: int = 5,
    timeout: int = 30,
    max_content_length: int = 500_000,
) -> list[ScrapedContent]:
    """Scrape a batch of search results concurrently.

    Uses ``asyncio.Semaphore`` to limit concurrency.

    Args:
        results: Search results to scrape.
        max_concurrent: Maximum concurrent HTTP requests.
        timeout: Per-request timeout in seconds.
        max_content_length: Maximum characters per page.

    Returns:
        List of successfully scraped content (failures are logged and skipped).

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_scrape_batch is not yet implemented")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def scrape_node(state: ResearchState) -> dict[str, Any]:
    """Fetch and extract content from search result URLs.

    Uses Trafilatura for content extraction and scores each page for
    quality using the scraping quality module.

    Args:
        state: Current research state with ``search_results`` populated.

    Returns:
        Partial state update with ``scraped_content``, ``step``, and
        ``step_index``.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    search_results = state.get("search_results", [])
    logger.info("scrape_start", num_urls=len(search_results))

    raise NotImplementedError("scrape_node is not yet implemented")
