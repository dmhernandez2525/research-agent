"""Trafilatura content extraction node.

Fetches URLs from search results, extracts clean text content via
Trafilatura, scores quality, sanitizes against prompt injection, and
accumulates scraped content into the graph state.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from research_agent.state import ScrapedContent

if TYPE_CHECKING:
    from research_agent.state import ResearchState, SearchResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_CONCURRENT = 5
_DEFAULT_MAX_CONTENT_LENGTH = 500_000

_MIN_QUALITY_SCORE = 0.4
_FLAG_QUALITY_THRESHOLD = 0.7

# Patterns that indicate prompt injection attempts in extracted content
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on\w+\s*=\s*[\"']", re.IGNORECASE),
    re.compile(r"ignore\s+(previous|above|all)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*iframe\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# HTML sanitization
# ---------------------------------------------------------------------------


def _sanitize_content(text: str) -> str:
    """Remove potential prompt injection patterns from extracted content.

    Strips script tags, event handlers, and common injection phrases
    that could manipulate downstream LLM processing.

    Args:
        text: Raw extracted text content.

    Returns:
        Sanitized text with injection patterns removed.
    """
    sanitized = text
    for pattern in _INJECTION_PATTERNS:
        sanitized = pattern.sub("[REMOVED]", sanitized)
    return sanitized


# ---------------------------------------------------------------------------
# Content quality scoring
# ---------------------------------------------------------------------------


def _score_content_quality(text: str) -> float:
    """Score the quality of extracted content on a 0.0-1.0 scale.

    Scoring factors:
    - Word count (longer, substantive content scores higher)
    - Paragraph count (well-structured content has multiple paragraphs)
    - Average sentence length (neither too short nor too long)

    Args:
        text: Extracted text content to score.

    Returns:
        Quality score between 0.0 and 1.0.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0

    words = stripped.split()
    word_count = len(words)

    # Very short content is low quality
    if word_count < 20:
        return 0.1

    # Word count score: ramp up to 1.0 at 500+ words
    word_score = min(word_count / 500.0, 1.0)

    # Paragraph score: multiple paragraphs indicate structure
    paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    para_count = len(paragraphs)
    para_score = min(para_count / 5.0, 1.0)

    # Sentence score: check for proper sentence structure
    sentences = re.split(r"[.!?]+", stripped)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    sentence_score = min(sentence_count / 10.0, 1.0) if sentence_count > 0 else 0.0

    # Weighted combination
    score = 0.5 * word_score + 0.25 * para_score + 0.25 * sentence_score
    return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Fetch and extract
# ---------------------------------------------------------------------------


async def _fetch_and_extract(
    result: SearchResult,
    timeout: int = _DEFAULT_TIMEOUT,
    max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
) -> ScrapedContent | None:
    """Fetch a URL and extract content using Trafilatura.

    Fetches HTML via httpx, extracts main content with Trafilatura,
    sanitizes against prompt injection, scores quality, and truncates
    to max_content_length.

    Args:
        result: The search result to scrape.
        timeout: HTTP request timeout in seconds.
        max_content_length: Maximum characters to retain.

    Returns:
        A ``ScrapedContent`` model, or ``None`` if extraction failed.
    """
    import trafilatura

    url = result.url
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except Exception as exc:
        logger.warning("fetch_failed", url=url, error=str(exc))
        return None

    # Extract main content with Trafilatura
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
    except Exception as exc:
        logger.warning("extract_failed", url=url, error=str(exc))
        return None

    if not extracted:
        logger.debug("extract_empty", url=url)
        return None

    # Sanitize against prompt injection
    content = _sanitize_content(extracted)

    # Truncate to max length
    if len(content) > max_content_length:
        content = content[:max_content_length]

    # Score quality
    quality = _score_content_quality(content)

    word_count = len(content.split())

    logger.info(
        "scrape_ok",
        url=url,
        word_count=word_count,
        quality_score=quality,
    )

    return ScrapedContent(
        url=url,
        sub_question_id=result.sub_question_id,
        title=result.title,
        content=content,
        word_count=word_count,
        quality_score=quality,
    )


async def _scrape_batch(
    results: list[SearchResult],
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    timeout: int = _DEFAULT_TIMEOUT,
    max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
) -> list[ScrapedContent]:
    """Scrape a batch of search results concurrently.

    Uses ``asyncio.Semaphore`` to limit concurrency. Failed scrapes are
    logged and skipped.

    Args:
        results: Search results to scrape.
        max_concurrent: Maximum concurrent HTTP requests.
        timeout: Per-request timeout in seconds.
        max_content_length: Maximum characters per page.

    Returns:
        List of successfully scraped content (failures are logged and skipped).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_fetch(r: SearchResult) -> ScrapedContent | None:
        async with semaphore:
            return await _fetch_and_extract(
                r,
                timeout=timeout,
                max_content_length=max_content_length,
            )

    tasks = [_limited_fetch(r) for r in results]
    raw_results = await asyncio.gather(*tasks)
    return [r for r in raw_results if r is not None]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def scrape_node(state: ResearchState) -> dict[str, Any]:
    """Fetch and extract content from search result URLs.

    Uses Trafilatura for content extraction, scores each page for quality,
    and filters out pages below the minimum quality threshold.

    Args:
        state: Current research state with ``search_results`` populated.

    Returns:
        Partial state update with ``scraped_content``, ``step``, and
        ``step_index``.
    """
    search_results = state.get("search_results", [])
    logger.info("scrape_start", num_urls=len(search_results))

    if not search_results:
        return {"scraped_content": [], "step": "scrape", "step_index": 2}

    scraped = await _scrape_batch(search_results)

    # Filter by quality threshold
    accepted: list[ScrapedContent] = []
    for item in scraped:
        if item.quality_score < _MIN_QUALITY_SCORE:
            logger.info(
                "scrape_rejected",
                url=item.url,
                quality_score=item.quality_score,
                reason="below_threshold",
            )
            continue
        if item.quality_score < _FLAG_QUALITY_THRESHOLD:
            logger.info(
                "scrape_flagged",
                url=item.url,
                quality_score=item.quality_score,
                reason="low_quality",
            )
        accepted.append(item)

    logger.info(
        "scrape_complete",
        total_urls=len(search_results),
        scraped=len(scraped),
        accepted=len(accepted),
        rejected=len(scraped) - len(accepted),
    )

    return {
        "scraped_content": accepted,
        "step": "scrape",
        "step_index": 2,
    }
