"""Tavily search node with ExpandSearch pattern.

Generates 3 query variations per sub-question using LLM-powered expansion,
then executes searches for all variations concurrently, accumulating
deduplicated results into the graph state.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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

# Tracking parameters to strip during URL normalization
_TRACKING_PARAMS: re.Pattern[str] = re.compile(
    r"^(utm_\w+|fbclid|gclid|gclsrc|dclid|msclkid|mc_[ce]id|"
    r"ref|affiliate|campaign_id|ad_id|zanpid|_ga|_gid|_gl|"
    r"yclid|_openstat|wbraid|gbraid)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication comparison.

    Applies the following transformations:
    - Lowercase the scheme and hostname
    - Strip trailing slash from the path
    - Remove URL fragment (#section)
    - Remove tracking query parameters (utm_*, fbclid, gclid, etc.)
    - Sort remaining query parameters for consistent comparison

    Args:
        url: The raw URL to normalize.

    Returns:
        Normalized URL string.
    """
    parsed = urlparse(url)

    # Lowercase scheme and hostname
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip trailing slash from path (but keep "/" for root)
    path = parsed.path.rstrip("/") or "/"

    # Remove fragment
    fragment = ""

    # Filter out tracking parameters, sort remaining
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if not _TRACKING_PARAMS.match(k)}
        # Sort keys and use first value for each param for stable output
        query = urlencode(
            sorted(filtered.items()),
            doseq=True,
        )
    else:
        query = ""

    return urlunparse((scheme, netloc, path, parsed.params, query, fragment))


_EXPAND_SYSTEM_PROMPT = """\
You are a search query expansion specialist. Given a research sub-question,
generate exactly 3 diverse search query reformulations optimized for web search.

Strategy for the 3 variations:
1. **Direct query**: A focused, keyword-rich reformulation of the question.
2. **Broader context**: A query that captures the wider topic or background.
3. **Specific detail**: A query targeting specific facts, data, or examples.

Guidelines:
- Keep queries concise (under 15 words each).
- Use different vocabulary across variations to maximize result diversity.
- Roughly 63% syntax reformulations, 37% semantic expansions.
- Do NOT include the original question verbatim as a variation.
"""


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
# Query expansion (ExpandSearch pattern)
# ---------------------------------------------------------------------------


_EXPAND_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in this format: "
    '{"original": "<the question>", "variations": ["query1", "query2", "query3"]}'
)


async def _expand_queries(question: str) -> ExpandedQueries:
    """Use an LLM to expand a sub-question into 3 search query variations.

    Uses the FAST tier model (Haiku) for cost efficiency. Falls back to
    the original question wrapped in an ExpandedQueries if the LLM call fails.

    Args:
        question: The original sub-question to expand.

    Returns:
        An ExpandedQueries instance with 3 search query variations.

    Raises:
        Exception: Re-raises any LLM error so the caller can handle fallback.
    """
    import litellm

    from research_agent.models import _extract_json

    response = await litellm.acompletion(
        model="anthropic/claude-haiku-3-5-20241022",
        messages=[
            {"role": "system", "content": _EXPAND_SYSTEM_PROMPT + _EXPAND_JSON_INSTRUCTION},
            {"role": "user", "content": question},
        ],
        max_tokens=256,
        temperature=0.7,
    )

    content = response.choices[0].message.content
    data = _extract_json(content)
    result = ExpandedQueries(**data)

    logger.info(
        "expand_queries_ok",
        original=question,
        variations=result.variations,
    )
    return result


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
    results: list[dict[str, Any]] = response.get("results", [])
    return results


def _parse_results(
    raw_results: list[dict[str, Any]],
    subtopic_id: int,
    query: str,
) -> list[SearchResult]:
    """Convert raw Tavily results to SearchResult models.

    Filters out results below the minimum relevance score and sorts
    by score descending.

    Args:
        raw_results: Raw dicts from Tavily API.
        subtopic_id: The sub-question these results belong to.
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
                subtopic_id=subtopic_id,
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
    """Remove duplicate search results by normalized URL.

    Uses ``_normalize_url`` so that URLs differing only in tracking params,
    trailing slashes, fragments, or casing are treated as duplicates.

    Args:
        results: Raw list of search results (may contain duplicates).

    Returns:
        Deduplicated list preserving first occurrence order.
    """
    seen_urls: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        norm = _normalize_url(result.url)
        if norm not in seen_urls:
            seen_urls.add(norm)
            unique.append(result)
    return unique


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


async def execute_search(
    query: str,
    subtopic_id: int,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[SearchResult]:
    """Execute a Tavily search and return parsed, filtered results.

    Args:
        query: The search query string.
        subtopic_id: ID of the originating sub-question.
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
    return _parse_results(raw, subtopic_id, query)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Search the web for the current subtopic's query.

    For the current subtopic (indexed by current_subtopic_index), executes
    the sub-question as a search query with rate limiting and deduplication.

    Args:
        state: Current research state with ``subtopics`` populated.

    Returns:
        Partial state update with ``search_results``, ``seen_urls``,
        ``step``, ``step_index``, and ``search_retry_count``.
    """
    subtopics = state.get("subtopics", [])
    current_idx = state.get("current_subtopic_index", 0)
    seen_urls = state.get("seen_urls", [])

    if current_idx >= len(subtopics):
        logger.info("search_skip", reason="no more sub-questions")
        return {"search_results": [], "step": "search", "step_index": 1}

    sub_q = subtopics[current_idx]
    sub_q_id = sub_q.get("id", current_idx + 1) if isinstance(sub_q, dict) else sub_q.id
    question = sub_q.get("question", "") if isinstance(sub_q, dict) else sub_q.question

    logger.info(
        "search_start",
        subtopic_id=sub_q_id,
        question=question,
    )

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
    all_results: list[SearchResult] = []

    async def _search_one(q: str) -> list[SearchResult]:
        async with semaphore:
            try:
                return await execute_search(
                    query=q,
                    subtopic_id=sub_q_id,
                )
            except Exception as exc:
                logger.warning("search_query_failed", query=q, error=str(exc))
                return []

    # ExpandSearch: generate 3 query variations via LLM, fallback to original
    try:
        expanded = await _expand_queries(question)
        queries = expanded.variations
    except Exception as exc:
        logger.warning(
            "expand_queries_failed",
            question=question,
            error=str(exc),
        )
        queries = [question]

    # Search all query variations concurrently
    tasks = [_search_one(q) for q in queries]
    batch_results = await asyncio.gather(*tasks)
    for results in batch_results:
        all_results.extend(results)

    # Deduplicate within this batch (uses normalized URLs)
    unique = _deduplicate_results(all_results)

    # Filter out already-seen URLs (cross-subtopic dedup with normalization)
    seen_set = {_normalize_url(u) for u in seen_urls}
    new_results = [r for r in unique if _normalize_url(r.url) not in seen_set]
    new_urls = [_normalize_url(r.url) for r in new_results]

    # Dedup statistics
    batch_dupes = len(all_results) - len(unique)
    cross_dupes = len(unique) - len(new_results)
    logger.info(
        "search_complete",
        subtopic_id=sub_q_id,
        total_raw=len(all_results),
        batch_deduped=batch_dupes,
        cross_deduped=cross_dupes,
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
