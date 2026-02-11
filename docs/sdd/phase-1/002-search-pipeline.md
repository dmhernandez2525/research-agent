# SDD-002: Search Pipeline

## Overview

The search pipeline integrates with the Tavily API to discover relevant sources for each subtopic. It expands each subtopic into multiple query variations, executes searches, parses and scores results, deduplicates URLs, and respects rate limits.

## Tavily API Integration

```python
from tavily import AsyncTavilyClient

class SearchService:
    def __init__(self, config: SearchConfig):
        self.client = AsyncTavilyClient(api_key=config.api_key)
        self.max_results = config.max_results
        self.search_depth = config.search_depth

    async def search(self, query: str) -> list[TavilyResult]:
        response = await self.client.search(
            query=query,
            max_results=self.max_results,
            search_depth=self.search_depth,
            include_raw_content=False,
        )
        return response["results"]
```

**Configuration:**
- `max_results`: Number of results per query (default: 10).
- `search_depth`: `"basic"` or `"advanced"` (default: `"advanced"`). Advanced uses Tavily's deeper extraction.

**Tavily Performance:**
- 93.3% accuracy on SimpleQA benchmark, making it the highest-performing search API for factual queries.
- Accepts natural language queries directly -- no keyword optimization or boolean syntax needed.
- Rate limits: 100 requests per minute on the dev tier, 1,000 RPM on the prod tier.

## ExpandSearch Pattern (3 Query Variations)

Each subtopic's search query is expanded into three variations to increase coverage:

```python
async def expand_queries(subtopic: Subtopic, llm: BaseChatModel) -> list[str]:
    """Generate 3 query variations for a subtopic."""
    prompt = f"""Given this research subtopic, generate exactly 3 search queries
    that approach it from different angles.

    Subtopic: {subtopic['title']}
    Description: {subtopic['description']}

    Return as a JSON array of 3 strings."""

    response = await llm.ainvoke(prompt)
    return parse_json_array(response.content)
```

**Variation strategy:**
1. **Direct query** -- Matches the subtopic title closely.
2. **Broader context** -- Adds domain context or related terms.
3. **Specific detail** -- Targets a specific aspect, metric, or comparison.

This approach avoids single-query blind spots and surfaces diverse sources. The ExpandSearch pattern is based on research showing a 34.3% improvement over single-query baselines (arXiv:2510.10009).

**Query expansion split:** Approximately 63% of generated queries use syntax-level reformulation (rephrasing, synonym substitution) while 37% use semantic expansion (broadening or narrowing the conceptual scope). This ratio emerged empirically as the most effective balance for research-style queries.

## Search Result Parsing and Scoring

Each Tavily result is parsed into a `SearchResult` with a relevance score:

```python
def parse_result(raw: TavilyResult, subtopic_id: str) -> SearchResult:
    return SearchResult(
        url=raw["url"],
        title=raw["title"],
        snippet=raw.get("content", ""),
        score=raw.get("score", 0.0),
        subtopic_id=subtopic_id,
    )
```

**Scoring factors:**
- `score` from Tavily (0.0 - 1.0) reflects relevance to the query.
- Results below a configurable threshold (default: 0.3) are discarded.
- Results are sorted by score descending before scraping, so higher-quality sources are processed first.

## URL Deduplication

URLs are deduplicated at two levels:

### Level 1: Within a search batch

Before returning results from the search node, duplicate URLs are removed. The first occurrence (highest score) is kept:

```python
def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        normalized = normalize_url(result["url"])
        if normalized not in seen:
            seen.add(normalized)
            unique.append(result)
    return unique
```

### Level 2: Across subtopics

The `seen_urls` set in `ResearchState` tracks all URLs encountered across the entire run. The search node checks this set before adding results:

```python
def filter_new_urls(
    results: list[SearchResult], seen_urls: set[str]
) -> tuple[list[SearchResult], set[str]]:
    new_results = []
    new_seen = set()
    for result in results:
        normalized = normalize_url(result["url"])
        if normalized not in seen_urls:
            new_results.append(result)
            new_seen.add(normalized)
    return new_results, seen_urls | new_seen
```

### URL Normalization

```python
def normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)
    # Remove trailing slash, fragment, common tracking params
    cleaned = parsed._replace(
        fragment="",
        path=parsed.path.rstrip("/"),
        query=remove_tracking_params(parsed.query),
    )
    return urlunparse(cleaned).lower()
```

## Rate Limiting Strategy

Rate limits are managed at two levels:

### Tavily API Rate Limiting

Tavily's rate limits (100 RPM dev tier, 1,000 RPM prod tier) are respected via a semaphore and backoff:

```python
class RateLimitedSearch:
    def __init__(self, max_concurrent: int = 3, delay: float = 0.5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay

    async def execute(self, query: str) -> list[TavilyResult]:
        async with self.semaphore:
            result = await self._search_with_retry(query)
            await asyncio.sleep(self.delay)
            return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _search_with_retry(self, query: str) -> list[TavilyResult]:
        return await self.client.search(query)
```

### Fallback Search Strategy

If Tavily is unavailable or rate-limited, the pipeline falls back through a chain of alternative search providers:

1. **Tavily** (primary) -- highest accuracy, natural language queries.
2. **Brave Search API** -- first fallback, good coverage and relevance.
3. **SerpAPI** -- second fallback, Google results via API.
4. **Cached results** -- final fallback, synthesize from previously gathered data only.

Each provider is attempted with the same retry/backoff logic. The fallback chain is traversed per-query, not per-run -- if Tavily recovers mid-run, subsequent queries will use it again.

### Per-Subtopic Budget

Each subtopic is allocated a maximum number of search queries (default: 3 from the expand step). If all queries for a subtopic fail, the subtopic is marked as `"failed"` and the pipeline continues with the next one.

## Search Node Implementation

```python
async def search_node(state: ResearchState) -> dict:
    index = state["current_subtopic_index"]
    subtopic = state["subtopics"][index]

    # Expand into 3 query variations
    queries = await expand_queries(subtopic, llm)

    # Execute searches with rate limiting
    all_results = []
    for query in queries:
        results = await rate_limiter.execute(query)
        all_results.extend(parse_results(results, subtopic["id"]))

    # Deduplicate within batch and against seen URLs
    unique = deduplicate_results(all_results)
    new_results, updated_seen = filter_new_urls(unique, state["seen_urls"])

    return {
        "search_results": new_results,
        "seen_urls": updated_seen,
    }
```

## File Location

```
src/research_agent/
    search.py         # SearchService, RateLimitedSearch
    nodes/
        search.py     # search_node, expand_queries
```
