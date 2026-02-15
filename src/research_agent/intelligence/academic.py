"""Academic paper search across Semantic Scholar and arXiv."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from research_agent.intelligence.models import AcademicPaper

if TYPE_CHECKING:
    from collections.abc import Callable

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ProviderRateLimiter:
    """Simple per-provider request limiter."""

    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, provider: str, per_minute: int) -> bool:
        now = time.time()
        bucket = self._requests[provider]
        cutoff = now - 60.0

        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= per_minute:
            return False

        bucket.append(now)
        return True


class AcademicSearch:
    """Search peer-reviewed papers and preprints with fallback behavior."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        fallback_search: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=10.0)
        self._limiter = ProviderRateLimiter()
        self._fallback_search = fallback_search

    def close(self) -> None:
        self._client.close()

    def search(self, query: str, max_results: int = 10) -> list[AcademicPaper]:
        """Search Semantic Scholar + arXiv and return ranked results."""
        results: list[AcademicPaper] = []
        provider_errors = 0

        if self._limiter.allow("semantic_scholar", per_minute=90):
            try:
                results.extend(self._semantic_scholar(query, max_results=max_results))
            except Exception as exc:
                provider_errors += 1
                logger.warning(
                    "academic_provider_failed",
                    provider="semantic_scholar",
                    error=str(exc),
                )

        if self._limiter.allow("arxiv", per_minute=30):
            try:
                results.extend(self._arxiv(query, max_results=max_results))
            except Exception as exc:
                provider_errors += 1
                logger.warning(
                    "academic_provider_failed",
                    provider="arxiv",
                    error=str(exc),
                )

        if (not results or provider_errors > 0) and self._fallback_search is not None:
            results.extend(self._fallback(query, max_results=max_results))

        ranked = sorted(results, key=lambda paper: paper.relevance_score, reverse=True)
        return ranked[:max_results]

    def _semantic_scholar(self, query: str, max_results: int) -> list[AcademicPaper]:
        endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
        response = self._client.get(
            endpoint,
            params={
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,authors,citationCount,year",
            },
        )
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", [])
        if not isinstance(data, list):
            return []

        papers: list[AcademicPaper] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            papers.append(
                AcademicPaper(
                    title=str(item.get("title", "")),
                    abstract=str(item.get("abstract", "")),
                    authors=[
                        author.get("name", "")
                        for author in item.get("authors", [])
                        if isinstance(author, dict)
                    ],
                    citation_count=int(item.get("citationCount", 0) or 0),
                    year=int(item.get("year", 0) or 0),
                    source="semantic_scholar",
                    relevance_score=self._score(
                        citation_count=int(item.get("citationCount", 0) or 0),
                        year=int(item.get("year", 0) or 0),
                    ),
                )
            )
        return papers

    def _arxiv(self, query: str, max_results: int) -> list[AcademicPaper]:
        endpoint = "http://export.arxiv.org/api/query"
        response = self._client.get(
            endpoint,
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
            },
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers: list[AcademicPaper] = []
        for entry in root.findall("atom:entry", ns):
            title = (
                entry.findtext("atom:title", default="", namespaces=ns) or ""
            ).strip()
            abstract = (
                entry.findtext("atom:summary", default="", namespaces=ns) or ""
            ).strip()
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = int(published[:4]) if published and published[:4].isdigit() else 0
            authors = [
                (author.findtext("atom:name", default="", namespaces=ns) or "")
                for author in entry.findall("atom:author", ns)
            ]

            papers.append(
                AcademicPaper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    citation_count=0,
                    year=year,
                    source="arxiv",
                    relevance_score=self._score(citation_count=0, year=year),
                )
            )
        return papers

    def _fallback(self, query: str, max_results: int) -> list[AcademicPaper]:
        if self._fallback_search is None:
            return []

        papers: list[AcademicPaper] = []
        for item in self._fallback_search(query)[:max_results]:
            title = str(item.get("title", ""))
            abstract = str(item.get("content", ""))
            papers.append(
                AcademicPaper(
                    title=title,
                    abstract=abstract,
                    authors=[],
                    citation_count=0,
                    year=datetime.now(tz=UTC).year,
                    source="tavily",
                    relevance_score=self._score(
                        citation_count=0, year=datetime.now(tz=UTC).year
                    ),
                )
            )
        return papers

    def _score(self, citation_count: int, year: int) -> float:
        current_year = datetime.now(tz=UTC).year
        recency = max(0, 1 - (current_year - year) / 15) if year else 0.25
        citation = min(citation_count / 500, 1.0)
        return round(0.6 * recency + 0.4 * citation, 3)
