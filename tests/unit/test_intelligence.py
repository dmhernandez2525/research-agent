"""Unit tests for Phase 21 multi-source intelligence providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from research_agent.intelligence.academic import AcademicSearch, ProviderRateLimiter
from research_agent.intelligence.github import GitHubRepositoryAnalyzer
from research_agent.intelligence.rss import RSSMonitor
from research_agent.intelligence.youtube import YouTubeTranscriptExtractor

if TYPE_CHECKING:
    from pathlib import Path


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "https://example.com"),
                response=httpx.Response(self.status_code),
            )


def test_provider_rate_limiter_respects_quota() -> None:
    limiter = ProviderRateLimiter()
    assert limiter.allow("semantic_scholar", per_minute=2) is True
    assert limiter.allow("semantic_scholar", per_minute=2) is True
    assert limiter.allow("semantic_scholar", per_minute=2) is False


def test_academic_search_fallback_on_provider_failure() -> None:
    class _FailingClient:
        def get(self, *_args: Any, **_kwargs: Any) -> _Response:
            raise RuntimeError("provider down")

        def close(self) -> None:
            return

    search = AcademicSearch(
        client=_FailingClient(),  # type: ignore[arg-type]
        fallback_search=lambda query: [
            {"title": f"{query} fallback", "content": "fallback body"}
        ],
    )
    results = search.search("agent systems", max_results=3)
    assert results
    assert results[0].source == "tavily"


def test_github_search_and_dependency_matching() -> None:
    class _Client:
        def get(self, url: str, *_args: Any, **_kwargs: Any) -> _Response:
            if "search/repositories" in url:
                return _Response(
                    payload={
                        "items": [
                            {
                                "full_name": "octo/repo",
                                "description": "demo",
                                "stargazers_count": 42,
                                "language": "Python",
                                "license": {"spdx_id": "MIT"},
                                "pushed_at": "2026-02-01T00:00:00Z",
                            }
                        ]
                    },
                    headers={"x-ratelimit-remaining": "4999"},
                )
            return _Response(
                payload={
                    "content": "IyBESU5HCmBgYHB5dGhvbgoKaW1wb3J0IGZhc3RhcGkKYGBg",
                },
                headers={"x-ratelimit-remaining": "4998"},
            )

        def close(self) -> None:
            return

    analyzer = GitHubRepositoryAnalyzer(client=_Client())  # type: ignore[arg-type]
    repos = analyzer.search_repositories(
        "fastapi",
        project_dependencies=["fastapi", "pydantic"],
        limit=1,
    )
    assert len(repos) == 1
    assert repos[0].full_name == "octo/repo"
    assert repos[0].shared_dependencies == ["fastapi"]
    assert repos[0].code_snippets


def test_youtube_chunking_and_quality_filter(tmp_path: Path) -> None:
    extractor = YouTubeTranscriptExtractor(tmp_path / "yt")
    chunks = extractor._chunk_transcript(
        "[00:00:00] Intro\n[00:01:50] Details\nPlain text segment"
    )
    assert chunks
    assert chunks[0].start_seconds == 0.0

    assert extractor._is_quality_video(
        {
            "duration": 600,
            "language": "en",
            "title": "Production deployment walkthrough",
        }
    )
    assert not extractor._is_quality_video(
        {"duration": 50, "language": "en", "title": "tiny clip"}
    )
    assert not extractor._is_quality_video(
        {"duration": 600, "language": "en", "title": "music video"}
    )


def test_rss_import_poll_incremental_and_dedup(tmp_path: Path) -> None:
    opml_path = tmp_path / "feeds.opml"
    opml_path.write_text(
        """<?xml version="1.0"?>
<opml><body>
  <outline text="A" xmlUrl="https://example.com/feed.xml" />
</body></opml>
""",
        encoding="utf-8",
    )

    rss_xml = """<?xml version="1.0"?>
<rss><channel>
  <title>Feed</title>
  <item>
    <guid>entry-1</guid>
    <title>One</title>
    <link>https://example.com/one</link>
    <description>Summary one</description>
  </item>
</channel></rss>
"""

    class _Client:
        def __init__(self) -> None:
            self.calls = 0
            self.received_headers: dict[str, str] = {}

        def get(self, _url: str, headers: dict[str, str] | None = None) -> _Response:
            self.calls += 1
            if headers:
                self.received_headers = headers
            if self.calls == 1:
                return _Response(
                    text=rss_xml,
                    headers={
                        "etag": '"abc"',
                        "last-modified": "Mon, 01 Jan 2026 00:00:00 GMT",
                    },
                )
            return _Response(status_code=304)

        def close(self) -> None:
            return

    client = _Client()
    monitor = RSSMonitor(tmp_path / "state.json", client=client)  # type: ignore[arg-type]
    feeds = monitor.import_opml(opml_path)
    assert feeds == ["https://example.com/feed.xml"]

    entries_first = monitor.poll(feeds, existing_urls=set())
    assert len(entries_first) == 1
    assert entries_first[0].title == "One"

    entries_second = monitor.poll(feeds, existing_urls={"https://example.com/one"})
    assert entries_second == []
    assert "If-None-Match" in client.received_headers
