"""Tests for multi-source intelligence orchestration."""

from __future__ import annotations

from research_agent.intelligence import orchestrator


def test_gather_multi_source_intelligence_collects_all_sources(
    monkeypatch, tmp_path
) -> None:
    closed: list[str] = []

    class FakeAcademic:
        def search(self, query: str, max_results: int) -> list[dict[str, object]]:
            assert query == "https://youtube.com/watch?v=abc"
            assert max_results == 5
            return [{"paper": "p1"}]

        def close(self) -> None:
            closed.append("academic")

    class FakeGitHub:
        def search_repositories(
            self, query: str, deps: list[str], limit: int
        ) -> list[dict[str, object]]:
            assert query == "https://youtube.com/watch?v=abc"
            assert deps == ["fastapi"]
            assert limit == 5
            return [{"repo": "r1"}]

        def close(self) -> None:
            closed.append("github")

    class FakeRSS:
        def __init__(self, state_path) -> None:
            assert str(state_path).endswith("rss_state.json")

        def poll(
            self, feed_urls: list[str], existing_urls: set[str]
        ) -> list[dict[str, object]]:
            assert feed_urls == ["https://example.com/feed.xml"]
            assert existing_urls == set()
            return [{"entry": "e1"}]

        def close(self) -> None:
            closed.append("rss")

    class FakeYouTube:
        def __init__(self, work_dir) -> None:
            assert str(work_dir).endswith("youtube")

        def extract(self, query: str) -> dict[str, object]:
            assert query.startswith("http")
            return {"video": "v1"}

    monkeypatch.setattr(orchestrator, "AcademicSearch", FakeAcademic)
    monkeypatch.setattr(orchestrator, "GitHubRepositoryAnalyzer", FakeGitHub)
    monkeypatch.setattr(orchestrator, "RSSMonitor", FakeRSS)
    monkeypatch.setattr(orchestrator, "YouTubeTranscriptExtractor", FakeYouTube)

    result = orchestrator.gather_multi_source_intelligence(
        query="https://youtube.com/watch?v=abc",
        project_dependencies=["fastapi"],
        feed_urls=["https://example.com/feed.xml"],
        workspace=tmp_path,
    )

    assert result["academic"] == [{"paper": "p1"}]
    assert result["github"] == [{"repo": "r1"}]
    assert result["rss"] == [{"entry": "e1"}]
    assert result["youtube"] == {"video": "v1"}
    assert set(closed) == {"academic", "github", "rss"}


def test_gather_multi_source_intelligence_skips_optional_sources(
    monkeypatch, tmp_path
) -> None:
    calls: dict[str, int] = {"rss_poll": 0, "youtube_extract": 0}
    closed: list[str] = []

    class FakeAcademic:
        def search(self, query: str, max_results: int) -> list[object]:
            return []

        def close(self) -> None:
            closed.append("academic")

    class FakeGitHub:
        def search_repositories(
            self, query: str, deps: list[str], limit: int
        ) -> list[object]:
            return []

        def close(self) -> None:
            closed.append("github")

    class FakeRSS:
        def __init__(self, state_path) -> None:
            self._state_path = state_path

        def poll(self, feed_urls: list[str], existing_urls: set[str]) -> list[object]:
            calls["rss_poll"] += 1
            return []

        def close(self) -> None:
            closed.append("rss")

    class FakeYouTube:
        def __init__(self, work_dir) -> None:
            self._work_dir = work_dir

        def extract(self, query: str) -> object:
            calls["youtube_extract"] += 1
            return None

    monkeypatch.setattr(orchestrator, "AcademicSearch", FakeAcademic)
    monkeypatch.setattr(orchestrator, "GitHubRepositoryAnalyzer", FakeGitHub)
    monkeypatch.setattr(orchestrator, "RSSMonitor", FakeRSS)
    monkeypatch.setattr(orchestrator, "YouTubeTranscriptExtractor", FakeYouTube)

    result = orchestrator.gather_multi_source_intelligence(
        query="non-url query",
        project_dependencies=[],
        feed_urls=[],
        workspace=tmp_path,
    )

    assert result["rss"] == []
    assert result["youtube"] is None
    assert calls["rss_poll"] == 0
    assert calls["youtube_extract"] == 0
    assert set(closed) == {"academic", "github", "rss"}
