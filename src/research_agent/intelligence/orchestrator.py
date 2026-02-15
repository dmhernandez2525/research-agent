"""Orchestration helpers across multi-source intelligence providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from research_agent.intelligence.academic import AcademicSearch
from research_agent.intelligence.github import GitHubRepositoryAnalyzer
from research_agent.intelligence.rss import RSSMonitor
from research_agent.intelligence.youtube import YouTubeTranscriptExtractor

if TYPE_CHECKING:
    from pathlib import Path


def gather_multi_source_intelligence(
    query: str,
    project_dependencies: list[str],
    feed_urls: list[str],
    workspace: Path,
) -> dict[str, Any]:
    """Collect intelligence from academic, GitHub, YouTube, and RSS sources."""
    academic = AcademicSearch()
    github = GitHubRepositoryAnalyzer()
    rss = RSSMonitor(workspace / "rss_state.json")
    youtube = YouTubeTranscriptExtractor(workspace / "youtube")

    try:
        papers = academic.search(query, max_results=5)
        repos = github.search_repositories(query, project_dependencies, limit=5)
        feed_entries = rss.poll(feed_urls, existing_urls=set()) if feed_urls else []
        video = youtube.extract(query) if query.startswith("http") else None

        return {
            "academic": papers,
            "github": repos,
            "rss": feed_entries,
            "youtube": video,
        }
    finally:
        academic.close()
        github.close()
        rss.close()
