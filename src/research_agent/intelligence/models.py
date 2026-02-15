"""Data models for multi-source intelligence providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AcademicPaper:
    """Normalized academic paper metadata."""

    title: str
    abstract: str
    authors: list[str]
    citation_count: int
    year: int
    source: str
    relevance_score: float


@dataclass(slots=True)
class GitHubRepositoryInsight:
    """Repository summary used in research synthesis."""

    full_name: str
    description: str
    stars: int
    language: str
    license_name: str
    last_commit_date: str
    readme_excerpt: str
    shared_dependencies: list[str]
    code_snippets: list[str]


@dataclass(slots=True)
class VideoTranscriptChunk:
    """Timestamped transcript chunk from YouTube content."""

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(slots=True)
class VideoTranscriptResult:
    """Transcript result + metadata for a YouTube video."""

    video_id: str
    title: str
    channel: str
    duration_seconds: int
    views: int
    language: str
    chunks: list[VideoTranscriptChunk]


@dataclass(slots=True)
class FeedEntry:
    """Normalized RSS/Atom feed entry payload."""

    feed_title: str
    entry_id: str
    title: str
    link: str
    published: str
    summary: str
    full_text: str
