"""Unit tests for research_agent.state - data models and ResearchState."""

from __future__ import annotations

import operator
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from research_agent.state import (
    ErrorEntry,
    ResearchState,
    ScrapedPage,
    SearchResult,
    Source,
    Subtopic,
    SubtopicSummary,
)

# ---- Subtopic ----------------------------------------------------------------


class TestSubtopic:
    """Subtopic model construction and validation."""

    def test_valid_construction(self) -> None:
        sq = Subtopic(id=1, question="What is RAG?")
        assert sq.id == 1
        assert sq.question == "What is RAG?"
        assert sq.rationale == ""

    def test_with_rationale(self) -> None:
        sq = Subtopic(id=2, question="How?", rationale="Key context")
        assert sq.rationale == "Key context"


# ---- SearchResult ------------------------------------------------------------


class TestSearchResult:
    """SearchResult model construction and validation."""

    def test_valid_construction(self) -> None:
        sr = SearchResult(subtopic_id=1, query="RAG", url="https://example.com")
        assert sr.subtopic_id == 1
        assert sr.url == "https://example.com"
        assert sr.score == 0.0

    def test_score_bounds(self) -> None:
        sr = SearchResult(subtopic_id=1, query="q", url="https://a.com", score=0.95)
        assert sr.score == 0.95

    def test_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(subtopic_id=1, query="q", url="https://a.com", score=1.5)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(subtopic_id=1, query="q", url="https://a.com", score=-0.1)


# ---- ScrapedPage -------------------------------------------------------------


class TestScrapedPage:
    """ScrapedPage model construction and validation."""

    def test_valid_construction(self) -> None:
        sc = ScrapedPage(url="https://example.com", subtopic_id=1)
        assert sc.content == ""
        assert sc.word_count == 0
        assert sc.quality_score == 0.0

    def test_with_content(self) -> None:
        sc = ScrapedPage(
            url="https://example.com",
            subtopic_id=1,
            content="Some text",
            word_count=2,
            quality_score=0.8,
        )
        assert sc.word_count == 2
        assert sc.quality_score == 0.8

    def test_negative_word_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScrapedPage(url="https://example.com", subtopic_id=1, word_count=-1)

    def test_quality_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScrapedPage(
                url="https://example.com", subtopic_id=1, quality_score=1.5
            )


# ---- SubtopicSummary ---------------------------------------------------------


class TestSubtopicSummary:
    """SubtopicSummary model construction."""

    def test_valid_construction(self) -> None:
        s = SubtopicSummary(subtopic_id=1, summary="Key findings here.")
        assert s.summary == "Key findings here."
        assert s.source_urls == []
        assert s.key_findings == []

    def test_with_sources_and_findings(self) -> None:
        s = SubtopicSummary(
            subtopic_id=1,
            summary="text",
            source_urls=["https://a.com"],
            key_findings=["Finding 1"],
        )
        assert len(s.source_urls) == 1
        assert len(s.key_findings) == 1


# ---- Source ------------------------------------------------------------------


class TestSource:
    """Source model construction."""

    def test_valid_construction(self) -> None:
        s = Source(url="https://example.com")
        assert s.title == ""
        assert s.accessed_at == ""

    def test_with_all_fields(self) -> None:
        s = Source(
            url="https://example.com",
            title="Example",
            accessed_at="2026-01-01T00:00:00Z",
            relevance="Primary source",
        )
        assert s.title == "Example"
        assert s.relevance == "Primary source"


# ---- ErrorEntry --------------------------------------------------------------


class TestErrorEntry:
    """ErrorEntry model construction."""

    def test_valid_construction(self) -> None:
        e = ErrorEntry(step="search", message="Rate limited")
        assert e.step == "search"
        assert e.recoverable is True

    def test_non_recoverable(self) -> None:
        e = ErrorEntry(step="plan", message="Fatal", recoverable=False)
        assert e.recoverable is False


# ---- ResearchState -----------------------------------------------------------


class TestResearchState:
    """ResearchState TypedDict structure and accumulator annotations."""

    def test_can_construct_as_dict(self) -> None:
        state: ResearchState = {"query": "test", "step": "plan", "step_index": 0}
        assert state["query"] == "test"

    def test_total_false_allows_partial(self) -> None:
        state: ResearchState = {"query": "test"}
        assert "step" not in state

    def test_search_results_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        sr_hint = hints["search_results"]
        assert hasattr(sr_hint, "__metadata__")
        assert sr_hint.__metadata__[0] is operator.add

    def test_scraped_pages_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        sc_hint = hints["scraped_pages"]
        assert sc_hint.__metadata__[0] is operator.add

    def test_subtopic_summaries_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        s_hint = hints["subtopic_summaries"]
        assert s_hint.__metadata__[0] is operator.add

    def test_error_log_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        e_hint = hints["error_log"]
        assert e_hint.__metadata__[0] is operator.add

    def test_seen_urls_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        su_hint = hints["seen_urls"]
        assert hasattr(su_hint, "__metadata__")
        assert su_hint.__metadata__[0] is operator.add

    def test_sources_has_add_reducer(self) -> None:
        hints = get_type_hints(ResearchState, include_extras=True)
        src_hint = hints["sources"]
        assert src_hint.__metadata__[0] is operator.add
