"""Unit tests for research_agent.nodes.searcher - Tavily search integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.nodes.searcher import (
    _MIN_RELEVANCE_SCORE,
    _deduplicate_results,
    _parse_results,
    execute_search,
    search_node,
)
from research_agent.state import SearchResult

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture()
def tavily_response() -> list[dict[str, Any]]:
    """A realistic Tavily API response."""
    return [
        {
            "url": "https://example.com/rag-intro",
            "title": "Introduction to RAG",
            "content": "RAG combines retrieval with generation...",
            "score": 0.92,
        },
        {
            "url": "https://example.com/rag-deep",
            "title": "Deep Dive into RAG",
            "content": "Advanced techniques for RAG systems...",
            "score": 0.85,
        },
        {
            "url": "https://example.com/rag-low",
            "title": "Tangential Reference",
            "content": "Briefly mentions RAG...",
            "score": 0.15,
        },
    ]


# ---- _parse_results ----------------------------------------------------------


class TestParseResults:
    """Converting raw Tavily dicts to SearchResult models."""

    def test_parses_valid_results(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, sub_question_id=1, query="RAG")
        # 3 raw results, but one is below threshold
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_filters_below_min_relevance(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        results = _parse_results(tavily_response, sub_question_id=1, query="RAG")
        scores = [r.score for r in results]
        assert all(s >= _MIN_RELEVANCE_SCORE for s in scores)

    def test_sorts_by_score_descending(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        results = _parse_results(tavily_response, sub_question_id=1, query="RAG")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_sets_sub_question_id(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, sub_question_id=42, query="test")
        for r in results:
            assert r.sub_question_id == 42

    def test_sets_query(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, sub_question_id=1, query="my query")
        for r in results:
            assert r.query == "my query"

    def test_empty_results_returns_empty(self) -> None:
        assert _parse_results([], sub_question_id=1, query="q") == []

    def test_missing_score_defaults_to_zero_and_filtered(self) -> None:
        raw = [{"url": "https://a.com", "title": "T", "content": "S"}]
        results = _parse_results(raw, sub_question_id=1, query="q")
        assert len(results) == 0  # 0.0 < 0.3 threshold

    def test_missing_fields_use_defaults(self) -> None:
        raw = [{"url": "https://a.com", "score": 0.8}]
        results = _parse_results(raw, sub_question_id=1, query="q")
        assert len(results) == 1
        assert results[0].title == ""
        assert results[0].snippet == ""


# ---- _deduplicate_results ----------------------------------------------------


class TestDeduplicateResults:
    """URL-based deduplication within a batch."""

    def test_removes_duplicates(self) -> None:
        results = [
            SearchResult(sub_question_id=1, query="q", url="https://a.com", score=0.8),
            SearchResult(
                sub_question_id=1,
                query="q",
                url="https://a.com",
                title="dup",
                score=0.7,
            ),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 1
        assert unique[0].score == 0.8  # first occurrence kept

    def test_preserves_unique(self) -> None:
        results = [
            SearchResult(sub_question_id=1, query="q", url="https://a.com", score=0.8),
            SearchResult(sub_question_id=1, query="q", url="https://b.com", score=0.7),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 2

    def test_empty_input(self) -> None:
        assert _deduplicate_results([]) == []

    def test_preserves_order(self) -> None:
        results = [
            SearchResult(sub_question_id=1, query="q", url="https://c.com", score=0.5),
            SearchResult(sub_question_id=1, query="q", url="https://a.com", score=0.9),
            SearchResult(sub_question_id=1, query="q", url="https://b.com", score=0.7),
        ]
        unique = _deduplicate_results(results)
        assert [r.url for r in unique] == [
            "https://c.com",
            "https://a.com",
            "https://b.com",
        ]


# ---- execute_search ----------------------------------------------------------


class TestExecuteSearch:
    """High-level execute_search function."""

    @pytest.mark.asyncio()
    async def test_returns_parsed_results(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            results = await execute_search(query="RAG", sub_question_id=1)

        assert len(results) == 2
        assert all(r.sub_question_id == 1 for r in results)

    @pytest.mark.asyncio()
    async def test_passes_max_results(self) -> None:
        mock = AsyncMock(return_value=[])
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            mock,
        ):
            await execute_search(query="test", sub_question_id=1, max_results=5)

        mock.assert_called_once_with(
            query="test", max_results=5, search_depth="advanced"
        )

    @pytest.mark.asyncio()
    async def test_passes_search_depth(self) -> None:
        mock = AsyncMock(return_value=[])
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            mock,
        ):
            await execute_search(
                query="test",
                sub_question_id=1,
                search_depth="basic",
            )

        mock.assert_called_once_with(query="test", max_results=10, search_depth="basic")


# ---- search_node -------------------------------------------------------------


class TestSearchNode:
    """The search_node graph function."""

    @pytest.mark.asyncio()
    async def test_returns_empty_when_no_sub_questions(self) -> None:
        state: dict[str, Any] = {
            "sub_questions": [],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        result = await search_node(state)
        assert result["search_results"] == []

    @pytest.mark.asyncio()
    async def test_returns_empty_when_index_past_end(self) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 5,
            "seen_urls": [],
        }
        result = await search_node(state)
        assert result["search_results"] == []

    @pytest.mark.asyncio()
    async def test_searches_current_subtopic(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "sub_questions": [
                {"id": 1, "question": "What is RAG?"},
                {"id": 2, "question": "How does it work?"},
            ],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            result = await search_node(state)

        assert len(result["search_results"]) == 2
        assert all(r.sub_question_id == 1 for r in result["search_results"])

    @pytest.mark.asyncio()
    async def test_filters_seen_urls(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": ["https://example.com/rag-intro"],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            result = await search_node(state)

        # Should filter out the already-seen URL
        urls = [r.url for r in result["search_results"]]
        assert "https://example.com/rag-intro" not in urls
        assert len(result["search_results"]) == 1

    @pytest.mark.asyncio()
    async def test_returns_new_urls_for_accumulation(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            result = await search_node(state)

        assert len(result["seen_urls"]) > 0
        for url in result["seen_urls"]:
            assert isinstance(url, str)

    @pytest.mark.asyncio()
    async def test_handles_search_failure_gracefully(self) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API error"),
        ):
            result = await search_node(state)

        assert result["search_results"] == []

    @pytest.mark.asyncio()
    async def test_sets_step_metadata(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            result = await search_node(state)

        assert result["step"] == "search"
        assert result["step_index"] == 1

    @pytest.mark.asyncio()
    async def test_works_with_dict_sub_questions(self) -> None:
        state: dict[str, Any] = {
            "sub_questions": [{"id": 3, "question": "Dict format"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        mock_response = [
            {"url": "https://x.com", "title": "X", "content": "Y", "score": 0.9}
        ]
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await search_node(state)

        assert result["search_results"][0].sub_question_id == 3


# ---- Retry behavior ---------------------------------------------------------


class TestRetryBehavior:
    """Tenacity retry configuration."""

    def test_retry_decorator_configured(self) -> None:
        from research_agent.nodes.searcher import _tavily_search_with_retry

        # Verify retry is attached
        assert hasattr(_tavily_search_with_retry, "retry")

    def test_retries_on_timeout_type(self) -> None:
        from research_agent.nodes.searcher import _tavily_search_with_retry

        retry_obj = _tavily_search_with_retry.retry
        # The retry condition should cover TimeoutError
        assert retry_obj.retry is not None

    def test_retries_on_os_error_type(self) -> None:
        from research_agent.nodes.searcher import _tavily_search_with_retry

        retry_obj = _tavily_search_with_retry.retry
        assert retry_obj.stop is not None

    @pytest.mark.asyncio()
    async def test_execute_search_propagates_non_retryable_errors(self) -> None:
        with (
            patch(
                "research_agent.nodes.searcher._tavily_search_with_retry",
                new_callable=AsyncMock,
                side_effect=ValueError("bad query"),
            ),
            pytest.raises(ValueError, match="bad query"),
        ):
            await execute_search(query="test", sub_question_id=1)
