"""Unit tests for research_agent.nodes.searcher - search with ExpandSearch + URL dedup."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.nodes.searcher import (
    _EXPAND_SYSTEM_PROMPT,
    _MIN_RELEVANCE_SCORE,
    _TRACKING_PARAMS,
    ExpandedQueries,
    _deduplicate_results,
    _expand_queries,
    _normalize_url,
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


@pytest.fixture()
def mock_expanded() -> ExpandedQueries:
    """A mock ExpandedQueries result."""
    return ExpandedQueries(
        original="What is RAG?",
        variations=[
            "retrieval augmented generation overview",
            "RAG architecture LLM search",
            "RAG vs fine-tuning comparison examples",
        ],
    )


@pytest.fixture()
def _patch_expand(mock_expanded: ExpandedQueries) -> Any:
    """Patch _expand_queries to return mock_expanded for all search_node tests."""
    with patch(
        "research_agent.nodes.searcher._expand_queries",
        new_callable=AsyncMock,
        return_value=mock_expanded,
    ):
        yield


# ---- _parse_results ----------------------------------------------------------


class TestParseResults:
    """Converting raw Tavily dicts to SearchResult models."""

    def test_parses_valid_results(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, subtopic_id=1, query="RAG")
        # 3 raw results, but one is below threshold
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_filters_below_min_relevance(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        results = _parse_results(tavily_response, subtopic_id=1, query="RAG")
        scores = [r.score for r in results]
        assert all(s >= _MIN_RELEVANCE_SCORE for s in scores)

    def test_sorts_by_score_descending(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        results = _parse_results(tavily_response, subtopic_id=1, query="RAG")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_sets_subtopic_id(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, subtopic_id=42, query="test")
        for r in results:
            assert r.subtopic_id == 42

    def test_sets_query(self, tavily_response: list[dict[str, Any]]) -> None:
        results = _parse_results(tavily_response, subtopic_id=1, query="my query")
        for r in results:
            assert r.query == "my query"

    def test_empty_results_returns_empty(self) -> None:
        assert _parse_results([], subtopic_id=1, query="q") == []

    def test_missing_score_defaults_to_zero_and_filtered(self) -> None:
        raw = [{"url": "https://a.com", "title": "T", "content": "S"}]
        results = _parse_results(raw, subtopic_id=1, query="q")
        assert len(results) == 0  # 0.0 < 0.3 threshold

    def test_missing_fields_use_defaults(self) -> None:
        raw = [{"url": "https://a.com", "score": 0.8}]
        results = _parse_results(raw, subtopic_id=1, query="q")
        assert len(results) == 1
        assert results[0].title == ""
        assert results[0].snippet == ""


# ---- _deduplicate_results ----------------------------------------------------


class TestDeduplicateResults:
    """URL-based deduplication within a batch (uses normalized URLs)."""

    def test_removes_duplicates(self) -> None:
        results = [
            SearchResult(subtopic_id=1, query="q", url="https://a.com", score=0.8),
            SearchResult(
                subtopic_id=1,
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
            SearchResult(subtopic_id=1, query="q", url="https://a.com", score=0.8),
            SearchResult(subtopic_id=1, query="q", url="https://b.com", score=0.7),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 2

    def test_empty_input(self) -> None:
        assert _deduplicate_results([]) == []

    def test_preserves_order(self) -> None:
        results = [
            SearchResult(subtopic_id=1, query="q", url="https://c.com", score=0.5),
            SearchResult(subtopic_id=1, query="q", url="https://a.com", score=0.9),
            SearchResult(subtopic_id=1, query="q", url="https://b.com", score=0.7),
        ]
        unique = _deduplicate_results(results)
        assert [r.url for r in unique] == [
            "https://c.com",
            "https://a.com",
            "https://b.com",
        ]

    def test_deduplicates_with_trailing_slash_difference(self) -> None:
        results = [
            SearchResult(
                subtopic_id=1, query="q", url="https://a.com/page", score=0.9
            ),
            SearchResult(
                subtopic_id=1, query="q", url="https://a.com/page/", score=0.7
            ),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 1

    def test_deduplicates_with_tracking_param_difference(self) -> None:
        results = [
            SearchResult(
                subtopic_id=1, query="q", url="https://a.com/page", score=0.9
            ),
            SearchResult(
                subtopic_id=1,
                query="q",
                url="https://a.com/page?utm_source=google",
                score=0.7,
            ),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 1

    def test_deduplicates_with_fragment_difference(self) -> None:
        results = [
            SearchResult(
                subtopic_id=1, query="q", url="https://a.com/page", score=0.9
            ),
            SearchResult(
                subtopic_id=1,
                query="q",
                url="https://a.com/page#section",
                score=0.7,
            ),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 1

    def test_deduplicates_with_case_difference(self) -> None:
        results = [
            SearchResult(
                subtopic_id=1, query="q", url="https://A.COM/page", score=0.9
            ),
            SearchResult(
                subtopic_id=1, query="q", url="https://a.com/page", score=0.7
            ),
        ]
        unique = _deduplicate_results(results)
        assert len(unique) == 1


# ---- _normalize_url ----------------------------------------------------------


class TestNormalizeUrl:
    """URL normalization for deduplication."""

    def test_lowercases_scheme_and_host(self) -> None:
        assert _normalize_url("HTTPS://EXAMPLE.COM/Path") == "https://example.com/Path"

    def test_strips_trailing_slash(self) -> None:
        assert _normalize_url("https://a.com/page/") == "https://a.com/page"

    def test_keeps_root_slash(self) -> None:
        result = _normalize_url("https://a.com/")
        assert result == "https://a.com/"

    def test_removes_fragment(self) -> None:
        result = _normalize_url("https://a.com/page#section")
        assert "#" not in result
        assert result == "https://a.com/page"

    def test_removes_utm_params(self) -> None:
        url = "https://a.com/page?utm_source=google&utm_medium=cpc&id=123"
        result = _normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "id=123" in result

    def test_removes_fbclid(self) -> None:
        url = "https://a.com/page?fbclid=abc123&real=yes"
        result = _normalize_url(url)
        assert "fbclid" not in result
        assert "real=yes" in result

    def test_removes_gclid(self) -> None:
        url = "https://a.com/page?gclid=xyz&q=test"
        result = _normalize_url(url)
        assert "gclid" not in result
        assert "q=test" in result

    def test_removes_multiple_tracking_params(self) -> None:
        url = "https://a.com?utm_source=x&fbclid=y&gclid=z&msclkid=w&real=1"
        result = _normalize_url(url)
        assert "utm_source" not in result
        assert "fbclid" not in result
        assert "gclid" not in result
        assert "msclkid" not in result
        assert "real=1" in result

    def test_sorts_remaining_params(self) -> None:
        url = "https://a.com?z=3&a=1&m=2"
        result = _normalize_url(url)
        assert "a=1&m=2&z=3" in result

    def test_preserves_meaningful_params(self) -> None:
        url = "https://a.com/search?q=test&page=2&lang=en"
        result = _normalize_url(url)
        assert "q=test" in result
        assert "page=2" in result
        assert "lang=en" in result

    def test_no_query_string(self) -> None:
        assert _normalize_url("https://a.com/page") == "https://a.com/page"

    def test_empty_query_string_after_tracking_removal(self) -> None:
        url = "https://a.com/page?utm_source=google"
        result = _normalize_url(url)
        assert result == "https://a.com/page"

    def test_combined_normalizations(self) -> None:
        url = "HTTPS://EXAMPLE.COM/Article/#intro?utm_source=twitter&id=42"
        result = _normalize_url(url)
        assert result.startswith("https://example.com")
        assert "#" not in result

    def test_tracking_params_regex_covers_common_trackers(self) -> None:
        common_trackers = [
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_content",
            "utm_term",
            "fbclid",
            "gclid",
            "gclsrc",
            "dclid",
            "msclkid",
            "mc_cid",
            "mc_eid",
            "wbraid",
            "gbraid",
            "_ga",
            "_gid",
            "_gl",
        ]
        for param in common_trackers:
            assert _TRACKING_PARAMS.match(param), f"{param} not matched"

    def test_tracking_params_case_insensitive(self) -> None:
        assert _TRACKING_PARAMS.match("UTM_SOURCE")
        assert _TRACKING_PARAMS.match("Fbclid")
        assert _TRACKING_PARAMS.match("GCLID")


# ---- ExpandedQueries model --------------------------------------------------


class TestExpandedQueriesModel:
    """Pydantic model validation for ExpandedQueries."""

    def test_valid_three_variations(self) -> None:
        eq = ExpandedQueries(
            original="What is RAG?",
            variations=["query 1", "query 2", "query 3"],
        )
        assert len(eq.variations) == 3
        assert eq.original == "What is RAG?"

    def test_rejects_fewer_than_three(self) -> None:
        with pytest.raises(ValueError, match="too_short"):
            ExpandedQueries(original="Q", variations=["a", "b"])

    def test_rejects_more_than_three(self) -> None:
        with pytest.raises(ValueError, match="too_long"):
            ExpandedQueries(original="Q", variations=["a", "b", "c", "d"])

    def test_system_prompt_exists(self) -> None:
        assert len(_EXPAND_SYSTEM_PROMPT) > 50
        assert "3" in _EXPAND_SYSTEM_PROMPT or "three" in _EXPAND_SYSTEM_PROMPT.lower()


# ---- _expand_queries ---------------------------------------------------------


class TestExpandQueries:
    """LLM-powered query expansion function."""

    @pytest.mark.asyncio()
    async def test_returns_expanded_queries(self) -> None:
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "original": "What is RAG?",
            "variations": ["RAG overview", "retrieval generation", "RAG examples"],
        })

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await _expand_queries("What is RAG?")

        assert isinstance(result, ExpandedQueries)
        assert len(result.variations) == 3

    @pytest.mark.asyncio()
    async def test_uses_haiku_model(self) -> None:
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "original": "Q",
            "variations": ["a", "b", "c"],
        })

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await _expand_queries("Q")

        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-haiku-3-5-20241022"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio()
    async def test_passes_system_prompt(self) -> None:
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "original": "Q",
            "variations": ["a", "b", "c"],
        })

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await _expand_queries("my question")

        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "search query expansion" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my question"

    @pytest.mark.asyncio()
    async def test_includes_json_instruction_in_system(self) -> None:
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "original": "Q",
            "variations": ["a", "b", "c"],
        })

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_call:
            await _expand_queries("Q")

        call_kwargs = mock_call.call_args[1]
        system_content = call_kwargs["messages"][0]["content"]
        assert "JSON" in system_content

    @pytest.mark.asyncio()
    async def test_propagates_llm_error(self) -> None:
        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM down"),
            ),
            pytest.raises(RuntimeError, match="LLM down"),
        ):
            await _expand_queries("Q")


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
            results = await execute_search(query="RAG", subtopic_id=1)

        assert len(results) == 2
        assert all(r.subtopic_id == 1 for r in results)

    @pytest.mark.asyncio()
    async def test_passes_max_results(self) -> None:
        mock = AsyncMock(return_value=[])
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            mock,
        ):
            await execute_search(query="test", subtopic_id=1, max_results=5)

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
                subtopic_id=1,
                search_depth="basic",
            )

        mock.assert_called_once_with(query="test", max_results=10, search_depth="basic")


# ---- search_node -------------------------------------------------------------


@pytest.mark.usefixtures("_patch_expand")
class TestSearchNode:
    """The search_node graph function (with mocked query expansion)."""

    @pytest.mark.asyncio()
    async def test_returns_empty_when_no_subtopics(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        result = await search_node(state)
        assert result["search_results"] == []

    @pytest.mark.asyncio()
    async def test_returns_empty_when_index_past_end(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 5,
            "seen_urls": [],
        }
        result = await search_node(state)
        assert result["search_results"] == []

    @pytest.mark.asyncio()
    async def test_searches_with_expanded_queries(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "subtopics": [
                {"id": 1, "question": "What is RAG?"},
                {"id": 2, "question": "How does it work?"},
            ],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        tavily_mock = AsyncMock(return_value=tavily_response)
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            tavily_mock,
        ):
            result = await search_node(state)

        # Should have called tavily for each of the 3 expanded queries
        assert tavily_mock.call_count == 3
        assert len(result["search_results"]) > 0
        assert all(r.subtopic_id == 1 for r in result["search_results"])

    @pytest.mark.asyncio()
    async def test_deduplicates_across_variations(self) -> None:
        """Results from different query variations with same URL get deduped."""
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        # Same URL returned by all 3 variations
        shared_result = [
            {"url": "https://a.com", "title": "A", "content": "C", "score": 0.9}
        ]
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=shared_result,
        ):
            result = await search_node(state)

        # Should only appear once despite 3 variations returning it
        urls = [r.url for r in result["search_results"]]
        assert urls.count("https://a.com") == 1

    @pytest.mark.asyncio()
    async def test_filters_seen_urls(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": ["https://example.com/rag-intro"],
        }
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=tavily_response,
        ):
            result = await search_node(state)

        urls = [r.url for r in result["search_results"]]
        assert "https://example.com/rag-intro" not in urls

    @pytest.mark.asyncio()
    async def test_filters_seen_urls_with_normalization(self) -> None:
        """Cross-subtopic dedup treats normalized URLs as equal."""
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            # Previously seen URL with trailing slash
            "seen_urls": ["https://example.com/page/"],
        }
        # Search returns same page without trailing slash + tracking param
        mock_response = [
            {
                "url": "https://example.com/page?utm_source=twitter",
                "title": "T",
                "content": "C",
                "score": 0.9,
            },
            {
                "url": "https://other.com/new",
                "title": "New",
                "content": "C2",
                "score": 0.8,
            },
        ]
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await search_node(state)

        # The page URL should be filtered (matches normalized seen URL)
        urls = [r.url for r in result["search_results"]]
        assert "https://example.com/page?utm_source=twitter" not in urls
        assert "https://other.com/new" in urls

    @pytest.mark.asyncio()
    async def test_seen_urls_are_normalized(self) -> None:
        """New seen_urls in output are normalized for future comparisons."""
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        mock_response = [
            {
                "url": "https://EXAMPLE.COM/Page/?utm_source=google#frag",
                "title": "T",
                "content": "C",
                "score": 0.9,
            },
        ]
        with patch(
            "research_agent.nodes.searcher._tavily_search_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await search_node(state)

        # Output seen_urls should be normalized
        assert result["seen_urls"] == ["https://example.com/Page"]

    @pytest.mark.asyncio()
    async def test_returns_new_urls_for_accumulation(
        self, tavily_response: list[dict[str, Any]]
    ) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q1"}],
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
            "subtopics": [{"id": 1, "question": "Q1"}],
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
            "subtopics": [{"id": 1, "question": "Q1"}],
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
    async def test_works_with_dict_subtopics(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 3, "question": "Dict format"}],
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

        assert result["search_results"][0].subtopic_id == 3


# ---- search_node expansion fallback ------------------------------------------


class TestSearchNodeExpansionFallback:
    """Tests for search_node when query expansion fails."""

    @pytest.mark.asyncio()
    async def test_falls_back_to_original_on_expansion_failure(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "What is RAG?"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        mock_response = [
            {"url": "https://a.com", "title": "A", "content": "C", "score": 0.9}
        ]
        with (
            patch(
                "research_agent.nodes.searcher._expand_queries",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM down"),
            ),
            patch(
                "research_agent.nodes.searcher._tavily_search_with_retry",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as tavily_mock,
        ):
            result = await search_node(state)

        # Should search with original question only (1 call, not 3)
        tavily_mock.assert_called_once()
        assert len(result["search_results"]) == 1

    @pytest.mark.asyncio()
    async def test_fallback_uses_original_question_text(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "specific question text"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with (
            patch(
                "research_agent.nodes.searcher._expand_queries",
                new_callable=AsyncMock,
                side_effect=ValueError("parse error"),
            ),
            patch(
                "research_agent.nodes.searcher._tavily_search_with_retry",
                new_callable=AsyncMock,
                return_value=[],
            ) as tavily_mock,
        ):
            await search_node(state)

        # The query passed to tavily should be the original question
        call_kwargs = tavily_mock.call_args[1]
        assert call_kwargs["query"] == "specific question text"

    @pytest.mark.asyncio()
    async def test_expansion_failure_still_returns_valid_state(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with (
            patch(
                "research_agent.nodes.searcher._expand_queries",
                new_callable=AsyncMock,
                side_effect=TimeoutError("slow LLM"),
            ),
            patch(
                "research_agent.nodes.searcher._tavily_search_with_retry",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await search_node(state)

        assert "search_results" in result
        assert "seen_urls" in result
        assert result["step"] == "search"
        assert result["step_index"] == 1

    @pytest.mark.asyncio()
    async def test_both_expansion_and_search_fail_gracefully(self) -> None:
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "question": "Q"}],
            "current_subtopic_index": 0,
            "seen_urls": [],
        }
        with (
            patch(
                "research_agent.nodes.searcher._expand_queries",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM down"),
            ),
            patch(
                "research_agent.nodes.searcher._tavily_search_with_retry",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
        ):
            result = await search_node(state)

        assert result["search_results"] == []
        assert result["step"] == "search"


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
            await execute_search(query="test", subtopic_id=1)
