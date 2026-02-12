"""Unit tests for research_agent.nodes.summarizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.nodes.summarizer import (
    SummarizerOutput,
    _build_content_block,
    _group_content_by_question,
    _summarize_group,
    summarize_node,
)
from research_agent.state import ScrapedPage, SubtopicSummary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scraped_items() -> list[ScrapedPage]:
    """Multiple scraped content items across two sub-questions."""
    return [
        ScrapedPage(
            url="https://a.com/1",
            subtopic_id=1,
            title="Article A",
            content="Content about topic A with many details and findings.",
            word_count=50,
            quality_score=0.8,
        ),
        ScrapedPage(
            url="https://b.com/2",
            subtopic_id=1,
            title="Article B",
            content="More content about topic A from a different source.",
            word_count=40,
            quality_score=0.7,
        ),
        ScrapedPage(
            url="https://c.com/3",
            subtopic_id=2,
            title="Article C",
            content="Content about topic B entirely.",
            word_count=30,
            quality_score=0.9,
        ),
    ]


@pytest.fixture()
def mock_summarizer_result() -> SummarizerOutput:
    """A valid SummarizerOutput for mocking LLM responses."""
    return SummarizerOutput(
        summary="This is a comprehensive summary of the research findings.",
        key_findings=[
            "Finding one about the topic",
            "Finding two with specific data",
            "Finding three about implications",
        ],
        disagreements="Source A and B disagree on the timeline.",
    )


@pytest.fixture()
def sample_subtopics() -> list[dict]:
    """Subtopics as dicts (as stored in state)."""
    return [
        {"id": 1, "question": "What is topic A?", "rationale": "Core question"},
        {"id": 2, "question": "What is topic B?", "rationale": "Supporting question"},
    ]


# ---------------------------------------------------------------------------
# TestGroupContentByQuestion
# ---------------------------------------------------------------------------


class TestGroupContentByQuestion:
    """_group_content_by_question groups scraped content by subtopic_id."""

    def test_groups_by_subtopic_id(
        self, scraped_items: list[ScrapedPage]
    ) -> None:
        groups = _group_content_by_question(scraped_items)
        assert set(groups.keys()) == {1, 2}
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1

    def test_empty_input_returns_empty(self) -> None:
        groups = _group_content_by_question([])
        assert groups == {}

    def test_single_group(self) -> None:
        items = [
            ScrapedPage(
                url="https://a.com", subtopic_id=1, content="text", word_count=5
            ),
            ScrapedPage(
                url="https://b.com", subtopic_id=1, content="more", word_count=5
            ),
        ]
        groups = _group_content_by_question(items)
        assert list(groups.keys()) == [1]
        assert len(groups[1]) == 2

    def test_preserves_item_order(self, scraped_items: list[ScrapedPage]) -> None:
        groups = _group_content_by_question(scraped_items)
        assert groups[1][0].url == "https://a.com/1"
        assert groups[1][1].url == "https://b.com/2"


# ---------------------------------------------------------------------------
# TestBuildContentBlock
# ---------------------------------------------------------------------------


class TestBuildContentBlock:
    """_build_content_block formats scraped items for the LLM prompt."""

    def test_single_item_format(self) -> None:
        items = [
            ScrapedPage(
                url="https://example.com",
                subtopic_id=1,
                title="Test Article",
                content="Article content here.",
                word_count=3,
            ),
        ]
        result = _build_content_block(items)
        assert "Source: Test Article (https://example.com)" in result
        assert "Article content here." in result

    def test_multiple_items_separated_by_rule(
        self, scraped_items: list[ScrapedPage]
    ) -> None:
        result = _build_content_block(scraped_items[:2])
        assert "---" in result
        assert "Article A" in result
        assert "Article B" in result

    def test_empty_items_returns_empty(self) -> None:
        result = _build_content_block([])
        assert result == ""

    def test_preserves_content(self) -> None:
        items = [
            ScrapedPage(
                url="https://x.com",
                subtopic_id=1,
                title="X",
                content="Specific data: 42% increase in Q3.",
                word_count=7,
            ),
        ]
        result = _build_content_block(items)
        assert "42% increase in Q3" in result


# ---------------------------------------------------------------------------
# TestSummarizerOutput
# ---------------------------------------------------------------------------


class TestSummarizerOutput:
    """SummarizerOutput validates structured LLM responses."""

    def test_valid_output(self) -> None:
        output = SummarizerOutput(
            summary="Summary text.",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            disagreements="None noted.",
        )
        assert output.summary == "Summary text."
        assert len(output.key_findings) == 3

    def test_disagreements_default_empty(self) -> None:
        output = SummarizerOutput(
            summary="Summary.",
            key_findings=["Finding 1"],
        )
        assert output.disagreements == ""

    def test_requires_at_least_one_finding(self) -> None:
        with pytest.raises(ValueError):
            SummarizerOutput(summary="Summary.", key_findings=[])


# ---------------------------------------------------------------------------
# TestSummarizeGroup
# ---------------------------------------------------------------------------


class TestSummarizeGroup:
    """_summarize_group calls the LLM and returns a Summary model."""

    def _make_mock_response(self, summarizer_result: SummarizerOutput) -> MagicMock:
        """Build a mock litellm response from a SummarizerOutput."""
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "summary": summarizer_result.summary,
            "key_findings": summarizer_result.key_findings,
            "disagreements": summarizer_result.disagreements,
        })
        return mock_response

    @pytest.mark.asyncio()
    async def test_returns_summary_model(
        self,
        scraped_items: list[ScrapedPage],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_response = self._make_mock_response(mock_summarizer_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "You are a summarizer.",
                "user": "Sub-question: {sub_question}\nContent from {num_sources} sources:\n{content}",
            }
            result = await _summarize_group(1, "What is topic A?", scraped_items[:2])

        assert isinstance(result, SubtopicSummary)
        assert result.subtopic_id == 1
        assert result.sub_question == "What is topic A?"
        assert result.summary == mock_summarizer_result.summary
        assert result.key_findings == mock_summarizer_result.key_findings

    @pytest.mark.asyncio()
    async def test_extracts_source_urls(
        self,
        scraped_items: list[ScrapedPage],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_response = self._make_mock_response(mock_summarizer_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{sub_question} {num_sources} {content}",
            }
            result = await _summarize_group(1, "Question?", scraped_items[:2])

        assert set(result.source_urls) == {"https://a.com/1", "https://b.com/2"}

    @pytest.mark.asyncio()
    async def test_passes_content_to_llm(
        self,
        scraped_items: list[ScrapedPage],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_response = self._make_mock_response(mock_summarizer_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{sub_question} {num_sources} {content}",
            }
            await _summarize_group(1, "What is topic A?", scraped_items[:2])

        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is topic A?" in messages[1]["content"]

    @pytest.mark.asyncio()
    async def test_uses_smart_tier_model(
        self,
        scraped_items: list[ScrapedPage],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_response = self._make_mock_response(mock_summarizer_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{sub_question} {num_sources} {content}",
            }
            await _summarize_group(1, "Question?", scraped_items[:1])

        call_kwargs = mock_call.call_args[1]
        assert "sonnet" in call_kwargs["model"]


# ---------------------------------------------------------------------------
# TestSummarizeNode
# ---------------------------------------------------------------------------


class TestSummarizeNode:
    """summarize_node orchestrates per-subtopic summarization."""

    @pytest.mark.asyncio()
    async def test_returns_summary_for_current_subtopic(
        self,
        scraped_items: list[ScrapedPage],
        sample_subtopics: list[dict],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        state = {
            "scraped_pages": scraped_items,
            "subtopics": sample_subtopics,
            "current_subtopic_index": 0,
        }

        mock_summary = SubtopicSummary(
            subtopic_id=1,
            sub_question="What is topic A?",
            summary="LLM summary.",
            source_urls=["https://a.com/1"],
            key_findings=["Finding 1"],
        )

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ):
            result = await summarize_node(state)

        assert len(result["subtopic_summaries"]) == 1
        assert result["subtopic_summaries"][0].subtopic_id == 1

    @pytest.mark.asyncio()
    async def test_increments_subtopic_index(
        self,
        scraped_items: list[ScrapedPage],
        sample_subtopics: list[dict],
    ) -> None:
        state = {
            "scraped_pages": scraped_items,
            "subtopics": sample_subtopics,
            "current_subtopic_index": 0,
        }

        mock_summary = SubtopicSummary(
            subtopic_id=1,
            sub_question="Q",
            summary="S",
            key_findings=["F"],
        )

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ):
            result = await summarize_node(state)

        assert result["current_subtopic_index"] == 1

    @pytest.mark.asyncio()
    async def test_empty_scraped_pages_returns_empty(
        self,
        sample_subtopics: list[dict],
    ) -> None:
        state = {
            "scraped_pages": [],
            "subtopics": sample_subtopics,
            "current_subtopic_index": 0,
        }
        result = await summarize_node(state)
        assert result["subtopic_summaries"] == []
        assert result["current_subtopic_index"] == 1

    @pytest.mark.asyncio()
    async def test_no_subtopics_returns_empty(self) -> None:
        state = {
            "scraped_pages": [],
            "subtopics": [],
            "current_subtopic_index": 0,
        }
        result = await summarize_node(state)
        assert result["subtopic_summaries"] == []

    @pytest.mark.asyncio()
    async def test_index_out_of_range_returns_empty(
        self,
        sample_subtopics: list[dict],
    ) -> None:
        state = {
            "scraped_pages": [],
            "subtopics": sample_subtopics,
            "current_subtopic_index": 10,
        }
        result = await summarize_node(state)
        assert result["subtopic_summaries"] == []

    @pytest.mark.asyncio()
    async def test_sets_step_metadata(
        self,
        scraped_items: list[ScrapedPage],
        sample_subtopics: list[dict],
    ) -> None:
        state = {
            "scraped_pages": scraped_items,
            "subtopics": sample_subtopics,
            "current_subtopic_index": 0,
        }

        mock_summary = SubtopicSummary(
            subtopic_id=1,
            sub_question="Q",
            summary="S",
            key_findings=["F"],
        )

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ):
            result = await summarize_node(state)

        assert result["step"] == "summarize"
        assert result["step_index"] == 3

    @pytest.mark.asyncio()
    async def test_handles_llm_failure_gracefully(
        self,
        scraped_items: list[ScrapedPage],
        sample_subtopics: list[dict],
    ) -> None:
        state = {
            "scraped_pages": scraped_items,
            "subtopics": sample_subtopics,
            "current_subtopic_index": 0,
        }

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM API error"),
        ):
            result = await summarize_node(state)

        assert result["subtopic_summaries"] == []
        assert result["current_subtopic_index"] == 1

    @pytest.mark.asyncio()
    async def test_filters_content_for_current_subtopic_only(
        self,
        scraped_items: list[ScrapedPage],
        sample_subtopics: list[dict],
    ) -> None:
        """Only content matching the current subtopic_id is passed to _summarize_group."""
        state = {
            "scraped_pages": scraped_items,
            "subtopics": sample_subtopics,
            "current_subtopic_index": 1,  # subtopic_id=2
        }

        mock_summary = SubtopicSummary(
            subtopic_id=2,
            sub_question="What is topic B?",
            summary="Summary B",
            key_findings=["Finding B"],
        )

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ) as mock_fn:
            result = await summarize_node(state)

        # Should only pass the 1 item with subtopic_id=2
        call_args = mock_fn.call_args
        assert call_args[0][0] == 2  # subtopic_id
        assert len(call_args[0][2]) == 1  # only 1 content item for sub_q 2
        assert result["subtopic_summaries"][0].subtopic_id == 2
