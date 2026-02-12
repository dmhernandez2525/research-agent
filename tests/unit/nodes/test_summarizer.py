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
from research_agent.state import ScrapedContent, Summary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scraped_items() -> list[ScrapedContent]:
    """Multiple scraped content items across two sub-questions."""
    return [
        ScrapedContent(
            url="https://a.com/1",
            sub_question_id=1,
            title="Article A",
            content="Content about topic A with many details and findings.",
            word_count=50,
            quality_score=0.8,
        ),
        ScrapedContent(
            url="https://b.com/2",
            sub_question_id=1,
            title="Article B",
            content="More content about topic A from a different source.",
            word_count=40,
            quality_score=0.7,
        ),
        ScrapedContent(
            url="https://c.com/3",
            sub_question_id=2,
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
def sample_sub_questions() -> list[dict]:
    """Sub-questions as dicts (as stored in state)."""
    return [
        {"id": 1, "question": "What is topic A?", "rationale": "Core question"},
        {"id": 2, "question": "What is topic B?", "rationale": "Supporting question"},
    ]


# ---------------------------------------------------------------------------
# TestGroupContentByQuestion
# ---------------------------------------------------------------------------


class TestGroupContentByQuestion:
    """_group_content_by_question groups scraped content by sub_question_id."""

    def test_groups_by_sub_question_id(
        self, scraped_items: list[ScrapedContent]
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
            ScrapedContent(
                url="https://a.com", sub_question_id=1, content="text", word_count=5
            ),
            ScrapedContent(
                url="https://b.com", sub_question_id=1, content="more", word_count=5
            ),
        ]
        groups = _group_content_by_question(items)
        assert list(groups.keys()) == [1]
        assert len(groups[1]) == 2

    def test_preserves_item_order(self, scraped_items: list[ScrapedContent]) -> None:
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
            ScrapedContent(
                url="https://example.com",
                sub_question_id=1,
                title="Test Article",
                content="Article content here.",
                word_count=3,
            ),
        ]
        result = _build_content_block(items)
        assert "Source: Test Article (https://example.com)" in result
        assert "Article content here." in result

    def test_multiple_items_separated_by_rule(
        self, scraped_items: list[ScrapedContent]
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
            ScrapedContent(
                url="https://x.com",
                sub_question_id=1,
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

    @pytest.mark.asyncio()
    async def test_returns_summary_model(
        self,
        scraped_items: list[ScrapedContent],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_summarizer_result)

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_model),
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "You are a summarizer.",
                "user": "Sub-question: {sub_question}\nContent from {num_sources} sources:\n{content}",
            }
            result = await _summarize_group(1, "What is topic A?", scraped_items[:2])

        assert isinstance(result, Summary)
        assert result.sub_question_id == 1
        assert result.sub_question == "What is topic A?"
        assert result.summary == mock_summarizer_result.summary
        assert result.key_findings == mock_summarizer_result.key_findings

    @pytest.mark.asyncio()
    async def test_extracts_source_urls(
        self,
        scraped_items: list[ScrapedContent],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_summarizer_result)

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_model),
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
        scraped_items: list[ScrapedContent],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_summarizer_result)

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_model),
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{sub_question} {num_sources} {content}",
            }
            await _summarize_group(1, "What is topic A?", scraped_items[:2])

        call_args = mock_structured.ainvoke.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
        assert "What is topic A?" in call_args[1]["content"]

    @pytest.mark.asyncio()
    async def test_uses_smart_tier_model(
        self,
        scraped_items: list[ScrapedContent],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_summarizer_result)

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured

        with (
            patch(
                "langchain_anthropic.ChatAnthropic", return_value=mock_model
            ) as mock_cls,
            patch("research_agent.nodes.summarizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{sub_question} {num_sources} {content}",
            }
            await _summarize_group(1, "Question?", scraped_items[:1])

        call_kwargs = mock_cls.call_args[1]
        assert "sonnet" in call_kwargs["model"]


# ---------------------------------------------------------------------------
# TestSummarizeNode
# ---------------------------------------------------------------------------


class TestSummarizeNode:
    """summarize_node orchestrates per-subtopic summarization."""

    @pytest.mark.asyncio()
    async def test_returns_summary_for_current_subtopic(
        self,
        scraped_items: list[ScrapedContent],
        sample_sub_questions: list[dict],
        mock_summarizer_result: SummarizerOutput,
    ) -> None:
        state = {
            "scraped_content": scraped_items,
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 0,
        }

        mock_summary = Summary(
            sub_question_id=1,
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

        assert len(result["summaries"]) == 1
        assert result["summaries"][0].sub_question_id == 1

    @pytest.mark.asyncio()
    async def test_increments_subtopic_index(
        self,
        scraped_items: list[ScrapedContent],
        sample_sub_questions: list[dict],
    ) -> None:
        state = {
            "scraped_content": scraped_items,
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 0,
        }

        mock_summary = Summary(
            sub_question_id=1,
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
    async def test_empty_scraped_content_returns_empty(
        self,
        sample_sub_questions: list[dict],
    ) -> None:
        state = {
            "scraped_content": [],
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 0,
        }
        result = await summarize_node(state)
        assert result["summaries"] == []
        assert result["current_subtopic_index"] == 1

    @pytest.mark.asyncio()
    async def test_no_sub_questions_returns_empty(self) -> None:
        state = {
            "scraped_content": [],
            "sub_questions": [],
            "current_subtopic_index": 0,
        }
        result = await summarize_node(state)
        assert result["summaries"] == []

    @pytest.mark.asyncio()
    async def test_index_out_of_range_returns_empty(
        self,
        sample_sub_questions: list[dict],
    ) -> None:
        state = {
            "scraped_content": [],
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 10,
        }
        result = await summarize_node(state)
        assert result["summaries"] == []

    @pytest.mark.asyncio()
    async def test_sets_step_metadata(
        self,
        scraped_items: list[ScrapedContent],
        sample_sub_questions: list[dict],
    ) -> None:
        state = {
            "scraped_content": scraped_items,
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 0,
        }

        mock_summary = Summary(
            sub_question_id=1,
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
        scraped_items: list[ScrapedContent],
        sample_sub_questions: list[dict],
    ) -> None:
        state = {
            "scraped_content": scraped_items,
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 0,
        }

        with patch(
            "research_agent.nodes.summarizer._summarize_group",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM API error"),
        ):
            result = await summarize_node(state)

        assert result["summaries"] == []
        assert result["current_subtopic_index"] == 1

    @pytest.mark.asyncio()
    async def test_filters_content_for_current_subtopic_only(
        self,
        scraped_items: list[ScrapedContent],
        sample_sub_questions: list[dict],
    ) -> None:
        """Only content matching the current sub_question_id is passed to _summarize_group."""
        state = {
            "scraped_content": scraped_items,
            "sub_questions": sample_sub_questions,
            "current_subtopic_index": 1,  # sub_question_id=2
        }

        mock_summary = Summary(
            sub_question_id=2,
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

        # Should only pass the 1 item with sub_question_id=2
        call_args = mock_fn.call_args
        assert call_args[0][0] == 2  # sub_question_id
        assert len(call_args[0][2]) == 1  # only 1 content item for sub_q 2
        assert result["summaries"][0].sub_question_id == 2
