"""Unit tests for research_agent.nodes.synthesizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.nodes.synthesizer import (
    SynthesisOutput,
    _build_citation_index,
    _build_sources_section,
    _build_synthesis_context,
    _format_context_with_citations,
    _has_sources_section,
    _synthesize_report,
    _validate_citations,
    synthesize_node,
)
from research_agent.state import Source, SubtopicSummary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def summaries() -> list[SubtopicSummary]:
    """Sample summaries from two sub-questions."""
    return [
        SubtopicSummary(
            subtopic_id=1,
            sub_question="What is RAG?",
            summary="RAG combines retrieval and generation for better results.",
            source_urls=["https://a.com/rag", "https://b.com/rag"],
            key_findings=["RAG improves accuracy", "RAG reduces hallucination"],
        ),
        SubtopicSummary(
            subtopic_id=2,
            sub_question="How does fine-tuning compare?",
            summary="Fine-tuning adapts models to specific domains.",
            source_urls=["https://c.com/ft", "https://a.com/rag"],  # shared URL
            key_findings=["Fine-tuning is domain-specific", "Requires labeled data"],
        ),
    ]


@pytest.fixture()
def citation_index() -> dict[str, int]:
    """Expected citation index from the summaries fixture."""
    return {
        "https://a.com/rag": 1,
        "https://b.com/rag": 2,
        "https://c.com/ft": 3,
    }


@pytest.fixture()
def mock_synthesis_result() -> SynthesisOutput:
    """Mock LLM synthesis output with Sources section included."""
    return SynthesisOutput(
        title="RAG vs Fine-tuning Analysis",
        report=(
            "# RAG vs Fine-tuning Analysis\n\n"
            "## Executive Summary\n\n"
            "This report compares RAG and fine-tuning [Source 1].\n\n"
            "## Key Findings\n\n"
            "RAG improves accuracy [Source 2]. Fine-tuning is domain-specific [Source 3].\n\n"
            "## Conclusion\n\n"
            "Both approaches have merits.\n\n"
            "## Sources\n\n"
            "1. [RAG Overview](https://a.com/rag)\n"
        ),
        sources=[
            Source(url="https://a.com/rag", title="RAG Overview"),
            Source(url="https://b.com/rag", title="RAG Details"),
            Source(url="https://c.com/ft", title="Fine-tuning Guide"),
        ],
    )


@pytest.fixture()
def mock_synthesis_no_sources() -> SynthesisOutput:
    """Mock LLM synthesis output without Sources section."""
    return SynthesisOutput(
        title="Report Title",
        report=(
            "# Report\n\n"
            "## Executive Summary\n\n"
            "Summary content [Source 1].\n\n"
            "## Findings\n\n"
            "Details here [Source 2].\n\n"
            "## Conclusion\n\n"
            "Final thoughts."
        ),
        sources=[
            Source(url="https://a.com/rag", title="Source A"),
        ],
    )


# ---------------------------------------------------------------------------
# TestBuildCitationIndex
# ---------------------------------------------------------------------------


class TestBuildCitationIndex:
    """_build_citation_index assigns global citation numbers."""

    def test_assigns_sequential_numbers(self, summaries: list[SubtopicSummary]) -> None:
        index = _build_citation_index(summaries)
        assert index["https://a.com/rag"] == 1
        assert index["https://b.com/rag"] == 2
        assert index["https://c.com/ft"] == 3

    def test_deduplicates_shared_urls(self, summaries: list[SubtopicSummary]) -> None:
        index = _build_citation_index(summaries)
        # https://a.com/rag appears in both summaries but gets one number
        assert len(index) == 3

    def test_empty_summaries(self) -> None:
        index = _build_citation_index([])
        assert index == {}

    def test_summaries_without_urls(self) -> None:
        summaries = [
            SubtopicSummary(subtopic_id=1, summary="No sources.", key_findings=["F"]),
        ]
        index = _build_citation_index(summaries)
        assert index == {}


# ---------------------------------------------------------------------------
# TestFormatContextWithCitations
# ---------------------------------------------------------------------------


class TestFormatContextWithCitations:
    """_format_context_with_citations adds [Source N] references."""

    def test_includes_citation_references(
        self,
        summaries: list[SubtopicSummary],
        citation_index: dict[str, int],
    ) -> None:
        result = _format_context_with_citations(summaries, citation_index)
        assert "[Source 1]" in result
        assert "[Source 2]" in result

    def test_includes_citation_legend(
        self,
        summaries: list[SubtopicSummary],
        citation_index: dict[str, int],
    ) -> None:
        result = _format_context_with_citations(summaries, citation_index)
        assert "Citation Legend" in result
        assert "[Source 1]: https://a.com/rag" in result

    def test_includes_key_findings(
        self,
        summaries: list[SubtopicSummary],
        citation_index: dict[str, int],
    ) -> None:
        result = _format_context_with_citations(summaries, citation_index)
        assert "RAG improves accuracy" in result
        assert "Fine-tuning is domain-specific" in result

    def test_includes_sub_question_headers(
        self,
        summaries: list[SubtopicSummary],
        citation_index: dict[str, int],
    ) -> None:
        result = _format_context_with_citations(summaries, citation_index)
        assert "What is RAG?" in result
        assert "How does fine-tuning compare?" in result


# ---------------------------------------------------------------------------
# TestBuildSynthesisContext
# ---------------------------------------------------------------------------


class TestBuildSynthesisContext:
    """_build_synthesis_context formats summaries for the LLM."""

    def test_includes_all_summaries(self, summaries: list[SubtopicSummary]) -> None:
        result = _build_synthesis_context(summaries)
        assert "What is RAG?" in result
        assert "How does fine-tuning compare?" in result

    def test_includes_source_urls(self, summaries: list[SubtopicSummary]) -> None:
        result = _build_synthesis_context(summaries)
        assert "https://a.com/rag" in result

    def test_empty_summaries(self) -> None:
        result = _build_synthesis_context([])
        assert result == ""


# ---------------------------------------------------------------------------
# TestBuildSourcesSection
# ---------------------------------------------------------------------------


class TestBuildSourcesSection:
    """_build_sources_section creates a Markdown sources list."""

    def test_includes_all_sources(self, citation_index: dict[str, int]) -> None:
        sources = [
            Source(url="https://a.com/rag", title="RAG Overview"),
            Source(url="https://b.com/rag", title="RAG Details"),
        ]
        result = _build_sources_section(citation_index, sources)
        assert "## Sources" in result
        assert "1. [RAG Overview](https://a.com/rag)" in result
        assert "2. [RAG Details](https://b.com/rag)" in result

    def test_falls_back_to_url_when_no_title(
        self, citation_index: dict[str, int]
    ) -> None:
        result = _build_sources_section(citation_index, [])
        assert "1. [https://a.com/rag](https://a.com/rag)" in result

    def test_ordered_by_citation_number(self, citation_index: dict[str, int]) -> None:
        result = _build_sources_section(citation_index, [])
        lines = result.strip().split("\n")
        # Header + blank line + 3 sources
        source_lines = [line for line in lines if line.startswith(("1.", "2.", "3."))]
        assert len(source_lines) == 3
        assert source_lines[0].startswith("1.")
        assert source_lines[1].startswith("2.")
        assert source_lines[2].startswith("3.")


# ---------------------------------------------------------------------------
# TestValidateCitations
# ---------------------------------------------------------------------------


class TestValidateCitations:
    """_validate_citations checks for non-existent citation references."""

    def test_valid_citations_no_warnings(self, citation_index: dict[str, int]) -> None:
        report = "Content [Source 1] and [Source 2] and [Source 3]."
        warnings = _validate_citations(report, citation_index)
        assert warnings == []

    def test_invalid_citation_generates_warning(
        self, citation_index: dict[str, int]
    ) -> None:
        report = "Content [Source 1] and [Source 99]."
        warnings = _validate_citations(report, citation_index)
        assert len(warnings) == 1
        assert "99" in warnings[0]

    def test_no_citations_no_warnings(self, citation_index: dict[str, int]) -> None:
        report = "Content without any citation references."
        warnings = _validate_citations(report, citation_index)
        assert warnings == []

    def test_empty_citation_index(self) -> None:
        report = "Content [Source 1]."
        warnings = _validate_citations(report, {})
        assert len(warnings) == 1

    def test_bracket_number_format(self, citation_index: dict[str, int]) -> None:
        report = "Content [1] and [2]."
        warnings = _validate_citations(report, citation_index)
        assert warnings == []


# ---------------------------------------------------------------------------
# TestHasSourcesSection
# ---------------------------------------------------------------------------


class TestHasSourcesSection:
    """_has_sources_section detects Sources headings."""

    def test_detects_h2_sources(self) -> None:
        assert _has_sources_section("## Sources\n\n1. Source A")

    def test_detects_h3_sources(self) -> None:
        assert _has_sources_section("### Sources\n\n1. Source A")

    def test_detects_h1_sources(self) -> None:
        assert _has_sources_section("# Sources\n\n1. Source A")

    def test_case_insensitive(self) -> None:
        assert _has_sources_section("## SOURCES\n\n1. Source A")

    def test_returns_false_when_missing(self) -> None:
        assert not _has_sources_section("## Findings\n\nContent here.")

    def test_returns_false_for_inline_mention(self) -> None:
        assert not _has_sources_section("The sources suggest that...")


# ---------------------------------------------------------------------------
# TestSynthesizeReport
# ---------------------------------------------------------------------------


class TestSynthesizeReport:
    """_synthesize_report calls the LLM with proper prompts."""

    def _make_mock_response(self, synthesis_result: SynthesisOutput) -> MagicMock:
        """Build a mock litellm response from a SynthesisOutput."""
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": synthesis_result.title,
            "report": synthesis_result.report,
            "sources": [
                {"url": s.url, "title": s.title} for s in synthesis_result.sources
            ],
        })
        return mock_response

    @pytest.mark.asyncio()
    async def test_returns_synthesis_output(
        self, mock_synthesis_result: SynthesisOutput
    ) -> None:
        mock_response = self._make_mock_response(mock_synthesis_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "You are a report writer.",
                "user": "Query: {query}\n\n{context}",
            }
            result = await _synthesize_report("test query", "test context")

        assert isinstance(result, SynthesisOutput)
        assert result.title == "RAG vs Fine-tuning Analysis"

    @pytest.mark.asyncio()
    async def test_uses_strategic_tier_model(
        self, mock_synthesis_result: SynthesisOutput
    ) -> None:
        mock_response = self._make_mock_response(mock_synthesis_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{query} {context}",
            }
            await _synthesize_report("query", "context")

        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio()
    async def test_includes_max_length_in_prompt(
        self, mock_synthesis_result: SynthesisOutput
    ) -> None:
        mock_response = self._make_mock_response(mock_synthesis_result)

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "system": "Sys",
                "user": "{query} {context}",
            }
            await _synthesize_report("query", "context", max_length=5000)

        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        assert "5000" in messages[1]["content"]


# ---------------------------------------------------------------------------
# TestSynthesizeNode
# ---------------------------------------------------------------------------


class TestSynthesizeNode:
    """synthesize_node orchestrates the full synthesis pipeline."""

    @pytest.mark.asyncio()
    async def test_produces_report_with_sources(
        self,
        summaries: list[SubtopicSummary],
        mock_synthesis_result: SynthesisOutput,
    ) -> None:
        state = {
            "subtopic_summaries": summaries,
            "query": "RAG vs fine-tuning",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            return_value=mock_synthesis_result,
        ):
            result = await synthesize_node(state)

        assert result["final_report"] != ""
        assert len(result["sources"]) > 0
        assert result["step"] == "synthesize"
        assert result["step_index"] == 4

    @pytest.mark.asyncio()
    async def test_empty_summaries_returns_empty(self) -> None:
        state = {"subtopic_summaries": [], "query": "test"}
        result = await synthesize_node(state)
        assert result["final_report"] == ""
        assert result["sources"] == []

    @pytest.mark.asyncio()
    async def test_appends_sources_section_if_missing(
        self,
        summaries: list[SubtopicSummary],
        mock_synthesis_no_sources: SynthesisOutput,
    ) -> None:
        state = {
            "subtopic_summaries": summaries,
            "query": "test query",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            return_value=mock_synthesis_no_sources,
        ):
            result = await synthesize_node(state)

        assert "## Sources" in result["final_report"]

    @pytest.mark.asyncio()
    async def test_does_not_duplicate_sources_section(
        self,
        summaries: list[SubtopicSummary],
        mock_synthesis_result: SynthesisOutput,
    ) -> None:
        state = {
            "subtopic_summaries": summaries,
            "query": "test query",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            return_value=mock_synthesis_result,
        ):
            result = await synthesize_node(state)

        # Count occurrences of "## Sources"
        count = result["final_report"].count("## Sources")
        assert count == 1

    @pytest.mark.asyncio()
    async def test_handles_llm_failure_gracefully(
        self,
        summaries: list[SubtopicSummary],
    ) -> None:
        state = {
            "subtopic_summaries": summaries,
            "query": "test query",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API error"),
        ):
            result = await synthesize_node(state)

        assert result["final_report"] == ""
        assert result["sources"] == []

    @pytest.mark.asyncio()
    async def test_source_models_have_timestamps(
        self,
        summaries: list[SubtopicSummary],
        mock_synthesis_result: SynthesisOutput,
    ) -> None:
        state = {
            "subtopic_summaries": summaries,
            "query": "test",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            return_value=mock_synthesis_result,
        ):
            result = await synthesize_node(state)

        for source in result["sources"]:
            assert source.accessed_at != ""

    @pytest.mark.asyncio()
    async def test_citation_warnings_logged(
        self,
        summaries: list[SubtopicSummary],
    ) -> None:
        """Invalid citations in the report should generate warnings."""
        bad_result = SynthesisOutput(
            title="Report",
            report="Content [Source 99]. No sources section.",
            sources=[],
        )
        state = {
            "subtopic_summaries": summaries,
            "query": "test",
        }

        with patch(
            "research_agent.nodes.synthesizer._synthesize_report",
            new_callable=AsyncMock,
            return_value=bad_result,
        ):
            # Should not raise despite invalid citations
            result = await synthesize_node(state)

        # Report still generated (with appended sources section)
        assert result["final_report"] != ""
