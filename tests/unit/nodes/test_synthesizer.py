"""Unit tests for research_agent.nodes.synthesizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.nodes.synthesizer import (
    ExecutiveSummaryOutput,
    SectionOutput,
    SynthesisOutput,
    _build_citation_index,
    _build_sources_section,
    _build_synthesis_context,
    _format_context_with_citations,
    _has_sources_section,
    _synthesize_executive_summary,
    _synthesize_report,
    _synthesize_section,
    _synthesize_serial,
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


# ---------------------------------------------------------------------------
# Serial synthesis model tests
# ---------------------------------------------------------------------------


class TestSectionOutput:
    """SectionOutput model validation."""

    def test_valid_construction(self) -> None:
        section = SectionOutput(section_title="Overview", section_body="Content here.")
        assert section.section_title == "Overview"
        assert section.section_body == "Content here."

    def test_missing_fields_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SectionOutput()  # type: ignore[call-arg]


class TestExecutiveSummaryOutput:
    """ExecutiveSummaryOutput model validation."""

    def test_valid_construction(self) -> None:
        output = ExecutiveSummaryOutput(
            executive_summary="Summary text.",
            introduction="Intro paragraph.",
            conclusion="Conclusion text.",
            title="Report Title",
        )
        assert output.title == "Report Title"
        assert output.executive_summary == "Summary text."

    def test_missing_fields_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExecutiveSummaryOutput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestSynthesizeSection
# ---------------------------------------------------------------------------


class TestSynthesizeSection:
    """_synthesize_section generates one report section via LLM."""

    def _make_section_response(self, title: str, body: str) -> MagicMock:
        """Build a mock litellm response for a section."""
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "section_title": title,
            "section_body": body,
        })
        return mock_response

    @pytest.mark.asyncio()
    async def test_returns_section_output(self) -> None:
        summary = SubtopicSummary(
            subtopic_id=1,
            sub_question="What is RAG?",
            summary="RAG combines retrieval and generation.",
            source_urls=["https://a.com/rag"],
            key_findings=["Improves accuracy"],
        )
        citation_index = {"https://a.com/rag": 1}
        mock_response = self._make_section_response("RAG Overview", "RAG is great [Source 1].")

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "section_system": "Write a section.",
                "section_user": "{query} {section_number} {total_sections} {subtopic_question} {section_context} {prior_context}",
            }
            result = await _synthesize_section(
                query="RAG research",
                summary=summary,
                citation_index=citation_index,
                section_number=1,
                total_sections=3,
                prior_sections=[],
            )

        assert isinstance(result, SectionOutput)
        assert result.section_title == "RAG Overview"
        assert "[Source 1]" in result.section_body

    @pytest.mark.asyncio()
    async def test_includes_prior_context(self) -> None:
        summary = SubtopicSummary(
            subtopic_id=2,
            sub_question="Fine-tuning?",
            summary="Fine-tuning adapts models.",
            key_findings=["Domain-specific"],
        )
        mock_response = self._make_section_response("Fine-tuning", "Details here.")

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "section_system": "Write a section.",
                "section_user": "{query} {section_number} {total_sections} {subtopic_question} {section_context} {prior_context}",
            }
            await _synthesize_section(
                query="topic",
                summary=summary,
                citation_index={},
                section_number=2,
                total_sections=3,
                prior_sections=["- Section 1: RAG Overview"],
            )

        call_kwargs = mock_call.call_args[1]
        user_message = call_kwargs["messages"][1]["content"]
        assert "Section 1: RAG Overview" in user_message

    @pytest.mark.asyncio()
    async def test_uses_subtopic_id_when_no_sub_question(self) -> None:
        summary = SubtopicSummary(
            subtopic_id=3,
            summary="Content.",
            key_findings=["Finding"],
        )
        mock_response = self._make_section_response("Section 3", "Body.")

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "section_system": "Sys.",
                "section_user": "{query} {section_number} {total_sections} {subtopic_question} {section_context} {prior_context}",
            }
            await _synthesize_section(
                query="topic",
                summary=summary,
                citation_index={},
                section_number=3,
                total_sections=5,
                prior_sections=[],
            )

        user_message = mock_call.call_args[1]["messages"][1]["content"]
        assert "Subtopic 3" in user_message


# ---------------------------------------------------------------------------
# TestSynthesizeExecutiveSummary
# ---------------------------------------------------------------------------


class TestSynthesizeExecutiveSummary:
    """_synthesize_executive_summary generates framing content."""

    def _make_exec_response(self) -> MagicMock:
        import json

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "executive_summary": "This report covers RAG and fine-tuning.",
            "introduction": "Background on modern NLP techniques.",
            "conclusion": "Both approaches have merits. Key takeaways: ...",
            "title": "RAG vs Fine-tuning Analysis",
        })
        return mock_response

    @pytest.mark.asyncio()
    async def test_returns_executive_summary_output(self) -> None:
        mock_response = self._make_exec_response()

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "executive_summary_system": "Write exec summary.",
                "executive_summary_user": "{query} {sections_text} {citation_legend}",
            }
            result = await _synthesize_executive_summary(
                query="RAG vs fine-tuning",
                sections_text="## Section 1\n\nContent here.",
                citation_legend="[Source 1]: https://a.com",
            )

        assert isinstance(result, ExecutiveSummaryOutput)
        assert result.title == "RAG vs Fine-tuning Analysis"
        assert "RAG" in result.executive_summary

    @pytest.mark.asyncio()
    async def test_passes_citation_legend(self) -> None:
        mock_response = self._make_exec_response()

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_call,
            patch("research_agent.nodes.synthesizer._load_prompt") as mock_prompt,
        ):
            mock_prompt.return_value = {
                "executive_summary_system": "Sys.",
                "executive_summary_user": "{query} {sections_text} {citation_legend}",
            }
            await _synthesize_executive_summary(
                query="topic",
                sections_text="sections",
                citation_legend="[Source 1]: https://a.com",
            )

        user_message = mock_call.call_args[1]["messages"][1]["content"]
        assert "[Source 1]" in user_message


# ---------------------------------------------------------------------------
# TestSynthesizeSerial
# ---------------------------------------------------------------------------


class TestSynthesizeSerial:
    """_synthesize_serial orchestrates section-by-section generation."""

    @pytest.mark.asyncio()
    async def test_generates_all_sections(self, summaries: list[SubtopicSummary]) -> None:
        citation_index = {"https://a.com/rag": 1, "https://b.com/rag": 2, "https://c.com/ft": 3}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_section",
                new_callable=AsyncMock,
            ) as mock_section,
            patch(
                "research_agent.nodes.synthesizer._synthesize_executive_summary",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            mock_section.side_effect = [
                SectionOutput(section_title="RAG Overview", section_body="RAG content [Source 1]."),
                SectionOutput(section_title="Fine-tuning", section_body="FT content [Source 3]."),
            ]
            mock_exec.return_value = ExecutiveSummaryOutput(
                executive_summary="Summary.",
                introduction="Intro.",
                conclusion="Conclusion.",
                title="Full Report",
            )

            result = await _synthesize_serial("topic", summaries, citation_index)

        assert isinstance(result, SynthesisOutput)
        assert mock_section.call_count == 2
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio()
    async def test_report_contains_all_sections(self, summaries: list[SubtopicSummary]) -> None:
        citation_index = {"https://a.com/rag": 1}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_section",
                new_callable=AsyncMock,
            ) as mock_section,
            patch(
                "research_agent.nodes.synthesizer._synthesize_executive_summary",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            mock_section.side_effect = [
                SectionOutput(section_title="Section A", section_body="Body A."),
                SectionOutput(section_title="Section B", section_body="Body B."),
            ]
            mock_exec.return_value = ExecutiveSummaryOutput(
                executive_summary="Exec summary.",
                introduction="Background.",
                conclusion="Key takeaways.",
                title="Report Title",
            )

            result = await _synthesize_serial("topic", summaries, citation_index)

        assert "## Section A" in result.report
        assert "## Section B" in result.report
        assert "Body A." in result.report
        assert "Body B." in result.report

    @pytest.mark.asyncio()
    async def test_report_has_executive_summary_and_conclusion(
        self, summaries: list[SubtopicSummary]
    ) -> None:
        citation_index = {}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_section",
                new_callable=AsyncMock,
            ) as mock_section,
            patch(
                "research_agent.nodes.synthesizer._synthesize_executive_summary",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            mock_section.side_effect = [
                SectionOutput(section_title="S1", section_body="B1."),
                SectionOutput(section_title="S2", section_body="B2."),
            ]
            mock_exec.return_value = ExecutiveSummaryOutput(
                executive_summary="The exec summary.",
                introduction="The introduction.",
                conclusion="The conclusion.",
                title="Title",
            )

            result = await _synthesize_serial("topic", summaries, citation_index)

        assert "## Executive Summary" in result.report
        assert "The exec summary." in result.report
        assert "## Introduction" in result.report
        assert "The introduction." in result.report
        assert "## Conclusion" in result.report
        assert "The conclusion." in result.report

    @pytest.mark.asyncio()
    async def test_prior_sections_accumulate(self, summaries: list[SubtopicSummary]) -> None:
        citation_index = {}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_section",
                new_callable=AsyncMock,
            ) as mock_section,
            patch(
                "research_agent.nodes.synthesizer._synthesize_executive_summary",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            mock_section.side_effect = [
                SectionOutput(section_title="First", section_body="B1."),
                SectionOutput(section_title="Second", section_body="B2."),
            ]
            mock_exec.return_value = ExecutiveSummaryOutput(
                executive_summary="S.", introduction="I.", conclusion="C.", title="T",
            )

            await _synthesize_serial("topic", summaries, citation_index)

        # First call: no prior sections
        first_call = mock_section.call_args_list[0]
        assert first_call.kwargs["prior_sections"] == []

        # Second call: first section in prior context
        second_call = mock_section.call_args_list[1]
        assert len(second_call.kwargs["prior_sections"]) == 1
        assert "First" in second_call.kwargs["prior_sections"][0]

    @pytest.mark.asyncio()
    async def test_title_from_executive_summary(self, summaries: list[SubtopicSummary]) -> None:
        citation_index = {}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_section",
                new_callable=AsyncMock,
            ) as mock_section,
            patch(
                "research_agent.nodes.synthesizer._synthesize_executive_summary",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            mock_section.side_effect = [
                SectionOutput(section_title="S1", section_body="B1."),
                SectionOutput(section_title="S2", section_body="B2."),
            ]
            mock_exec.return_value = ExecutiveSummaryOutput(
                executive_summary="S.", introduction="I.", conclusion="C.",
                title="Custom Title",
            )

            result = await _synthesize_serial("topic", summaries, citation_index)

        assert result.title == "Custom Title"
        assert "# Custom Title" in result.report


# ---------------------------------------------------------------------------
# TestSynthesizeNodeSerial
# ---------------------------------------------------------------------------


class TestSynthesizeNodeSerial:
    """synthesize_node uses serial mode when subtopic count exceeds threshold."""

    @pytest.fixture()
    def many_summaries(self) -> list[SubtopicSummary]:
        """4 summaries to trigger serial mode (threshold=3)."""
        return [
            SubtopicSummary(
                subtopic_id=i,
                sub_question=f"Question {i}?",
                summary=f"Summary for question {i}.",
                source_urls=[f"https://example.com/{i}"],
                key_findings=[f"Finding {i}"],
            )
            for i in range(1, 5)
        ]

    @pytest.mark.asyncio()
    async def test_uses_serial_when_above_threshold(
        self, many_summaries: list[SubtopicSummary]
    ) -> None:
        state = {"subtopic_summaries": many_summaries, "query": "test"}
        serial_result = SynthesisOutput(
            title="Serial Report",
            report="# Serial Report\n\n## Sources\n\n1. Source",
            sources=[],
        )

        with patch(
            "research_agent.nodes.synthesizer._synthesize_serial",
            new_callable=AsyncMock,
            return_value=serial_result,
        ) as mock_serial:
            result = await synthesize_node(state, serial_threshold=3)

        mock_serial.assert_called_once()
        assert result["final_report"] != ""

    @pytest.mark.asyncio()
    async def test_uses_single_pass_at_or_below_threshold(
        self, summaries: list[SubtopicSummary], mock_synthesis_result: SynthesisOutput
    ) -> None:
        state = {"subtopic_summaries": summaries, "query": "test"}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_report",
                new_callable=AsyncMock,
                return_value=mock_synthesis_result,
            ) as mock_single,
            patch(
                "research_agent.nodes.synthesizer._synthesize_serial",
                new_callable=AsyncMock,
            ) as mock_serial,
        ):
            await synthesize_node(state, serial_threshold=3)

        mock_single.assert_called_once()
        mock_serial.assert_not_called()

    @pytest.mark.asyncio()
    async def test_serial_fallback_to_single_pass(
        self, many_summaries: list[SubtopicSummary], mock_synthesis_result: SynthesisOutput
    ) -> None:
        """If serial mode fails, falls back to single-pass synthesis."""
        state = {"subtopic_summaries": many_summaries, "query": "test"}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_serial",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Section generation failed"),
            ),
            patch(
                "research_agent.nodes.synthesizer._synthesize_report",
                new_callable=AsyncMock,
                return_value=mock_synthesis_result,
            ) as mock_single,
        ):
            result = await synthesize_node(state, serial_threshold=3)

        mock_single.assert_called_once()
        assert result["final_report"] != ""

    @pytest.mark.asyncio()
    async def test_serial_and_single_pass_both_fail(
        self, many_summaries: list[SubtopicSummary]
    ) -> None:
        """If both serial and single-pass fail, returns empty report."""
        state = {"subtopic_summaries": many_summaries, "query": "test"}

        with (
            patch(
                "research_agent.nodes.synthesizer._synthesize_serial",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Serial failed"),
            ),
            patch(
                "research_agent.nodes.synthesizer._synthesize_report",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Single-pass also failed"),
            ),
        ):
            result = await synthesize_node(state, serial_threshold=3)

        assert result["final_report"] == ""
        assert result["sources"] == []

    @pytest.mark.asyncio()
    async def test_custom_threshold(self, summaries: list[SubtopicSummary]) -> None:
        """Setting threshold=1 triggers serial even with 2 summaries."""
        state = {"subtopic_summaries": summaries, "query": "test"}
        serial_result = SynthesisOutput(
            title="Serial Report",
            report="# Serial Report\n\n## Sources\n\n1. Source",
            sources=[],
        )

        with patch(
            "research_agent.nodes.synthesizer._synthesize_serial",
            new_callable=AsyncMock,
            return_value=serial_result,
        ) as mock_serial:
            await synthesize_node(state, serial_threshold=1)

        mock_serial.assert_called_once()
