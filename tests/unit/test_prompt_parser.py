"""Unit tests for research_agent.prompt_parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from research_agent.prompt_parser import (
    ResearchPrompt,
    format_constraints_for_planner,
    load_research_prompt,
    parse_research_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_FULL_PROMPT = """\
# Topic
How does retrieval-augmented generation improve LLM accuracy?

## Constraints
- Focus on peer-reviewed sources from 2023-2025
- Exclude proprietary/closed-source implementations
- Maximum 5 subtopics

## Output Requirements
- Executive summary (200 words max)
- Detailed analysis with citations
- Comparison table of RAG approaches

## Existing Context
Previous research found that naive RAG achieves 60% accuracy on
factual questions, while advanced RAG with re-ranking reaches 85%.
"""

_MINIMAL_PROMPT = """\
# Topic
What is quantum computing?
"""

_NO_HEADINGS = """\
Just a plain text research query about machine learning.
"""

_ALTERNATIVE_HEADINGS = """\
# Research Question
What are the best practices for API rate limiting?

## Limitations
- English sources only
- Published after 2022

## Output Format
- Technical report format
- Include code examples

## Background
The team is building a high-traffic API gateway.
"""


# ---------------------------------------------------------------------------
# TestParseResearchPrompt
# ---------------------------------------------------------------------------


class TestParseResearchPrompt:
    """parse_research_prompt extracts structured fields from Markdown."""

    def test_extracts_topic(self) -> None:
        result = parse_research_prompt(_FULL_PROMPT)
        assert "retrieval-augmented generation" in result.topic

    def test_extracts_constraints(self) -> None:
        result = parse_research_prompt(_FULL_PROMPT)
        assert len(result.constraints) == 3
        assert "peer-reviewed" in result.constraints[0]

    def test_extracts_output_requirements(self) -> None:
        result = parse_research_prompt(_FULL_PROMPT)
        assert len(result.output_requirements) == 3
        assert "Executive summary" in result.output_requirements[0]

    def test_extracts_existing_context(self) -> None:
        result = parse_research_prompt(_FULL_PROMPT)
        assert "naive RAG" in result.existing_context

    def test_preserves_raw_text(self) -> None:
        result = parse_research_prompt(_FULL_PROMPT)
        assert result.raw_text == _FULL_PROMPT

    def test_minimal_prompt(self) -> None:
        result = parse_research_prompt(_MINIMAL_PROMPT)
        assert "quantum computing" in result.topic
        assert result.constraints == []
        assert result.output_requirements == []
        assert result.existing_context == ""

    def test_no_headings_uses_preamble(self) -> None:
        result = parse_research_prompt(_NO_HEADINGS)
        assert "machine learning" in result.topic

    def test_empty_string(self) -> None:
        result = parse_research_prompt("")
        assert result.topic == ""
        assert result.constraints == []

    def test_alternative_heading_names(self) -> None:
        result = parse_research_prompt(_ALTERNATIVE_HEADINGS)
        assert "rate limiting" in result.topic
        assert len(result.constraints) == 2
        assert len(result.output_requirements) == 2
        assert "API gateway" in result.existing_context

    def test_numbered_list_items(self) -> None:
        text = "# Topic\nTest\n\n## Constraints\n1. First\n2. Second\n3. Third\n"
        result = parse_research_prompt(text)
        assert len(result.constraints) == 3
        assert result.constraints[0] == "First"

    def test_mixed_list_markers(self) -> None:
        text = "# Topic\nTest\n\n## Constraints\n- Dash item\n* Star item\n+ Plus item\n"
        result = parse_research_prompt(text)
        assert len(result.constraints) == 3


# ---------------------------------------------------------------------------
# TestLoadResearchPrompt
# ---------------------------------------------------------------------------


class TestLoadResearchPrompt:
    """load_research_prompt reads and parses from file."""

    def test_loads_from_file(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "RESEARCH_PROMPT.md"
        prompt_file.write_text(_FULL_PROMPT)

        result = load_research_prompt(prompt_file)
        assert "retrieval-augmented generation" in result.topic

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_research_prompt(tmp_path / "nonexistent.md")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text(_MINIMAL_PROMPT)

        result = load_research_prompt(str(prompt_file))
        assert "quantum computing" in result.topic


# ---------------------------------------------------------------------------
# TestFormatConstraintsForPlanner
# ---------------------------------------------------------------------------


class TestFormatConstraintsForPlanner:
    """format_constraints_for_planner produces planner-ready text."""

    def test_formats_constraints(self) -> None:
        prompt = ResearchPrompt(
            constraints=["English only", "2023+ sources", "Max 5 topics"]
        )
        result = format_constraints_for_planner(prompt)
        assert "Research constraints:" in result
        assert "- English only" in result
        assert "- 2023+ sources" in result
        assert "- Max 5 topics" in result

    def test_empty_constraints_returns_empty(self) -> None:
        prompt = ResearchPrompt()
        result = format_constraints_for_planner(prompt)
        assert result == ""

    def test_single_constraint(self) -> None:
        prompt = ResearchPrompt(constraints=["Only open-source"])
        result = format_constraints_for_planner(prompt)
        assert result.count("- ") == 1


# ---------------------------------------------------------------------------
# TestResearchPromptDataclass
# ---------------------------------------------------------------------------


class TestResearchPromptDataclass:
    """ResearchPrompt dataclass defaults and structure."""

    def test_default_values(self) -> None:
        prompt = ResearchPrompt()
        assert prompt.topic == ""
        assert prompt.constraints == []
        assert prompt.output_requirements == []
        assert prompt.existing_context == ""
        assert prompt.raw_text == ""

    def test_custom_values(self) -> None:
        prompt = ResearchPrompt(
            topic="Test topic",
            constraints=["c1"],
            output_requirements=["r1"],
            existing_context="Some context",
            raw_text="# Raw",
        )
        assert prompt.topic == "Test topic"
        assert len(prompt.constraints) == 1
