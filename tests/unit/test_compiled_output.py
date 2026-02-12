"""Unit tests for research_agent.compiled_output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from research_agent.compiled_output import (
    format_compiled_research,
    write_compiled_research,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_SAMPLE_REPORT = """\
# Research on RAG Systems

## Executive Summary

RAG systems combine retrieval with generation for improved accuracy.

## Key Findings

- Finding 1: RAG improves factual accuracy by 25%
- Finding 2: Re-ranking further boosts precision

## Detailed Analysis

RAG pipelines consist of three main stages: retrieval, augmentation,
and generation. Each stage presents unique optimization opportunities.

## Technical Considerations

Vector databases like ChromaDB and Pinecone are common choices.
Embedding model selection significantly impacts retrieval quality.

## Sources

1. Smith et al. (2024) - RAG Survey
2. Johnson (2023) - Vector DB Comparison
"""

_MINIMAL_REPORT = """\
# Simple Report

This is a basic report without structured sections.
"""


# ---------------------------------------------------------------------------
# TestFormatCompiledResearch
# ---------------------------------------------------------------------------


class TestFormatCompiledResearch:
    """format_compiled_research restructures reports into compiled format."""

    def test_includes_query_as_title(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "RAG systems analysis")
        assert "# RAG systems analysis" in result

    def test_includes_timestamp(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "*Compiled:" in result
        assert "UTC*" in result

    def test_includes_executive_summary(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Executive Summary" in result
        assert "retrieval with generation" in result

    def test_includes_key_findings(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Key Findings" in result
        assert "factual accuracy" in result

    def test_includes_detailed_analysis(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Detailed Analysis" in result
        assert "three main stages" in result

    def test_includes_technical_considerations(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Technical Considerations" in result
        assert "ChromaDB" in result

    def test_includes_sources(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Sources" in result
        assert "Smith et al." in result

    def test_includes_methodology(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert "## Methodology" in result
        assert "multi-stage pipeline" in result

    def test_includes_cost_in_methodology(self) -> None:
        meta = {"cost_so_far": 0.0523, "llm_call_count": 12}
        result = format_compiled_research(_SAMPLE_REPORT, "test", metadata=meta)
        assert "$0.0523" in result
        assert "LLM calls: 12" in result

    def test_minimal_report_uses_defaults(self) -> None:
        result = format_compiled_research(_MINIMAL_REPORT, "test")
        assert "## Executive Summary" in result
        # The intro content should appear somewhere
        assert "basic report" in result

    def test_empty_report(self) -> None:
        result = format_compiled_research("", "test query")
        assert "# test query" in result
        assert "## Executive Summary" in result
        # Default fallback messages
        assert "No key findings extracted." in result

    def test_all_section_separators_present(self) -> None:
        result = format_compiled_research(_SAMPLE_REPORT, "test")
        assert result.count("---") >= 5


# ---------------------------------------------------------------------------
# TestWriteCompiledResearch
# ---------------------------------------------------------------------------


class TestWriteCompiledResearch:
    """write_compiled_research writes files to disk."""

    def test_creates_compiled_file(self, tmp_path: Path) -> None:
        path = write_compiled_research(_SAMPLE_REPORT, "RAG test", tmp_path)
        assert path.exists()
        assert "COMPILED_RESEARCH" in path.name
        assert path.suffix == ".md"

    def test_file_contains_compiled_content(self, tmp_path: Path) -> None:
        path = write_compiled_research(_SAMPLE_REPORT, "RAG test", tmp_path)
        content = path.read_text()
        assert "# RAG test" in content
        assert "## Executive Summary" in content

    def test_creates_metadata_sidecar(self, tmp_path: Path) -> None:
        path = write_compiled_research(_SAMPLE_REPORT, "RAG test", tmp_path)
        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["query"] == "RAG test"
        assert meta["format"] == "compiled_research"
        assert "compiled_at" in meta
        assert "word_count" in meta

    def test_includes_custom_metadata_in_sidecar(self, tmp_path: Path) -> None:
        path = write_compiled_research(
            _SAMPLE_REPORT,
            "test",
            tmp_path,
            metadata={"run_id": "run-123", "quality_score": 0.85},
        )
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "run-123"
        assert meta["quality_score"] == 0.85

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "compiled"
        write_compiled_research(_SAMPLE_REPORT, "test", nested)
        assert nested.exists()

    def test_sanitized_filename(self, tmp_path: Path) -> None:
        path = write_compiled_research(
            _SAMPLE_REPORT,
            "What is RAG? A Deep Dive!",
            tmp_path,
        )
        assert "what-is-rag" in path.name
        assert "COMPILED_RESEARCH" in path.name

    def test_string_output_dir(self, tmp_path: Path) -> None:
        path = write_compiled_research(_SAMPLE_REPORT, "test", str(tmp_path))
        assert path.exists()
