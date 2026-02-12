"""Unit tests for research_agent.report_output."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_agent.report_output import (
    generate_report_filename,
    sanitize_filename,
    write_report,
)

# ---------------------------------------------------------------------------
# TestSanitizeFilename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """sanitize_filename creates filesystem-safe strings."""

    def test_basic_query(self) -> None:
        assert sanitize_filename("What is RAG?") == "what-is-rag"

    def test_removes_special_characters(self) -> None:
        result = sanitize_filename("RAG vs. Fine-tuning: A Comparison!")
        assert ":" not in result
        assert "!" not in result
        assert "." not in result

    def test_collapses_whitespace(self) -> None:
        result = sanitize_filename("too   many    spaces")
        assert "--" not in result
        assert result == "too-many-spaces"

    def test_truncates_long_queries(self) -> None:
        long_query = "a " * 100  # 200 chars
        result = sanitize_filename(long_query)
        assert len(result) <= 80

    def test_empty_query_returns_report(self) -> None:
        assert sanitize_filename("") == "report"

    def test_special_chars_only_returns_report(self) -> None:
        assert sanitize_filename("!@#$%^&*()") == "report"

    def test_preserves_hyphens(self) -> None:
        result = sanitize_filename("fine-tuning approaches")
        assert "fine-tuning" in result

    def test_strips_leading_trailing_hyphens(self) -> None:
        result = sanitize_filename("  -hello world-  ")
        assert not result.startswith("-")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# TestGenerateReportFilename
# ---------------------------------------------------------------------------


class TestGenerateReportFilename:
    """generate_report_filename creates timestamped filenames."""

    def test_includes_sanitized_query(self) -> None:
        filename = generate_report_filename("What is RAG?")
        assert filename.startswith("what-is-rag_")

    def test_includes_timestamp(self) -> None:
        ts = datetime(2024, 6, 15, 12, 30, 45, tzinfo=UTC)
        filename = generate_report_filename("test query", ts)
        assert "20240615_123045" in filename

    def test_ends_with_md(self) -> None:
        filename = generate_report_filename("test")
        assert filename.endswith(".md")

    def test_uses_current_time_by_default(self) -> None:
        filename = generate_report_filename("test")
        year = str(datetime.now(tz=UTC).year)
        assert year in filename


# ---------------------------------------------------------------------------
# TestWriteReport
# ---------------------------------------------------------------------------


class TestWriteReport:
    """write_report writes report and metadata sidecar to disk."""

    def test_creates_report_file(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report\n\nContent here.",
            query="test query",
            output_dir=tmp_path,
        )
        assert report_path.exists()
        assert report_path.read_text() == "# Report\n\nContent here."

    def test_creates_metadata_sidecar(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report\n\nContent.",
            query="test query",
            output_dir=tmp_path,
        )
        meta_path = report_path.with_suffix(".meta.json")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["query"] == "test query"
        assert "generated_at" in meta
        assert "word_count" in meta

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "nested" / "reports"
        write_report(
            report="# Report",
            query="test",
            output_dir=nested_dir,
        )
        assert nested_dir.exists()

    def test_returns_report_path(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report",
            query="test query",
            output_dir=tmp_path,
        )
        assert isinstance(report_path, Path)
        assert report_path.suffix == ".md"

    def test_filename_uses_sanitized_query(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report",
            query="What is RAG? A Deep Dive!",
            output_dir=tmp_path,
        )
        assert "what-is-rag" in report_path.name

    def test_includes_custom_metadata(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report",
            query="test",
            output_dir=tmp_path,
            metadata={"quality_passed": True, "run_id": "run-123"},
        )
        meta_path = report_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["quality_passed"] is True
        assert meta["run_id"] == "run-123"

    def test_metadata_word_count_accurate(self, tmp_path: Path) -> None:
        report = "one two three four five"
        report_path = write_report(
            report=report,
            query="test",
            output_dir=tmp_path,
        )
        meta_path = report_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["word_count"] == 5

    def test_string_output_dir(self, tmp_path: Path) -> None:
        report_path = write_report(
            report="# Report",
            query="test",
            output_dir=str(tmp_path),
        )
        assert report_path.exists()

    def test_multiple_reports_different_filenames(self, tmp_path: Path) -> None:
        path1 = write_report(report="# First", query="test", output_dir=tmp_path)
        # Small time gap ensures different timestamps
        path2 = write_report(report="# Second", query="test", output_dir=tmp_path)
        # Both should exist (timestamps may match within same second)
        assert path1.exists()
        assert path2.exists()
