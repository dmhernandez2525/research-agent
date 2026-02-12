"""Integration tests for report output + structured logging provenance chain."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
import structlog

from research_agent.logging import (
    configure_logging,
    log_provenance,
    step_logging_context,
)
from research_agent.report_output import (
    generate_report_filename,
    sanitize_filename,
    write_report,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog state between tests."""
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Report output + logging integration
# ---------------------------------------------------------------------------


class TestReportWithLogging:
    """Report writing integrates with structured logging."""

    def test_write_report_creates_file_and_sidecar(self, tmp_path: Path) -> None:
        """write_report creates both .md and .meta.json files."""
        configure_logging(level="DEBUG", fmt="json")

        report_text = "# Research Report\n\nFindings about RAG."
        path = write_report(
            report=report_text,
            query="What is RAG?",
            output_dir=tmp_path,
        )

        assert path.exists()
        assert path.suffix == ".md"
        assert path.read_text() == report_text

        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["query"] == "What is RAG?"
        assert "word_count" in meta
        assert "generated_at" in meta

    def test_report_metadata_includes_custom_fields(self, tmp_path: Path) -> None:
        """Custom metadata is included in the sidecar file."""
        path = write_report(
            report="# Report\nContent.",
            query="RAG overview",
            output_dir=tmp_path,
            metadata={
                "total_cost_usd": 0.15,
                "total_sources": 12,
                "evaluation_score": 4.2,
            },
        )

        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["total_cost_usd"] == 0.15
        assert meta["total_sources"] == 12
        assert meta["evaluation_score"] == 4.2

    def test_report_directory_created_automatically(self, tmp_path: Path) -> None:
        """write_report creates the output directory if it doesn't exist."""
        nested_dir = tmp_path / "reports" / "2024" / "01"
        path = write_report(
            report="Content.",
            query="nested test",
            output_dir=nested_dir,
        )
        assert path.exists()
        assert nested_dir.exists()


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------


class TestFilenameSanitization:
    """Filenames derived from queries are safe for all filesystems."""

    def test_special_characters_removed(self) -> None:
        sanitized = sanitize_filename("What is RAG? (2024)")
        assert "?" not in sanitized
        assert "(" not in sanitized
        assert ")" not in sanitized

    def test_spaces_become_hyphens(self) -> None:
        sanitized = sanitize_filename("What is RAG")
        assert " " not in sanitized
        assert "-" in sanitized

    def test_long_queries_truncated(self) -> None:
        long_query = "a" * 200
        sanitized = sanitize_filename(long_query)
        assert len(sanitized) <= 80

    def test_empty_query_returns_default(self) -> None:
        assert sanitize_filename("") == "report"
        assert sanitize_filename("   ") == "report"

    def test_filename_includes_timestamp(self) -> None:
        filename = generate_report_filename("RAG overview")
        assert filename.endswith(".md")
        # Should contain date-like pattern (YYYYMMDD)
        assert "_20" in filename


# ---------------------------------------------------------------------------
# Provenance logging during report generation
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """Provenance logging works within step contexts for audit trails."""

    def test_provenance_logged_within_step_context(self, tmp_path: Path) -> None:
        """Provenance entries within a step context include session and step info."""
        log_file = tmp_path / "provenance.log"
        configure_logging(
            level="INFO",
            fmt="json",
            log_file=str(log_file),
            session_id="test-session-001",
        )

        with step_logging_context("scrape", step_index=2):
            log_provenance(
                source_url="https://example.com/article",
                action="scraped",
                step_name="scrape",
                details={"status_code": 200, "word_count": 500},
            )

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "provenance_entry" in content
        assert "https://example.com/article" in content
        assert "test-session-001" in content

    def test_multiple_provenance_entries_in_sequence(self, tmp_path: Path) -> None:
        """Multiple provenance entries accumulate in the log file."""
        log_file = tmp_path / "multi.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log_provenance(source_url="https://a.com", action="scraped")
        log_provenance(source_url="https://b.com", action="embedded")
        log_provenance(source_url="https://c.com", action="cited")

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert content.count("provenance_entry") == 3

    def test_step_context_unbinds_after_report_write(self, tmp_path: Path) -> None:
        """Step context metadata is cleaned up after the context exits."""
        log_file = tmp_path / "ctx.log"
        configure_logging(level="DEBUG", fmt="json", log_file=str(log_file))

        with step_logging_context("synthesize", step_index=4):
            log = structlog.get_logger("test")
            log.info("inside_context")

        # After exiting context, step metadata should be unbound
        ctx = structlog.contextvars.get_contextvars()
        assert "step_name" not in ctx
        assert "step_index" not in ctx


# ---------------------------------------------------------------------------
# End-to-end: report write + provenance + session
# ---------------------------------------------------------------------------


class TestReportProvenanceEndToEnd:
    """Full flow: configure logging, log provenance, write report."""

    def test_full_report_lifecycle(self, tmp_path: Path) -> None:
        """Simulate the complete report lifecycle with logging."""
        log_file = tmp_path / "lifecycle.log"
        configure_logging(
            level="INFO",
            fmt="json",
            log_file=str(log_file),
            session_id="lifecycle-session",
        )

        # Step 1: Log source scraping
        with step_logging_context("scrape", step_index=2):
            log_provenance(
                source_url="https://example.com/article-1",
                action="scraped",
                step_name="scrape",
            )
            log_provenance(
                source_url="https://example.com/article-2",
                action="scraped",
                step_name="scrape",
            )

        # Step 2: Log synthesis
        with step_logging_context("synthesize", step_index=4):
            log_provenance(
                source_url="https://example.com/article-1",
                action="cited",
                step_name="synthesize",
            )

        # Step 3: Write report
        report_text = "# RAG Research\n\nFindings based on multiple sources."
        report_path = write_report(
            report=report_text,
            query="RAG overview",
            output_dir=tmp_path / "reports",
            metadata={
                "session_id": "lifecycle-session",
                "num_sources": 2,
            },
        )

        # Verify report and metadata
        assert report_path.exists()
        meta_path = report_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "lifecycle-session"
        assert meta["num_sources"] == 2

        # Verify provenance log
        for h in logging.getLogger().handlers:
            h.flush()
        log_content = log_file.read_text()
        assert "lifecycle-session" in log_content
        assert log_content.count("provenance_entry") == 3
