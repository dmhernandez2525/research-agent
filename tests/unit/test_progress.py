"""Unit tests for research_agent.progress - progressive markdown output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.progress import ProgressWriter

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# ProgressWriter initialization
# ---------------------------------------------------------------------------


class TestProgressWriterInit:
    """ProgressWriter creates parent dirs and optional header."""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "progress.md"
        ProgressWriter(path)
        assert path.parent.exists()

    def test_no_file_without_title(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        ProgressWriter(path)
        # No title, so no header written, file not yet created
        assert not path.exists()

    def test_writes_header_with_title(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        ProgressWriter(path, title="AI Research Report")
        assert path.exists()
        content = path.read_text()
        assert "# AI Research Report" in content
        assert "Research in progress" in content

    def test_header_not_overwritten_on_reinit(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="First Title")
        writer.append_subtopic("Topic A", "Summary A")
        # Re-init with a different title should not overwrite
        writer2 = ProgressWriter(path, title="Second Title")
        content = writer2.read()
        assert "First Title" in content
        assert "Topic A" in content
        assert "Second Title" not in content


# ---------------------------------------------------------------------------
# append_subtopic
# ---------------------------------------------------------------------------


class TestAppendSubtopic:
    """Appending subtopic summaries to the progress file."""

    def test_appends_heading(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic("Machine Learning Basics", "ML is a subfield of AI.")
        content = writer.read()
        assert "## Machine Learning Basics" in content

    def test_appends_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic("Topic", "This is the summary text.")
        content = writer.read()
        assert "This is the summary text." in content

    def test_appends_citations(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic(
            "Topic",
            "Summary.",
            citations=["https://example.com/a", "https://example.com/b"],
        )
        content = writer.read()
        assert "**Sources:**" in content
        assert "- https://example.com/a" in content
        assert "- https://example.com/b" in content

    def test_appends_key_findings(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic(
            "Topic",
            "Summary.",
            key_findings=["Finding 1", "Finding 2"],
        )
        content = writer.read()
        assert "**Key Findings:**" in content
        assert "- Finding 1" in content
        assert "- Finding 2" in content

    def test_no_citations_section_when_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic("Topic", "Summary.")
        content = writer.read()
        assert "**Sources:**" not in content

    def test_no_findings_section_when_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic("Topic", "Summary.")
        content = writer.read()
        assert "**Key Findings:**" not in content

    def test_separator_after_each_subtopic(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_subtopic("Topic A", "Summary A.")
        content = writer.read()
        assert "---" in content

    def test_multiple_subtopics(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="Report")
        writer.append_subtopic("Topic A", "Summary A.")
        writer.append_subtopic("Topic B", "Summary B.")
        writer.append_subtopic("Topic C", "Summary C.")
        content = writer.read()
        assert content.count("## ") == 3
        assert "Topic A" in content
        assert "Topic B" in content
        assert "Topic C" in content


# ---------------------------------------------------------------------------
# append_error_note
# ---------------------------------------------------------------------------


class TestAppendErrorNote:
    """Error notes in progress output."""

    def test_appends_error_note(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_error_note("scrape", "Connection timeout")
        content = writer.read()
        assert "Error in *scrape*" in content
        assert "Connection timeout" in content

    def test_error_note_is_blockquote(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_error_note("search", "Rate limited")
        content = writer.read()
        assert content.strip().startswith(">")


# ---------------------------------------------------------------------------
# append_status
# ---------------------------------------------------------------------------


class TestAppendStatus:
    """Status updates in progress output."""

    def test_appends_status(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_status("Synthesizing final report...")
        content = writer.read()
        assert "Synthesizing final report..." in content

    def test_status_is_italic(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        writer.append_status("Processing")
        content = writer.read()
        assert "*Processing*" in content


# ---------------------------------------------------------------------------
# read and subtopic_count
# ---------------------------------------------------------------------------


class TestRead:
    """Reading progress file content."""

    def test_read_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        assert writer.read() == ""

    def test_read_returns_content(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="Test Report")
        writer.append_subtopic("Topic", "Content.")
        content = writer.read()
        assert "Test Report" in content
        assert "Content." in content


class TestSubtopicCount:
    """Counting completed subtopics."""

    def test_zero_when_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path)
        assert writer.subtopic_count() == 0

    def test_counts_subtopics(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="Report")
        writer.append_subtopic("A", "Summary A.")
        writer.append_subtopic("B", "Summary B.")
        assert writer.subtopic_count() == 2

    def test_excludes_header(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="Report")
        # Header is "# Report" (level 1), should not count
        assert writer.subtopic_count() == 0

    def test_count_after_mixed_content(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="Report")
        writer.append_subtopic("A", "Summary A.")
        writer.append_error_note("search", "Error")
        writer.append_subtopic("B", "Summary B.")
        writer.append_status("Done")
        assert writer.subtopic_count() == 2


# ---------------------------------------------------------------------------
# Integration: readable partial report
# ---------------------------------------------------------------------------


class TestPartialReport:
    """A partial report should be readable even without synthesis."""

    def test_partial_report_is_valid_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.md"
        writer = ProgressWriter(path, title="AI Safety Research")
        writer.append_subtopic(
            "Alignment Problem",
            "The alignment problem concerns ensuring AI systems act in accordance with human values.",
            citations=["https://arxiv.org/example"],
            key_findings=["Value alignment is unsolved"],
        )
        writer.append_subtopic(
            "Interpretability",
            "Interpretability research aims to understand how neural networks make decisions.",
            citations=["https://distill.pub/example"],
        )
        writer.append_error_note("scrape", "Timeout on third source")
        writer.append_status("Research paused. 2 of 5 subtopics completed.")

        content = writer.read()

        # Verify structure
        assert content.startswith("# AI Safety Research")
        assert "## Alignment Problem" in content
        assert "## Interpretability" in content
        assert "**Key Findings:**" in content
        assert "**Sources:**" in content
        assert "Error in *scrape*" in content
        assert "Research paused" in content
        assert writer.subtopic_count() == 2
