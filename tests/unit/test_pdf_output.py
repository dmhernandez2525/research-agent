"""Unit tests for research_agent.pdf_output."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from research_agent.pdf_output import (
    generate_pdf,
    markdown_to_html,
    write_pdf_report,
)

# ---------------------------------------------------------------------------
# TestMarkdownToHtml
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    """markdown_to_html converts Markdown to basic HTML."""

    def test_heading_h1(self) -> None:
        result = markdown_to_html("# Title")
        assert "<h1>" in result
        assert "Title" in result

    def test_heading_h2(self) -> None:
        result = markdown_to_html("## Subtitle")
        assert "<h2>" in result
        assert "Subtitle" in result

    def test_heading_h3(self) -> None:
        result = markdown_to_html("### Section")
        assert "<h3>" in result

    def test_heading_h4(self) -> None:
        result = markdown_to_html("#### Subsection")
        assert "<h4>" in result

    def test_paragraph(self) -> None:
        result = markdown_to_html("Hello world.")
        assert "<p>Hello world.</p>" in result

    def test_multi_line_paragraph(self) -> None:
        md = "First line\nSecond line"
        result = markdown_to_html(md)
        assert "<p>First line Second line</p>" in result

    def test_paragraphs_separated_by_blank_line(self) -> None:
        md = "Paragraph one.\n\nParagraph two."
        result = markdown_to_html(md)
        assert result.count("<p>") == 2

    def test_bold_text(self) -> None:
        result = markdown_to_html("This is **bold** text.")
        assert "<strong>bold</strong>" in result

    def test_italic_text(self) -> None:
        result = markdown_to_html("This is *italic* text.")
        assert "<em>italic</em>" in result

    def test_inline_code(self) -> None:
        result = markdown_to_html("Use `print()` here.")
        assert "<code>print()</code>" in result

    def test_link(self) -> None:
        result = markdown_to_html("[Click here](https://example.com)")
        assert '<a href="https://example.com">Click here</a>' in result

    def test_unordered_list(self) -> None:
        md = "- Item one\n- Item two\n- Item three"
        result = markdown_to_html(md)
        assert "<ul>" in result
        assert result.count("<li>") == 3
        assert "</ul>" in result

    def test_ordered_list(self) -> None:
        md = "1. First\n2. Second\n3. Third"
        result = markdown_to_html(md)
        assert "<ol>" in result
        assert result.count("<li>") == 3
        assert "</ol>" in result

    def test_fenced_code_block(self) -> None:
        md = "```\ndef hello():\n    pass\n```"
        result = markdown_to_html(md)
        assert "<pre><code>" in result
        assert "def hello():" in result
        assert "</code></pre>" in result

    def test_code_block_escapes_html(self) -> None:
        md = "```\n<script>alert('xss')</script>\n```"
        result = markdown_to_html(md)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_blockquote(self) -> None:
        result = markdown_to_html("> This is quoted.")
        assert "<blockquote>" in result
        assert "This is quoted." in result

    def test_horizontal_rule(self) -> None:
        result = markdown_to_html("---")
        assert "<hr/>" in result

    def test_horizontal_rule_asterisks(self) -> None:
        result = markdown_to_html("***")
        assert "<hr/>" in result

    def test_empty_input(self) -> None:
        result = markdown_to_html("")
        assert result == ""

    def test_mixed_content(self) -> None:
        md = "# Title\n\nSome **bold** text.\n\n- Item 1\n- Item 2\n\n---\n\n> Quote"
        result = markdown_to_html(md)
        assert "<h1>" in result
        assert "<strong>bold</strong>" in result
        assert "<ul>" in result
        assert "<hr/>" in result
        assert "<blockquote>" in result

    def test_html_entities_escaped(self) -> None:
        result = markdown_to_html("Use < and > symbols & ampersands.")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_unclosed_code_block_still_renders(self) -> None:
        md = "```\nsome code"
        result = markdown_to_html(md)
        assert "<pre><code>" in result
        assert "some code" in result

    def test_list_followed_by_paragraph(self) -> None:
        md = "- Item\n\nAfter list."
        result = markdown_to_html(md)
        assert "</ul>" in result
        assert "<p>After list.</p>" in result

    def test_plus_and_star_list_markers(self) -> None:
        md = "+ Plus item\n* Star item"
        result = markdown_to_html(md)
        assert result.count("<li>") == 2

    def test_ordered_to_unordered_list_transition(self) -> None:
        md = "1. First\n- Then"
        result = markdown_to_html(md)
        assert "</ol>" in result
        assert "<ul>" in result


# ---------------------------------------------------------------------------
# TestGeneratePdf
# ---------------------------------------------------------------------------


class TestGeneratePdf:
    """generate_pdf creates PDF from Markdown content."""

    def test_returns_none_when_pymupdf_not_installed(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"pymupdf": None}):
            result = generate_pdf("# Test", tmp_path / "test.pdf")
        assert result is None

    def test_generates_pdf_file(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            result = generate_pdf("# Test Report", tmp_path / "out.pdf")
        assert result == tmp_path / "out.pdf"

    def test_calls_story_api(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            generate_pdf("# Hello", tmp_path / "out.pdf")

        mock_pymupdf.Story.assert_called_once()
        mock_pymupdf.DocumentWriter.assert_called_once_with(
            str(tmp_path / "out.pdf")
        )

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        nested = tmp_path / "a" / "b" / "out.pdf"
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            generate_pdf("# Test", nested)
        assert nested.parent.exists()

    def test_sets_metadata_when_title_provided(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        mock_doc = MagicMock()
        mock_pymupdf.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            generate_pdf("# Test", tmp_path / "out.pdf", title="My Report")

        mock_pymupdf.open.assert_called_once_with(str(tmp_path / "out.pdf"))
        mock_doc.set_metadata.assert_called_once_with(
            {"title": "My Report", "producer": "research-agent"}
        )
        mock_doc.saveIncr.assert_called_once()
        mock_doc.close.assert_called_once()

    def test_skips_metadata_when_no_title(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        mock_pymupdf.open = MagicMock()

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            generate_pdf("# Test", tmp_path / "out.pdf")

        mock_pymupdf.open.assert_not_called()

    def test_multi_page_rendering(self, tmp_path: Path) -> None:
        """Verifies the story loop handles multi-page content."""
        call_count = 0

        def place_side_effect(rect: object) -> tuple[int, object]:
            nonlocal call_count
            call_count += 1
            # First call returns more=1, second returns more=0
            if call_count == 1:
                return (1, MagicMock())
            return (0, MagicMock())

        mock_pymupdf = _make_pymupdf_module(place_side_effect=place_side_effect)
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            generate_pdf("# Long content", tmp_path / "out.pdf")

        mock_writer = mock_pymupdf.DocumentWriter.return_value
        assert mock_writer.begin_page.call_count == 2
        assert mock_writer.end_page.call_count == 2


# ---------------------------------------------------------------------------
# TestWritePdfReport
# ---------------------------------------------------------------------------


class TestWritePdfReport:
    """write_pdf_report creates a PDF report with proper naming."""

    def test_returns_none_when_pymupdf_unavailable(self, tmp_path: Path) -> None:
        with patch.dict(sys.modules, {"pymupdf": None}):
            result = write_pdf_report("# Test", "my query", tmp_path)
        assert result is None

    def test_generates_pdf_with_correct_filename(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            result = write_pdf_report("# Report", "What is RAG?", tmp_path)
        assert result is not None
        assert result.suffix == ".pdf"
        assert "what-is-rag" in result.name

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        nested = tmp_path / "nested" / "reports"
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            write_pdf_report("# Report", "test", nested)
        assert nested.exists()

    def test_passes_query_as_title(self, tmp_path: Path) -> None:
        mock_pymupdf = _make_pymupdf_module()
        mock_doc = MagicMock()
        mock_pymupdf.open = MagicMock(return_value=mock_doc)

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            write_pdf_report("# Report", "My Research Query", tmp_path)

        mock_doc.set_metadata.assert_called_once()
        meta_arg = mock_doc.set_metadata.call_args[0][0]
        assert meta_arg["title"] == "My Research Query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pymupdf_module(
    place_side_effect: object | None = None,
) -> types.ModuleType:
    """Create a fake pymupdf module for testing.

    Args:
        place_side_effect: Optional callable for Story.place side effect.
            Defaults to returning (0, MagicMock()) on first call.

    Returns:
        A mock pymupdf module.
    """
    mod = types.ModuleType("pymupdf")

    # Rect mock
    mock_rect = MagicMock()
    mod.Rect = MagicMock(return_value=mock_rect)  # type: ignore[attr-defined]

    # Story mock
    mock_story = MagicMock()
    if place_side_effect:
        mock_story.place = MagicMock(side_effect=place_side_effect)
    else:
        mock_story.place = MagicMock(return_value=(0, MagicMock()))
    mock_story.draw = MagicMock()
    mod.Story = MagicMock(return_value=mock_story)  # type: ignore[attr-defined]

    # DocumentWriter mock
    mock_writer = MagicMock()
    mock_writer.begin_page = MagicMock(return_value=MagicMock())
    mock_writer.end_page = MagicMock()
    mock_writer.close = MagicMock()
    mod.DocumentWriter = MagicMock(return_value=mock_writer)  # type: ignore[attr-defined]

    # open() mock for metadata setting
    mock_doc = MagicMock()
    mod.open = MagicMock(return_value=mock_doc)  # type: ignore[attr-defined]

    return mod
