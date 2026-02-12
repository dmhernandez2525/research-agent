"""PDF report output using PyMuPDF Story API.

Converts Markdown reports to PDF with preserved headings, lists, code blocks,
and citation links. Falls back gracefully when pymupdf is not installed
(optional ``pdf`` dependency group).

Install with: ``pip install research-agent[pdf]``
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_WIDTH = 612  # US Letter (points)
_PAGE_HEIGHT = 792
_MARGIN = 54  # ~0.75 inch margin

_CSS = """
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 { font-size: 20pt; margin-top: 18pt; margin-bottom: 8pt; color: #111; }
h2 { font-size: 16pt; margin-top: 14pt; margin-bottom: 6pt; color: #222; }
h3 { font-size: 13pt; margin-top: 12pt; margin-bottom: 4pt; color: #333; }
h4 { font-size: 11pt; margin-top: 10pt; margin-bottom: 4pt; color: #444; }
p { margin-top: 4pt; margin-bottom: 4pt; }
ul, ol { margin-top: 4pt; margin-bottom: 4pt; padding-left: 20pt; }
li { margin-bottom: 2pt; }
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 1pt 3pt;
}
pre {
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 8pt;
    margin-top: 6pt;
    margin-bottom: 6pt;
    white-space: pre-wrap;
}
a { color: #1a73e8; text-decoration: underline; }
blockquote {
    margin-left: 16pt;
    padding-left: 8pt;
    border-left: 2pt solid #ccc;
    color: #555;
}
hr { border: none; border-top: 1pt solid #ccc; margin: 12pt 0; }
"""


# ---------------------------------------------------------------------------
# Markdown to HTML conversion
# ---------------------------------------------------------------------------

# Regex patterns for inline markup
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline_markup(text: str) -> str:
    """Convert inline Markdown markup to HTML."""
    text = html.escape(text, quote=False)
    # Order matters: bold before italic (both use asterisks)
    text = _INLINE_CODE_RE.sub(r"<code>\1</code>", text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)
    return text


def markdown_to_html(markdown: str) -> str:
    """Convert a Markdown string to basic HTML.

    Handles headings (h1-h4), paragraphs, unordered/ordered lists,
    fenced code blocks, blockquotes, horizontal rules, and inline
    markup (bold, italic, code, links).

    This is intentionally simple and designed for the structured
    reports produced by the research-agent synthesizer. It does not
    aim to be a full CommonMark implementation.

    Args:
        markdown: The Markdown source text.

    Returns:
        An HTML string suitable for the PyMuPDF Story API.
    """
    lines = markdown.split("\n")
    html_parts: list[str] = []
    in_code_block = False
    code_lines: list[str] = []
    in_list = False
    list_type = ""
    paragraph_lines: list[str] = []

    def _flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            html_parts.append(f"<p>{_inline_markup(text)}</p>")
            paragraph_lines.clear()

    def _flush_list() -> None:
        nonlocal in_list, list_type
        if in_list:
            html_parts.append(f"</{list_type}>")
            in_list = False
            list_type = ""

    for line in lines:
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith("```"):
            if in_code_block:
                # Close code block
                escaped = html.escape("\n".join(code_lines))
                html_parts.append(f"<pre><code>{escaped}</code></pre>")
                code_lines.clear()
                in_code_block = False
            else:
                _flush_paragraph()
                _flush_list()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            _flush_paragraph()
            _flush_list()
            html_parts.append("<hr/>")
            continue

        # Headings
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            _flush_paragraph()
            _flush_list()
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            html_parts.append(f"<h{level}>{_inline_markup(text)}</h{level}>")
            continue

        # Blockquote
        if stripped.startswith("> "):
            _flush_paragraph()
            _flush_list()
            text = stripped[2:]
            html_parts.append(f"<blockquote><p>{_inline_markup(text)}</p></blockquote>")
            continue

        # Unordered list item
        ul_match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if ul_match:
            _flush_paragraph()
            if not in_list or list_type != "ul":
                _flush_list()
                html_parts.append("<ul>")
                in_list = True
                list_type = "ul"
            html_parts.append(f"<li>{_inline_markup(ul_match.group(1))}</li>")
            continue

        # Ordered list item
        ol_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ol_match:
            _flush_paragraph()
            if not in_list or list_type != "ol":
                _flush_list()
                html_parts.append("<ol>")
                in_list = True
                list_type = "ol"
            html_parts.append(f"<li>{_inline_markup(ol_match.group(1))}</li>")
            continue

        # Empty line
        if not stripped:
            _flush_paragraph()
            _flush_list()
            continue

        # Regular text (accumulate into paragraph)
        paragraph_lines.append(stripped)

    # Flush remaining
    _flush_paragraph()
    _flush_list()
    if in_code_block and code_lines:
        escaped = html.escape("\n".join(code_lines))
        html_parts.append(f"<pre><code>{escaped}</code></pre>")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------


def generate_pdf(
    markdown: str,
    output_path: Path | str,
    title: str = "",
) -> Path | None:
    """Convert a Markdown report to a PDF file.

    Uses PyMuPDF's Story API to render HTML content across multiple
    pages with proper pagination.

    Args:
        markdown: The Markdown report content.
        output_path: Where to write the PDF file.
        title: Optional document title for PDF metadata.

    Returns:
        The Path to the generated PDF, or None if pymupdf is not installed.
    """
    try:
        import pymupdf
    except ImportError:
        logger.warning(
            "pymupdf_not_installed",
            hint="Install with: pip install research-agent[pdf]",
        )
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    body_html = markdown_to_html(markdown)
    full_html = f"<html><body>{body_html}</body></html>"

    mediabox = pymupdf.Rect(0, 0, _PAGE_WIDTH, _PAGE_HEIGHT)
    content_rect = pymupdf.Rect(
        _MARGIN, _MARGIN,
        _PAGE_WIDTH - _MARGIN, _PAGE_HEIGHT - _MARGIN,
    )

    story = pymupdf.Story(html=full_html, user_css=_CSS)
    writer = pymupdf.DocumentWriter(str(output))

    while True:
        device = writer.begin_page(mediabox)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
        if not more:
            break

    writer.close()

    if title:
        doc = pymupdf.open(str(output))
        doc.set_metadata({"title": title, "producer": "research-agent"})
        doc.saveIncr()
        doc.close()

    logger.info(
        "pdf_generated",
        path=str(output),
        title=title or "(untitled)",
    )

    return output


def write_pdf_report(
    report: str,
    query: str,
    output_dir: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Write a research report as PDF, with Markdown fallback.

    Attempts to generate a PDF. If pymupdf is not installed, logs a
    warning and returns None (the caller can fall back to Markdown).

    Args:
        report: The Markdown report content.
        query: The research query (used for filename and title).
        output_dir: Directory to write the PDF into.
        metadata: Optional metadata (not included in PDF, for caller use).

    Returns:
        Path to the PDF file, or None if PDF generation is unavailable.
    """
    from research_agent.report_output import (
        generate_report_filename,
    )

    md_filename = generate_report_filename(query)
    pdf_filename = md_filename.replace(".md", ".pdf")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / pdf_filename

    return generate_pdf(report, pdf_path, title=query)
