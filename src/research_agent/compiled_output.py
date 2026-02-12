"""COMPILED_RESEARCH.md structured output formatter.

Formats research reports into a standardized structure with sections for
executive summary, key findings, detailed analysis, technical
considerations, sources, and methodology. Designed to produce output
that integrates with BUILD_PROMPT.md workflows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from research_agent.report_output import sanitize_filename

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Section templates
# ---------------------------------------------------------------------------

_COMPILED_TEMPLATE = """\
# {title}

*Compiled: {timestamp}*

---

## Executive Summary

{executive_summary}

---

## Key Findings

{key_findings}

---

## Detailed Analysis

{detailed_analysis}

---

## Technical Considerations

{technical_considerations}

---

## Sources

{sources}

---

## Methodology

{methodology}
"""

_DEFAULT_METHODOLOGY = (
    "This research was conducted using an automated multi-stage pipeline: "
    "(1) query decomposition into focused subtopics, "
    "(2) web search across multiple providers, "
    "(3) content extraction and quality scoring, "
    "(4) per-subtopic summarization with source verification, "
    "(5) cross-subtopic synthesis with citation tracking."
)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _extract_sections(report: str) -> dict[str, str]:
    """Extract sections from a standard research report.

    Splits on ``##`` headings and maps normalized heading names to
    their body text.

    Args:
        report: The full Markdown report text.

    Returns:
        Dict mapping lowercase heading names to their body text.
    """
    sections: dict[str, str] = {}
    current_heading = "_intro"
    current_lines: list[str] = []

    for line in report.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line[3:].strip().lower()
            current_lines = []
        elif line.startswith("# ") and current_heading == "_intro":
            # Skip the top-level title
            continue
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _find_section(
    sections: dict[str, str],
    candidates: tuple[str, ...],
    default: str = "",
) -> str:
    """Find a section by trying multiple candidate heading names.

    Args:
        sections: Dict of heading name to body text.
        candidates: Tuple of heading names to try (lowercase).
        default: Default value if no candidate matches.

    Returns:
        The section body text, or the default.
    """
    for name in candidates:
        if name in sections:
            return sections[name]
    return default


def format_compiled_research(
    report: str,
    query: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format a research report as COMPILED_RESEARCH.md.

    Restructures the report into the standardized compiled format with
    executive summary, key findings, detailed analysis, technical
    considerations, sources, and methodology sections.

    Args:
        report: The original Markdown research report.
        query: The original research query (used as title).
        metadata: Optional metadata dict for methodology context.

    Returns:
        The formatted COMPILED_RESEARCH.md content.
    """
    sections = _extract_sections(report)

    executive_summary = _find_section(
        sections,
        ("executive summary", "summary", "overview", "_intro"),
        default="No executive summary available.",
    )

    key_findings = _find_section(
        sections,
        ("key findings", "findings", "results", "highlights"),
        default="No key findings extracted.",
    )

    detailed_analysis = _find_section(
        sections,
        ("detailed analysis", "analysis", "discussion", "detailed findings"),
        default=_find_section(sections, ("_intro",), default="No detailed analysis available."),
    )

    technical_considerations = _find_section(
        sections,
        (
            "technical considerations",
            "technical details",
            "implementation",
            "architecture",
            "technical notes",
        ),
        default="No technical considerations noted.",
    )

    sources = _find_section(
        sections,
        ("sources", "references", "bibliography", "citations"),
        default="No sources listed.",
    )

    # Build methodology with optional metadata
    methodology_parts = [_DEFAULT_METHODOLOGY]
    if metadata:
        cost = metadata.get("cost_so_far")
        llm_calls = metadata.get("llm_call_count")
        if cost is not None:
            methodology_parts.append(f"Total cost: ${cost:.4f}")
        if llm_calls is not None:
            methodology_parts.append(f"LLM calls: {llm_calls}")

    methodology = "\n\n".join(methodology_parts)

    now = datetime.now(tz=UTC)

    return _COMPILED_TEMPLATE.format(
        title=query,
        timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
        executive_summary=executive_summary,
        key_findings=key_findings,
        detailed_analysis=detailed_analysis,
        technical_considerations=technical_considerations,
        sources=sources,
        methodology=methodology,
    )


def write_compiled_research(
    report: str,
    query: str,
    output_dir: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a COMPILED_RESEARCH.md file with metadata sidecar.

    Args:
        report: The original Markdown research report.
        query: The original research query.
        output_dir: Directory to write the compiled output into.
        metadata: Optional metadata for the sidecar and methodology.

    Returns:
        Path to the written COMPILED_RESEARCH.md file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled = format_compiled_research(report, query, metadata)

    now = datetime.now(tz=UTC)
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    sanitized = sanitize_filename(query)
    filename = f"COMPILED_RESEARCH_{sanitized}_{ts_str}.md"

    output_path = out_dir / filename
    output_path.write_text(compiled, encoding="utf-8")

    # Write metadata sidecar
    meta = {
        "query": query,
        "compiled_at": now.isoformat(),
        "format": "compiled_research",
        "word_count": len(compiled.split()),
        "filename": filename,
    }
    if metadata:
        meta.update(metadata)

    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info(
        "compiled_research_written",
        path=str(output_path),
        word_count=meta["word_count"],
    )

    return output_path
