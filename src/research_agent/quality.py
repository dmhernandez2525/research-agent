"""Basic report quality check.

Validates the structure and content of generated research reports against
a set of quality criteria: required sections, citation presence, subtopic
coverage, and word count.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    "Executive Summary",
    "Findings",
    "Sources",
]

_CITATION_PATTERN = re.compile(r"\[(?:Source\s+)?(\d+)\]")
_HEADING_PATTERN = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

_MIN_SUBTOPIC_COVERAGE = 0.8  # 80%


# ---------------------------------------------------------------------------
# Quality result model
# ---------------------------------------------------------------------------


class QualityResult(BaseModel):
    """Result of a report quality check."""

    passed: bool = Field(description="Whether the report passes all quality checks.")
    word_count: int = Field(default=0, ge=0)
    has_executive_summary: bool = Field(default=False)
    has_findings: bool = Field(default=False)
    has_sources: bool = Field(default=False)
    citation_count: int = Field(default=0, ge=0)
    has_citations: bool = Field(default=False)
    subtopic_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    subtopic_coverage_ok: bool = Field(default=False)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _check_sections(report: str) -> dict[str, bool]:
    """Check for required report sections.

    Looks for heading patterns (# / ## / ###) matching the required
    section names (case-insensitive).

    Args:
        report: The report Markdown text.

    Returns:
        Mapping of section name to whether it was found.
    """
    headings = {m.group(1).strip().lower() for m in _HEADING_PATTERN.finditer(report)}

    results: dict[str, bool] = {}
    for section in _REQUIRED_SECTIONS:
        found = any(section.lower() in h for h in headings)
        results[section] = found
    return results


def _count_citations(report: str) -> int:
    """Count unique citation references in the report.

    Matches both [Source N] and [N] patterns.

    Args:
        report: The report Markdown text.

    Returns:
        Number of unique citation references found.
    """
    matches = {int(m.group(1)) for m in _CITATION_PATTERN.finditer(report)}
    return len(matches)


def _check_subtopic_coverage(
    report: str,
    subtopics: list[dict[str, Any]] | list[Any],
) -> float:
    """Check what fraction of subtopics are mentioned in the report.

    Checks if keywords from each sub-question appear in the report text
    (case-insensitive).

    Args:
        report: The report Markdown text.
        subtopics: Subtopics to check coverage for.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not subtopics:
        return 1.0

    report_lower = report.lower()
    covered = 0

    for sq in subtopics:
        question = (
            sq.get("question", "")
            if isinstance(sq, dict)
            else getattr(sq, "question", "")
        )
        if not question:
            covered += 1
            continue

        # Extract significant words (3+ chars) from the question
        words = [w.lower() for w in question.split() if len(w) >= 3]
        # Consider covered if at least 40% of significant words appear
        if not words:
            covered += 1
            continue

        matches = sum(1 for w in words if w in report_lower)
        if matches / len(words) >= 0.4:
            covered += 1

    return covered / len(subtopics)


# ---------------------------------------------------------------------------
# Main check function
# ---------------------------------------------------------------------------


def check_report_quality(
    report: str,
    subtopics: list[dict[str, Any]] | list[Any] | None = None,
) -> QualityResult:
    """Run all quality checks on a generated report.

    Evaluates structure (required sections), citation presence,
    subtopic coverage, and word count. Returns a composite pass/fail
    result with detailed per-check information.

    Args:
        report: The generated Markdown report.
        subtopics: Subtopics for coverage checking.

    Returns:
        A ``QualityResult`` with pass/fail and detailed metrics.
    """
    warnings: list[str] = []

    if not report.strip():
        return QualityResult(
            passed=False,
            warnings=["Report is empty"],
        )

    # Word count
    word_count = len(report.split())

    # Section checks
    sections = _check_sections(report)
    has_executive_summary = sections.get("Executive Summary", False)
    has_findings = sections.get("Findings", False)
    has_sources = sections.get("Sources", False)

    if not has_executive_summary:
        warnings.append("Missing 'Executive Summary' section")
    if not has_findings:
        warnings.append("Missing 'Findings' section")
    if not has_sources:
        warnings.append("Missing 'Sources' section")

    # Citation check
    citation_count = _count_citations(report)
    has_citations = citation_count > 0
    if not has_citations:
        warnings.append("No citation references found in report")

    # Subtopic coverage
    sqs = subtopics or []
    subtopic_coverage = _check_subtopic_coverage(report, sqs)
    subtopic_coverage_ok = subtopic_coverage >= _MIN_SUBTOPIC_COVERAGE
    if not subtopic_coverage_ok and sqs:
        warnings.append(
            f"Subtopic coverage {subtopic_coverage:.0%} is below "
            f"{_MIN_SUBTOPIC_COVERAGE:.0%} threshold"
        )

    # Composite pass/fail
    passed = (
        has_executive_summary
        and has_findings
        and has_sources
        and has_citations
        and subtopic_coverage_ok
    )

    result = QualityResult(
        passed=passed,
        word_count=word_count,
        has_executive_summary=has_executive_summary,
        has_findings=has_findings,
        has_sources=has_sources,
        citation_count=citation_count,
        has_citations=has_citations,
        subtopic_coverage=round(subtopic_coverage, 3),
        subtopic_coverage_ok=subtopic_coverage_ok,
        warnings=warnings,
    )

    logger.info(
        "quality_check_complete",
        passed=result.passed,
        word_count=result.word_count,
        citation_count=result.citation_count,
        subtopic_coverage=result.subtopic_coverage,
        num_warnings=len(result.warnings),
    )

    return result
