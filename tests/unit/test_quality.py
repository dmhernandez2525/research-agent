"""Unit tests for research_agent.quality - report quality checks."""

from __future__ import annotations

import pytest

from research_agent.quality import (
    QualityResult,
    _check_sections,
    _check_subtopic_coverage,
    _count_citations,
    check_report_quality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def good_report() -> str:
    """A well-structured report that passes all checks."""
    return (
        "# Research Report\n\n"
        "## Executive Summary\n\n"
        "This report examines RAG approaches [Source 1].\n\n"
        "## Findings\n\n"
        "RAG improves accuracy by 30% [Source 2]. Fine-tuning requires "
        "labeled data [Source 3]. Vector databases enable retrieval.\n\n"
        "## Analysis\n\n"
        "Both approaches have tradeoffs.\n\n"
        "## Conclusion\n\n"
        "RAG is preferred for general use cases.\n\n"
        "## Sources\n\n"
        "1. [Source A](https://a.com)\n"
        "2. [Source B](https://b.com)\n"
        "3. [Source C](https://c.com)\n"
    )


@pytest.fixture()
def minimal_report() -> str:
    """A report missing several required sections."""
    return "# Report\n\nSome content without proper structure."


@pytest.fixture()
def subtopics() -> list[dict]:
    """Sample subtopics for coverage testing."""
    return [
        {"id": 1, "question": "How does RAG improve accuracy?"},
        {"id": 2, "question": "What does fine-tuning require for data?"},
        {"id": 3, "question": "How do vector databases enable retrieval?"},
    ]


# ---------------------------------------------------------------------------
# TestCheckSections
# ---------------------------------------------------------------------------


class TestCheckSections:
    """_check_sections detects required report sections."""

    def test_finds_all_sections(self, good_report: str) -> None:
        result = _check_sections(good_report)
        assert result["Executive Summary"] is True
        assert result["Findings"] is True
        assert result["Sources"] is True

    def test_missing_sections(self, minimal_report: str) -> None:
        result = _check_sections(minimal_report)
        assert result["Executive Summary"] is False
        assert result["Findings"] is False
        assert result["Sources"] is False

    def test_case_insensitive(self) -> None:
        report = "## executive summary\n\nContent.\n\n## FINDINGS\n\n## sources\n"
        result = _check_sections(report)
        assert result["Executive Summary"] is True
        assert result["Findings"] is True
        assert result["Sources"] is True

    def test_h3_headings(self) -> None:
        report = (
            "### Executive Summary\n\nContent.\n\n### Key Findings\n\n### Sources\n"
        )
        result = _check_sections(report)
        assert result["Executive Summary"] is True
        assert result["Findings"] is True
        assert result["Sources"] is True

    def test_partial_match(self) -> None:
        report = "## Executive Summary\n\nContent.\n\n## Conclusion\n"
        result = _check_sections(report)
        assert result["Executive Summary"] is True
        assert result["Findings"] is False
        assert result["Sources"] is False


# ---------------------------------------------------------------------------
# TestCountCitations
# ---------------------------------------------------------------------------


class TestCountCitations:
    """_count_citations counts unique citation references."""

    def test_counts_source_n_format(self) -> None:
        report = "Content [Source 1] and [Source 2] and [Source 1] again."
        assert _count_citations(report) == 2

    def test_counts_bracket_n_format(self) -> None:
        report = "Content [1] and [2] and [3]."
        assert _count_citations(report) == 3

    def test_mixed_formats(self) -> None:
        report = "Content [Source 1] and [2]."
        assert _count_citations(report) == 2

    def test_no_citations(self) -> None:
        report = "Content without any citations."
        assert _count_citations(report) == 0

    def test_empty_report(self) -> None:
        assert _count_citations("") == 0


# ---------------------------------------------------------------------------
# TestCheckSubtopicCoverage
# ---------------------------------------------------------------------------


class TestCheckSubtopicCoverage:
    """_check_subtopic_coverage checks subtopic mention coverage."""

    def test_full_coverage(self, good_report: str, subtopics: list[dict]) -> None:
        coverage = _check_subtopic_coverage(good_report, subtopics)
        assert coverage >= 0.8

    def test_partial_coverage(self, subtopics: list[dict]) -> None:
        report = "RAG improves accuracy. Nothing about fine-tuning or databases."
        coverage = _check_subtopic_coverage(report, subtopics)
        assert 0.0 < coverage < 1.0

    def test_no_coverage(self, subtopics: list[dict]) -> None:
        report = "This report is about cooking recipes."
        coverage = _check_subtopic_coverage(report, subtopics)
        assert coverage < 0.5

    def test_empty_subtopics(self) -> None:
        coverage = _check_subtopic_coverage("Any report text.", [])
        assert coverage == 1.0

    def test_empty_report(self, subtopics: list[dict]) -> None:
        coverage = _check_subtopic_coverage("", subtopics)
        assert coverage < 1.0


# ---------------------------------------------------------------------------
# TestQualityResult
# ---------------------------------------------------------------------------


class TestQualityResult:
    """QualityResult model validation."""

    def test_defaults_to_not_passed(self) -> None:
        result = QualityResult(passed=False)
        assert result.passed is False
        assert result.word_count == 0
        assert result.warnings == []

    def test_passed_result(self) -> None:
        result = QualityResult(
            passed=True,
            word_count=500,
            has_executive_summary=True,
            has_findings=True,
            has_sources=True,
            citation_count=3,
            has_citations=True,
            subtopic_coverage=0.9,
            subtopic_coverage_ok=True,
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# TestCheckReportQuality
# ---------------------------------------------------------------------------


class TestCheckReportQuality:
    """check_report_quality runs all checks and returns composite result."""

    def test_good_report_passes(
        self, good_report: str, subtopics: list[dict]
    ) -> None:
        result = check_report_quality(good_report, subtopics)
        assert result.passed is True
        assert result.has_executive_summary is True
        assert result.has_findings is True
        assert result.has_sources is True
        assert result.has_citations is True
        assert result.word_count > 0
        assert result.warnings == []

    def test_empty_report_fails(self) -> None:
        result = check_report_quality("")
        assert result.passed is False
        assert "empty" in result.warnings[0].lower()

    def test_minimal_report_fails(self, minimal_report: str) -> None:
        result = check_report_quality(minimal_report)
        assert result.passed is False
        assert len(result.warnings) > 0

    def test_missing_citations_fails(self) -> None:
        report = (
            "## Executive Summary\n\nSummary.\n\n"
            "## Findings\n\nContent without citations.\n\n"
            "## Sources\n\n1. Source A\n"
        )
        result = check_report_quality(report)
        assert result.passed is False
        assert result.has_citations is False

    def test_low_subtopic_coverage_fails(self) -> None:
        report = (
            "## Executive Summary\n\nSummary [Source 1].\n\n"
            "## Findings\n\nOnly about cooking [Source 2].\n\n"
            "## Sources\n\n1. Source\n"
        )
        sqs = [
            {"id": 1, "question": "What is quantum computing?"},
            {"id": 2, "question": "How does quantum entanglement work?"},
            {"id": 3, "question": "What are quantum computing applications?"},
        ]
        result = check_report_quality(report, sqs)
        assert result.subtopic_coverage_ok is False

    def test_no_subtopics_defaults_full_coverage(self) -> None:
        report = (
            "## Executive Summary\n\nSummary [Source 1].\n\n"
            "## Findings\n\nDetails.\n\n"
            "## Sources\n\n1. Source\n"
        )
        result = check_report_quality(report, None)
        assert result.subtopic_coverage == 1.0
        assert result.subtopic_coverage_ok is True

    def test_warnings_list_populated_for_failures(self, minimal_report: str) -> None:
        result = check_report_quality(minimal_report)
        assert len(result.warnings) >= 3  # missing sections + no citations

    def test_word_count_tracked(self, good_report: str) -> None:
        result = check_report_quality(good_report)
        assert result.word_count > 20

    def test_returns_quality_result_type(self, good_report: str) -> None:
        result = check_report_quality(good_report)
        assert isinstance(result, QualityResult)
