"""Integration tests for the evaluate -> revise -> scorecard pipeline."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from research_agent.evaluation import (
    EVALUATION_DIMENSIONS,
    QUALITY_THRESHOLD,
    ReportEvaluator,
    RevisionManager,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_json_response(scores: dict[str, float], overall: str = "") -> str:
    """Build a mock LLM JSON response with per-dimension scores."""
    dims = []
    for name, _weight in EVALUATION_DIMENSIONS:
        dims.append(
            {
                "dimension": name,
                "score": scores.get(name, 3.0),
                "reasoning": f"{name} scored at {scores.get(name, 3.0)}",
            }
        )
    return json.dumps(
        {
            "dimensions": dims,
            "overall_reasoning": overall or "Assessment of the report.",
            "recommendations": ["Improve clarity", "Add more sources"],
        }
    )


def _make_high_score_response() -> str:
    """Build a response where all dimensions score 4.5+."""
    scores = {name: 4.5 for name, _ in EVALUATION_DIMENSIONS}
    return _make_llm_json_response(scores, "Excellent report.")


def _make_low_score_response() -> str:
    """Build a response where all dimensions score 2.0."""
    scores = {name: 2.0 for name, _ in EVALUATION_DIMENSIONS}
    return _make_llm_json_response(scores, "Needs significant improvement.")


def _make_medium_score_response() -> str:
    """Build a response where overall ~ 3.0 (below threshold)."""
    scores = {name: 3.0 for name, _ in EVALUATION_DIMENSIONS}
    return _make_llm_json_response(scores, "Adequate but below threshold.")


# ---------------------------------------------------------------------------
# Evaluate -> Scorecard pipeline
# ---------------------------------------------------------------------------


class TestEvaluateToScorecardPipeline:
    """Evaluation results flow into scorecard generation and file save."""

    @pytest.mark.asyncio()
    async def test_evaluate_then_format_scorecard(self) -> None:
        """Evaluate a report and produce a formatted scorecard."""
        evaluator = ReportEvaluator()
        response = _make_high_score_response()

        async def mock_llm(prompt: str) -> str:
            return response

        result = await evaluator.evaluate(
            query="What is RAG?",
            report="RAG combines retrieval with generation for grounded answers.",
            llm_callable=mock_llm,
        )

        scorecard = evaluator.format_scorecard(result)
        assert "Evaluation Scorecard" in scorecard
        assert "PASS" in scorecard
        assert "What is RAG?" in scorecard

    @pytest.mark.asyncio()
    async def test_evaluate_below_threshold_shows_fail(self) -> None:
        """A report scoring below threshold should show FAIL status."""
        evaluator = ReportEvaluator()
        response = _make_low_score_response()

        async def mock_llm(prompt: str) -> str:
            return response

        result = await evaluator.evaluate(
            query="What is RAG?",
            report="Short report.",
            llm_callable=mock_llm,
        )

        scorecard = evaluator.format_scorecard(result)
        assert "FAIL" in scorecard
        assert result.overall_score < QUALITY_THRESHOLD

    @pytest.mark.asyncio()
    async def test_evaluate_and_save_scorecard(self, tmp_path: Path) -> None:
        """Evaluate, format scorecard, and save to disk alongside a report."""
        evaluator = ReportEvaluator()
        response = _make_high_score_response()

        async def mock_llm(prompt: str) -> str:
            return response

        result = await evaluator.evaluate(
            query="RAG overview",
            report="Detailed RAG report content here.",
            llm_callable=mock_llm,
        )

        # Write a mock report file
        report_path = tmp_path / "rag-overview.md"
        report_path.write_text("# RAG Report\nContent here.")

        scorecard = evaluator.format_scorecard(result)
        saved = evaluator.save_scorecard(scorecard, report_path)

        assert saved.exists()
        assert saved.suffix == ".md"
        assert "scorecard" in saved.name
        content = saved.read_text()
        assert "Evaluation Scorecard" in content

    @pytest.mark.asyncio()
    async def test_rich_scorecard_format(self) -> None:
        """Rich terminal scorecard includes alignment and markers."""
        evaluator = ReportEvaluator()
        response = _make_high_score_response()

        async def mock_llm(prompt: str) -> str:
            return response

        result = await evaluator.evaluate(
            query="What is RAG?",
            report="Full report.",
            llm_callable=mock_llm,
        )

        rich = evaluator.format_scorecard_rich(result)
        assert "[PASS]" in rich
        assert "EVALUATION SCORECARD" in rich
        assert "Factual Accuracy" in rich


# ---------------------------------------------------------------------------
# Revision pipeline
# ---------------------------------------------------------------------------


class TestRevisionPipeline:
    """RevisionManager runs evaluate-revise loops end to end."""

    @pytest.mark.asyncio()
    async def test_passing_report_skips_revision(self) -> None:
        """A report that passes on first eval should not be revised."""
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return _make_high_score_response()

        async def mock_revise(report: str, feedback: str) -> str:
            return report + " [revised]"

        manager = RevisionManager()
        result = await manager.run(
            query="What is RAG?",
            report="Good report content.",
            llm_callable=mock_llm,
            revision_callable=mock_revise,
        )

        assert result.passed is True
        assert result.stop_reason == "passed"
        assert result.total_cycles == 0
        assert "[revised]" not in result.best_report

    @pytest.mark.asyncio()
    async def test_failing_report_triggers_revision(self) -> None:
        """A report below threshold should trigger revision cycles."""
        eval_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal eval_count
            eval_count += 1
            # Cycle 0: score 2.0, cycle 1: score 2.5 (improving), cycle 2: score 4.5
            if eval_count == 1:
                scores = {name: 2.0 for name, _ in EVALUATION_DIMENSIONS}
            elif eval_count == 2:
                scores = {name: 2.5 for name, _ in EVALUATION_DIMENSIONS}
            else:
                scores = {name: 4.5 for name, _ in EVALUATION_DIMENSIONS}
            return _make_llm_json_response(scores)

        async def mock_revise(report: str, feedback: str) -> str:
            return report + " [improved]"

        manager = RevisionManager(max_cycles=3)
        result = await manager.run(
            query="What is RAG?",
            report="Initial report.",
            llm_callable=mock_llm,
            revision_callable=mock_revise,
        )

        assert result.passed is True
        assert result.total_cycles >= 1
        assert len(result.history) >= 2

    @pytest.mark.asyncio()
    async def test_max_cycles_reached(self) -> None:
        """Revision should stop after max_cycles even if still failing."""
        eval_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal eval_count
            eval_count += 1
            # Scores improve enough to avoid diminishing_returns but stay below threshold
            base = 2.0 + (eval_count - 1) * 0.2
            scores = {name: min(base, 3.3) for name, _ in EVALUATION_DIMENSIONS}
            return _make_llm_json_response(scores)

        async def mock_revise(report: str, feedback: str) -> str:
            return report + " [attempt]"

        manager = RevisionManager(max_cycles=2)
        result = await manager.run(
            query="What is RAG?",
            report="Bad report.",
            llm_callable=mock_llm,
            revision_callable=mock_revise,
        )

        assert result.passed is False
        assert result.stop_reason == "max_cycles_reached"
        assert result.total_cycles == 2

    @pytest.mark.asyncio()
    async def test_diminishing_returns_stops_early(self) -> None:
        """Revision stops if score improvement is below threshold."""
        eval_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal eval_count
            eval_count += 1
            # Scores barely improve: 2.0 -> 2.05
            if eval_count == 1:
                scores = {name: 2.0 for name, _ in EVALUATION_DIMENSIONS}
            else:
                scores = {name: 2.05 for name, _ in EVALUATION_DIMENSIONS}
            return _make_llm_json_response(scores)

        async def mock_revise(report: str, feedback: str) -> str:
            return report + " [tweaked]"

        manager = RevisionManager(max_cycles=5, min_improvement=0.1)
        result = await manager.run(
            query="What is RAG?",
            report="Report.",
            llm_callable=mock_llm,
            revision_callable=mock_revise,
        )

        assert result.passed is False
        assert result.stop_reason == "diminishing_returns"
        assert result.total_cycles < 5

    @pytest.mark.asyncio()
    async def test_best_report_retained_across_cycles(self) -> None:
        """The best-scoring report should be returned even if a later cycle scores lower."""
        eval_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                scores = {name: 2.0 for name, _ in EVALUATION_DIMENSIONS}
            elif eval_count == 2:
                scores = {name: 3.2 for name, _ in EVALUATION_DIMENSIONS}
            else:
                scores = {name: 2.5 for name, _ in EVALUATION_DIMENSIONS}
            return _make_llm_json_response(scores)

        revision_count = 0

        async def mock_revise(report: str, feedback: str) -> str:
            nonlocal revision_count
            revision_count += 1
            return f"revision-{revision_count}"

        manager = RevisionManager(max_cycles=3, min_improvement=0.01)
        result = await manager.run(
            query="Q",
            report="initial",
            llm_callable=mock_llm,
            revision_callable=mock_revise,
        )

        # Best should be cycle 1 (score 3.2)
        assert result.best_evaluation.overall_score == pytest.approx(3.2, abs=0.1)
        assert result.best_report == "revision-1"
