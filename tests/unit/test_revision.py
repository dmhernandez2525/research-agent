"""Unit tests for research_agent.evaluation - auto-revision cycles."""

from __future__ import annotations

import json

import pytest

from research_agent.evaluation import (
    _DIMINISHING_RETURNS_THRESHOLD,
    EVALUATION_DIMENSIONS,
    MAX_REVISION_CYCLES,
    QUALITY_THRESHOLD,
    EvaluationResult,
    LLMCallable,
    RevisionCallable,
    RevisionManager,
    RevisionRecord,
    RevisionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_json(
    score: float,
    overall: str = "Test assessment.",
    recommendations: list[str] | None = None,
) -> str:
    """Build a well-formed LLM evaluation JSON response with uniform scores."""
    dims = [
        {"dimension": name, "score": score, "reasoning": f"Score {score}."}
        for name, _ in EVALUATION_DIMENSIONS
    ]
    return json.dumps({
        "dimensions": dims,
        "overall_reasoning": overall,
        "recommendations": recommendations or [],
    })


def _make_eval_callable(scores: list[float]) -> LLMCallable:
    """Return an LLM callable that yields different scores per call.

    Each call returns the next score in the list. Repeats the last score
    once exhausted.
    """
    call_count = 0

    async def _call(prompt: str) -> str:
        nonlocal call_count
        idx = min(call_count, len(scores) - 1)
        call_count += 1
        return _make_eval_json(scores[idx])

    return _call


def _make_revision_callable(
    suffix: str = " [revised]",
) -> RevisionCallable:
    """Return a revision callable that appends a suffix to the report."""

    async def _call(report: str, feedback: str) -> str:
        return report + suffix

    return _call


# ---------------------------------------------------------------------------
# RevisionRecord
# ---------------------------------------------------------------------------


class TestRevisionRecord:
    """RevisionRecord model validation."""

    def test_valid_record(self) -> None:
        evaluation = EvaluationResult(query="test", overall_score=3.0)
        record = RevisionRecord(cycle=0, report="Report text.", evaluation=evaluation)
        assert record.cycle == 0
        assert record.report == "Report text."
        assert record.evaluation.overall_score == 3.0

    def test_cycle_must_be_non_negative(self) -> None:
        from pydantic import ValidationError

        evaluation = EvaluationResult(query="test")
        with pytest.raises(ValidationError):
            RevisionRecord(cycle=-1, report="test", evaluation=evaluation)


# ---------------------------------------------------------------------------
# RevisionResult
# ---------------------------------------------------------------------------


class TestRevisionResult:
    """RevisionResult model validation."""

    def test_default_values(self) -> None:
        evaluation = EvaluationResult(query="test")
        result = RevisionResult(
            best_report="report",
            best_evaluation=evaluation,
            total_cycles=0,
            passed=False,
        )
        assert result.history == []
        assert result.stop_reason == ""

    def test_full_result(self) -> None:
        evaluation = EvaluationResult(query="test", overall_score=4.0)
        record = RevisionRecord(cycle=0, report="report", evaluation=evaluation)
        result = RevisionResult(
            best_report="report",
            best_evaluation=evaluation,
            total_cycles=1,
            passed=True,
            history=[record],
            stop_reason="passed",
        )
        assert len(result.history) == 1
        assert result.passed is True
        assert result.stop_reason == "passed"


# ---------------------------------------------------------------------------
# RevisionManager - Init
# ---------------------------------------------------------------------------


class TestRevisionManagerInit:
    """RevisionManager initialization."""

    def test_default_init(self) -> None:
        manager = RevisionManager()
        assert manager.max_cycles == MAX_REVISION_CYCLES
        assert manager.quality_threshold == QUALITY_THRESHOLD
        assert manager.min_improvement == _DIMINISHING_RETURNS_THRESHOLD

    def test_custom_params(self) -> None:
        manager = RevisionManager(
            max_cycles=5,
            quality_threshold=4.0,
            min_improvement=0.2,
        )
        assert manager.max_cycles == 5
        assert manager.quality_threshold == 4.0
        assert manager.min_improvement == 0.2


# ---------------------------------------------------------------------------
# RevisionManager - should_revise
# ---------------------------------------------------------------------------


class TestShouldRevise:
    """should_revise determines whether another cycle is needed."""

    def test_above_threshold_no_revise(self) -> None:
        manager = RevisionManager()
        assert manager.should_revise(score=4.0, cycle=0) is False

    def test_below_threshold_revise(self) -> None:
        manager = RevisionManager()
        assert manager.should_revise(score=2.0, cycle=0) is True

    def test_at_threshold_no_revise(self) -> None:
        manager = RevisionManager()
        assert manager.should_revise(score=QUALITY_THRESHOLD, cycle=0) is False

    def test_max_cycles_reached(self) -> None:
        manager = RevisionManager(max_cycles=2)
        assert manager.should_revise(score=2.0, cycle=2) is False

    def test_below_max_cycles_revise(self) -> None:
        manager = RevisionManager(max_cycles=2)
        assert manager.should_revise(score=2.0, cycle=1) is True

    def test_diminishing_returns_no_revise(self) -> None:
        manager = RevisionManager(min_improvement=0.1)
        # Previous 2.0, current 2.05 = improvement of 0.05 < 0.1
        assert manager.should_revise(score=2.05, cycle=1, previous_score=2.0) is False

    def test_sufficient_improvement_revise(self) -> None:
        manager = RevisionManager(min_improvement=0.1)
        # Previous 2.0, current 2.5 = improvement of 0.5 >= 0.1
        assert manager.should_revise(score=2.5, cycle=1, previous_score=2.0) is True

    def test_score_decreased_no_revise(self) -> None:
        manager = RevisionManager(min_improvement=0.1)
        # Score went down
        assert manager.should_revise(score=1.8, cycle=1, previous_score=2.0) is False

    def test_first_cycle_no_previous(self) -> None:
        manager = RevisionManager()
        assert manager.should_revise(score=2.0, cycle=0, previous_score=None) is True


# ---------------------------------------------------------------------------
# RevisionManager - _build_revision_feedback
# ---------------------------------------------------------------------------


class TestBuildRevisionFeedback:
    """_build_revision_feedback formats evaluation into actionable text."""

    def test_contains_overall_score(self) -> None:
        evaluation = EvaluationResult(query="test", overall_score=3.0)
        feedback = RevisionManager._build_revision_feedback(evaluation)
        assert "3.0/5.0" in feedback

    def test_contains_threshold(self) -> None:
        evaluation = EvaluationResult(query="test", overall_score=3.0)
        feedback = RevisionManager._build_revision_feedback(evaluation)
        assert str(QUALITY_THRESHOLD) in feedback

    def test_contains_dimension_feedback(self) -> None:
        from research_agent.evaluation import DimensionScore

        dims = [
            DimensionScore(
                dimension="Factual Accuracy",
                score=2.5,
                weight=0.30,
                reasoning="Needs more citations.",
            ),
        ]
        evaluation = EvaluationResult(
            query="test",
            overall_score=2.5,
            dimensions=dims,
        )
        feedback = RevisionManager._build_revision_feedback(evaluation)
        assert "Factual Accuracy" in feedback
        assert "Needs more citations." in feedback

    def test_contains_recommendations(self) -> None:
        evaluation = EvaluationResult(
            query="test",
            overall_score=3.0,
            recommendations=["Add more sources.", "Improve structure."],
        )
        feedback = RevisionManager._build_revision_feedback(evaluation)
        assert "Add more sources." in feedback
        assert "Improve structure." in feedback

    def test_contains_assessment(self) -> None:
        evaluation = EvaluationResult(
            query="test",
            overall_score=3.0,
            overall_reasoning="Report lacks depth.",
        )
        feedback = RevisionManager._build_revision_feedback(evaluation)
        assert "Report lacks depth." in feedback


# ---------------------------------------------------------------------------
# RevisionManager - run (async)
# ---------------------------------------------------------------------------


class TestRevisionManagerRun:
    """run() orchestrates the evaluation-revision loop."""

    @pytest.mark.asyncio
    async def test_passing_report_no_revision(self) -> None:
        """A report that passes on first evaluation needs no revision."""
        llm = _make_eval_callable([4.5])
        revision = _make_revision_callable()
        manager = RevisionManager()

        result = await manager.run("test query", "Good report.", llm, revision)

        assert result.passed is True
        assert result.total_cycles == 0
        assert result.stop_reason == "passed"
        assert len(result.history) == 1
        assert result.best_report == "Good report."

    @pytest.mark.asyncio
    async def test_improvement_across_cycles(self) -> None:
        """Report improves and passes on second evaluation."""
        # First eval: 2.0 (fail), second eval: 4.0 (pass)
        llm = _make_eval_callable([2.0, 4.0])
        revision = _make_revision_callable()
        manager = RevisionManager()

        result = await manager.run("test query", "Initial report.", llm, revision)

        assert result.passed is True
        assert result.total_cycles == 1
        assert result.stop_reason == "passed"
        assert len(result.history) == 2

    @pytest.mark.asyncio
    async def test_max_cycles_reached(self) -> None:
        """Revision stops after max_cycles even if still below threshold."""
        # All evaluations below threshold with sufficient improvement
        llm = _make_eval_callable([2.0, 2.5, 3.0])
        revision = _make_revision_callable()
        manager = RevisionManager(max_cycles=2)

        result = await manager.run("test query", "Weak report.", llm, revision)

        assert result.passed is False
        assert result.total_cycles == 2
        assert result.stop_reason == "max_cycles_reached"
        assert len(result.history) == 3  # initial + 2 revisions

    @pytest.mark.asyncio
    async def test_diminishing_returns_stops_early(self) -> None:
        """Revision stops when improvement is too small."""
        # First: 2.0, second: 2.05 (improvement < 0.1)
        llm = _make_eval_callable([2.0, 2.05])
        revision = _make_revision_callable()
        manager = RevisionManager(max_cycles=5, min_improvement=0.1)

        result = await manager.run("test query", "Mediocre report.", llm, revision)

        assert result.passed is False
        assert result.total_cycles == 1
        assert result.stop_reason == "diminishing_returns"

    @pytest.mark.asyncio
    async def test_best_version_retained(self) -> None:
        """The best-scoring version is returned even if later cycles score lower."""
        # Cycle 0: 3.0, Cycle 1: 3.4 (best), Cycle 2: 2.8 (worse, but improvement
        # from previous is negative so stops)
        llm = _make_eval_callable([3.0, 3.4, 2.8])
        revision = _make_revision_callable(" [rev]")
        manager = RevisionManager(max_cycles=5, min_improvement=0.1)

        result = await manager.run("test query", "Report.", llm, revision)

        # Best should be cycle 1 (score 3.4)
        assert result.best_evaluation.overall_score == pytest.approx(3.4)
        assert result.best_report == "Report. [rev]"

    @pytest.mark.asyncio
    async def test_history_tracks_all_cycles(self) -> None:
        """History records every cycle."""
        llm = _make_eval_callable([2.0, 2.5, 4.0])
        revision = _make_revision_callable()
        manager = RevisionManager(max_cycles=3)

        result = await manager.run("test query", "Start.", llm, revision)

        assert len(result.history) == 3
        assert result.history[0].cycle == 0
        assert result.history[1].cycle == 1
        assert result.history[2].cycle == 2

    @pytest.mark.asyncio
    async def test_revision_callable_receives_feedback(self) -> None:
        """The revision callable receives the report and formatted feedback."""
        received_args: list[tuple[str, str]] = []

        async def _capturing_revision(report: str, feedback: str) -> str:
            received_args.append((report, feedback))
            return report + " [revised]"

        llm = _make_eval_callable([2.0, 4.0])
        manager = RevisionManager()

        await manager.run("test query", "Initial.", llm, _capturing_revision)

        assert len(received_args) == 1
        report, feedback = received_args[0]
        assert report == "Initial."
        assert "2.0/5.0" in feedback

    @pytest.mark.asyncio
    async def test_query_preserved_in_result(self) -> None:
        """The query is preserved in the result evaluations."""
        llm = _make_eval_callable([4.5])
        revision = _make_revision_callable()
        manager = RevisionManager()

        result = await manager.run("my research query", "Good report.", llm, revision)

        assert result.best_evaluation.query == "my research query"

    @pytest.mark.asyncio
    async def test_single_cycle_max(self) -> None:
        """With max_cycles=0, only the initial evaluation runs."""
        llm = _make_eval_callable([2.0])
        revision = _make_revision_callable()
        manager = RevisionManager(max_cycles=0)

        result = await manager.run("test", "Report.", llm, revision)

        assert result.total_cycles == 0
        assert result.passed is False
        assert result.stop_reason == "max_cycles_reached"
        assert len(result.history) == 1

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """Custom quality threshold is respected."""
        llm = _make_eval_callable([3.8])
        revision = _make_revision_callable()
        manager = RevisionManager(quality_threshold=4.0)

        result = await manager.run("test", "Report.", llm, revision)

        # 3.8 < 4.0 threshold, so not passed
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_revision_produces_different_reports(self) -> None:
        """Each revision cycle produces a different report in history."""
        llm = _make_eval_callable([2.0, 2.5, 4.0])
        revision = _make_revision_callable(" [rev]")
        manager = RevisionManager(max_cycles=3)

        result = await manager.run("test", "Base.", llm, revision)

        reports = [r.report for r in result.history]
        assert reports[0] == "Base."
        assert reports[1] == "Base. [rev]"
        assert reports[2] == "Base. [rev] [rev]"
