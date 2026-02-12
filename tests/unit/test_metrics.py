"""Unit tests for research_agent.metrics - pipeline metrics collection."""

from __future__ import annotations

import time

import pytest

from research_agent.metrics import MetricsCollector, RunMetrics, StepMetric

# ---------------------------------------------------------------------------
# StepMetric model tests
# ---------------------------------------------------------------------------


class TestStepMetric:
    """StepMetric Pydantic model validation."""

    def test_default_construction(self) -> None:
        step = StepMetric(step_name="plan")
        assert step.step_name == "plan"
        assert step.input_tokens == 0
        assert step.output_tokens == 0
        assert step.cost_usd == 0.0
        assert step.sources_found == 0
        assert step.errors == 0
        assert step.finished_at is None

    def test_started_at_defaults_to_monotonic(self) -> None:
        before = time.monotonic()
        step = StepMetric(step_name="test")
        after = time.monotonic()
        assert before <= step.started_at <= after

    def test_is_complete_false_initially(self) -> None:
        step = StepMetric(step_name="test")
        assert step.is_complete is False

    def test_is_complete_true_after_finished(self) -> None:
        step = StepMetric(step_name="test")
        step.finished_at = time.monotonic()
        assert step.is_complete is True

    def test_duration_seconds_while_running(self) -> None:
        step = StepMetric(step_name="test")
        assert step.duration_seconds >= 0.0

    def test_duration_seconds_after_finished(self) -> None:
        step = StepMetric(step_name="test")
        step.finished_at = step.started_at + 5.0
        assert abs(step.duration_seconds - 5.0) < 0.01

    def test_duration_never_negative(self) -> None:
        step = StepMetric(step_name="test")
        step.finished_at = step.started_at - 1.0
        assert step.duration_seconds == 0.0

    def test_input_tokens_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            StepMetric(step_name="test", input_tokens=-1)

    def test_output_tokens_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            StepMetric(step_name="test", output_tokens=-1)

    def test_cost_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            StepMetric(step_name="test", cost_usd=-0.01)

    def test_sources_found_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            StepMetric(step_name="test", sources_found=-1)

    def test_errors_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            StepMetric(step_name="test", errors=-1)


# ---------------------------------------------------------------------------
# RunMetrics model tests
# ---------------------------------------------------------------------------


class TestRunMetrics:
    """RunMetrics Pydantic model validation."""

    def test_default_construction(self) -> None:
        m = RunMetrics()
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.total_cost_usd == 0.0
        assert m.total_sources == 0
        assert m.total_errors == 0
        assert m.total_findings == 0
        assert m.subtopics_completed == 0
        assert m.subtopics_total == 0
        assert m.budget_usd == 2.0
        assert m.current_step == "idle"
        assert m.model_usage == {}

    def test_total_tokens_property(self) -> None:
        m = RunMetrics(total_input_tokens=100, total_output_tokens=50)
        assert m.total_tokens == 150

    def test_budget_used_pct_zero_cost(self) -> None:
        m = RunMetrics(budget_usd=1.0)
        assert m.budget_used_pct == 0.0

    def test_budget_used_pct_half_cost(self) -> None:
        m = RunMetrics(budget_usd=2.0, total_cost_usd=1.0)
        assert m.budget_used_pct == 50.0

    def test_budget_used_pct_over_budget(self) -> None:
        m = RunMetrics(budget_usd=1.0, total_cost_usd=2.0)
        assert m.budget_used_pct == 100.0  # Capped at 100

    def test_budget_used_pct_zero_budget(self) -> None:
        m = RunMetrics(budget_usd=0.0)
        assert m.budget_used_pct == 100.0

    def test_budget_remaining_usd(self) -> None:
        m = RunMetrics(budget_usd=2.0, total_cost_usd=0.5)
        assert m.budget_remaining_usd == 1.5

    def test_budget_remaining_usd_over_budget(self) -> None:
        m = RunMetrics(budget_usd=1.0, total_cost_usd=2.0)
        assert m.budget_remaining_usd == 0.0  # Capped at 0

    def test_subtopic_progress_pct_zero_total(self) -> None:
        m = RunMetrics(subtopics_total=0)
        assert m.subtopic_progress_pct == 0.0

    def test_subtopic_progress_pct_half(self) -> None:
        m = RunMetrics(subtopics_total=4, subtopics_completed=2)
        assert m.subtopic_progress_pct == 50.0

    def test_subtopic_progress_pct_complete(self) -> None:
        m = RunMetrics(subtopics_total=3, subtopics_completed=3)
        assert m.subtopic_progress_pct == 100.0

    def test_subtopic_progress_pct_over_total(self) -> None:
        m = RunMetrics(subtopics_total=3, subtopics_completed=5)
        assert m.subtopic_progress_pct == 100.0  # Capped at 100

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError):
            RunMetrics(total_input_tokens=-1)


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    """MetricsCollector accumulation logic."""

    def test_default_budget(self) -> None:
        collector = MetricsCollector()
        assert collector.metrics.budget_usd == 2.0

    def test_custom_budget(self) -> None:
        collector = MetricsCollector(budget_usd=5.0)
        assert collector.metrics.budget_usd == 5.0

    def test_start_step_creates_metric(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("plan")
        assert step.step_name == "plan"
        assert not step.is_complete
        assert len(collector.steps) == 1

    def test_start_step_updates_current_step(self) -> None:
        collector = MetricsCollector()
        collector.start_step("search")
        assert collector.metrics.current_step == "search"

    def test_finish_step_sets_finished_at(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("plan")
        collector.finish_step(step)
        assert step.is_complete
        assert step.finished_at is not None

    def test_record_llm_call_accumulates_tokens(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("model-a", input_tokens=100, output_tokens=50)
        collector.record_llm_call("model-a", input_tokens=200, output_tokens=100)
        assert collector.metrics.total_input_tokens == 300
        assert collector.metrics.total_output_tokens == 150

    def test_record_llm_call_accumulates_cost(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("m", cost_usd=0.01)
        collector.record_llm_call("m", cost_usd=0.02)
        assert abs(collector.metrics.total_cost_usd - 0.03) < 1e-9

    def test_record_llm_call_tracks_model_usage(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("model-a")
        collector.record_llm_call("model-b")
        collector.record_llm_call("model-a")
        assert collector.metrics.model_usage["model-a"] == 2
        assert collector.metrics.model_usage["model-b"] == 1

    def test_record_llm_call_updates_current_step(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("search")
        collector.record_llm_call(
            "m", input_tokens=50, output_tokens=25, cost_usd=0.001
        )
        assert step.input_tokens == 50
        assert step.output_tokens == 25
        assert step.cost_usd == 0.001

    def test_record_llm_call_ignores_completed_step(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("plan")
        collector.finish_step(step)
        collector.record_llm_call("m", input_tokens=100)
        # Global metrics still update
        assert collector.metrics.total_input_tokens == 100
        # But the finished step does not
        assert step.input_tokens == 0

    def test_record_sources(self) -> None:
        collector = MetricsCollector()
        collector.start_step("search")
        collector.record_sources(5)
        collector.record_sources(3)
        assert collector.metrics.total_sources == 8

    def test_record_sources_updates_step(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("search")
        collector.record_sources(4)
        assert step.sources_found == 4

    def test_record_findings(self) -> None:
        collector = MetricsCollector()
        collector.record_findings(3)
        collector.record_findings(2)
        assert collector.metrics.total_findings == 5

    def test_record_error(self) -> None:
        collector = MetricsCollector()
        collector.start_step("scrape")
        collector.record_error()
        assert collector.metrics.total_errors == 1

    def test_record_error_updates_step(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("scrape")
        collector.record_error()
        collector.record_error()
        assert step.errors == 2

    def test_set_subtopics(self) -> None:
        collector = MetricsCollector()
        collector.set_subtopics(5)
        assert collector.metrics.subtopics_total == 5

    def test_complete_subtopic(self) -> None:
        collector = MetricsCollector()
        collector.set_subtopics(3)
        collector.complete_subtopic()
        collector.complete_subtopic()
        assert collector.metrics.subtopics_completed == 2

    def test_elapsed_seconds(self) -> None:
        collector = MetricsCollector()
        assert collector.elapsed_seconds >= 0.0

    def test_steps_returns_copy(self) -> None:
        collector = MetricsCollector()
        collector.start_step("plan")
        steps = collector.steps
        steps.clear()
        assert len(collector.steps) == 1  # Original not affected

    def test_snapshot_returns_dict(self) -> None:
        collector = MetricsCollector()
        snap = collector.snapshot()
        assert isinstance(snap, dict)
        assert "elapsed_seconds" in snap
        assert "total_tokens" in snap
        assert "model_usage" in snap

    def test_snapshot_steps_completed_count(self) -> None:
        collector = MetricsCollector()
        s1 = collector.start_step("plan")
        collector.finish_step(s1)
        collector.start_step("search")  # Not finished
        snap = collector.snapshot()
        assert snap["steps_completed"] == 1

    def test_snapshot_budget_fields(self) -> None:
        collector = MetricsCollector(budget_usd=1.0)
        collector.record_llm_call("m", cost_usd=0.25)
        snap = collector.snapshot()
        assert snap["budget_used_pct"] == 25.0
        assert snap["budget_remaining_usd"] == 0.75

    def test_multiple_steps_sequential(self) -> None:
        collector = MetricsCollector()
        for name in ["plan", "search", "scrape", "summarize", "synthesize"]:
            step = collector.start_step(name)
            collector.record_llm_call("m", input_tokens=10)
            collector.finish_step(step)
        assert len(collector.steps) == 5
        assert all(s.is_complete for s in collector.steps)
        assert collector.metrics.total_input_tokens == 50

    def test_no_steps_record_sources_only_global(self) -> None:
        collector = MetricsCollector()
        collector.record_sources(3)
        assert collector.metrics.total_sources == 3

    def test_no_steps_record_error_only_global(self) -> None:
        collector = MetricsCollector()
        collector.record_error()
        assert collector.metrics.total_errors == 1
