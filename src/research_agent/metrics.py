"""Pipeline run metrics collection and aggregation.

Accumulates real-time statistics during a research pipeline run,
including token usage, cost, source counts, error counts, and
per-step timing. Used by the dashboard for live display.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StepMetric(BaseModel):
    """Metrics for a single pipeline step execution."""

    step_name: str = Field(description="Name of the pipeline step.")
    started_at: float = Field(default_factory=time.monotonic)
    finished_at: float | None = Field(default=None)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    sources_found: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)

    @property
    def duration_seconds(self) -> float:
        """Elapsed time in seconds, or time since start if still running."""
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def is_complete(self) -> bool:
        """Whether the step has finished."""
        return self.finished_at is not None


class RunMetrics(BaseModel):
    """Aggregated metrics for an entire research pipeline run."""

    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_sources: int = Field(default=0, ge=0)
    total_errors: int = Field(default=0, ge=0)
    total_findings: int = Field(default=0, ge=0)
    subtopics_completed: int = Field(default=0, ge=0)
    subtopics_total: int = Field(default=0, ge=0)
    budget_usd: float = Field(default=2.0, ge=0.0)
    current_step: str = Field(default="idle")
    model_usage: dict[str, int] = Field(
        default_factory=dict, description="Call counts per model."
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def budget_used_pct(self) -> float:
        """Percentage of budget consumed."""
        if self.budget_usd <= 0:
            return 100.0
        return min(100.0, (self.total_cost_usd / self.budget_usd) * 100)

    @property
    def budget_remaining_usd(self) -> float:
        """Remaining budget in USD."""
        return max(0.0, self.budget_usd - self.total_cost_usd)

    @property
    def subtopic_progress_pct(self) -> float:
        """Subtopic completion percentage."""
        if self.subtopics_total <= 0:
            return 0.0
        return min(100.0, (self.subtopics_completed / self.subtopics_total) * 100)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Collects and aggregates pipeline run metrics in real time.

    Thread-safe accumulation of metrics from pipeline step callbacks.

    Attributes:
        metrics: Current aggregated metrics snapshot.
        steps: Per-step metric history.
    """

    def __init__(self, budget_usd: float = 2.0) -> None:
        """Initialize the metrics collector.

        Args:
            budget_usd: Total budget for the run.
        """
        self._metrics = RunMetrics(budget_usd=budget_usd)
        self._steps: list[StepMetric] = []
        self._start_time = time.monotonic()

    @property
    def metrics(self) -> RunMetrics:
        """Current metrics snapshot."""
        return self._metrics

    @property
    def steps(self) -> list[StepMetric]:
        """List of per-step metrics."""
        return list(self._steps)

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time since collection started."""
        return time.monotonic() - self._start_time

    def start_step(self, step_name: str) -> StepMetric:
        """Record the start of a pipeline step.

        Args:
            step_name: Name of the step (e.g. "plan", "search", "scrape").

        Returns:
            The created StepMetric instance.
        """
        step = StepMetric(step_name=step_name)
        self._steps.append(step)
        self._metrics.current_step = step_name
        logger.debug("metrics_step_started", step=step_name)
        return step

    def finish_step(self, step: StepMetric) -> None:
        """Record the completion of a pipeline step.

        Args:
            step: The StepMetric to finalize.
        """
        step.finished_at = time.monotonic()
        logger.debug(
            "metrics_step_finished",
            step=step.step_name,
            duration=round(step.duration_seconds, 2),
        )

    def record_llm_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record an LLM API call.

        Args:
            model: Model identifier.
            input_tokens: Input tokens consumed.
            output_tokens: Output tokens generated.
            cost_usd: Cost of this call in USD.
        """
        self._metrics.total_input_tokens += input_tokens
        self._metrics.total_output_tokens += output_tokens
        self._metrics.total_cost_usd += cost_usd
        self._metrics.model_usage[model] = self._metrics.model_usage.get(model, 0) + 1

        # Update current step if exists
        if self._steps:
            current = self._steps[-1]
            if not current.is_complete:
                current.input_tokens += input_tokens
                current.output_tokens += output_tokens
                current.cost_usd += cost_usd

    def record_sources(self, count: int) -> None:
        """Record discovered sources.

        Args:
            count: Number of new sources found.
        """
        self._metrics.total_sources += count
        if self._steps:
            current = self._steps[-1]
            if not current.is_complete:
                current.sources_found += count

    def record_findings(self, count: int) -> None:
        """Record extracted findings.

        Args:
            count: Number of key findings extracted.
        """
        self._metrics.total_findings += count

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._metrics.total_errors += 1
        if self._steps:
            current = self._steps[-1]
            if not current.is_complete:
                current.errors += 1

    def set_subtopics(self, total: int) -> None:
        """Set the total number of subtopics.

        Args:
            total: Total subtopic count from the planner.
        """
        self._metrics.subtopics_total = total

    def complete_subtopic(self) -> None:
        """Mark a subtopic as completed."""
        self._metrics.subtopics_completed += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of current metrics.

        Returns:
            Dictionary of all current metrics.
        """
        m = self._metrics
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "current_step": m.current_step,
            "total_tokens": m.total_tokens,
            "total_input_tokens": m.total_input_tokens,
            "total_output_tokens": m.total_output_tokens,
            "total_cost_usd": round(m.total_cost_usd, 4),
            "budget_used_pct": round(m.budget_used_pct, 1),
            "budget_remaining_usd": round(m.budget_remaining_usd, 4),
            "total_sources": m.total_sources,
            "total_findings": m.total_findings,
            "total_errors": m.total_errors,
            "subtopics_completed": m.subtopics_completed,
            "subtopics_total": m.subtopics_total,
            "subtopic_progress_pct": round(m.subtopic_progress_pct, 1),
            "model_usage": dict(m.model_usage),
            "steps_completed": sum(1 for s in self._steps if s.is_complete),
        }
