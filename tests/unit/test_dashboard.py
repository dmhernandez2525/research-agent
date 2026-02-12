"""Unit tests for research_agent.dashboard - Rich live dashboard."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from research_agent.dashboard import (
    _build_header,
    _build_metrics_table,
    _build_model_usage,
    _build_steps_table,
    _build_subtopic_progress,
    build_dashboard,
)
from research_agent.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _render_to_str(renderable: object) -> str:
    """Render a Rich object to a plain string for assertion checks."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    console.print(renderable)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _build_header tests
# ---------------------------------------------------------------------------


class TestBuildHeader:
    """Header panel rendering."""

    def test_contains_research_agent(self) -> None:
        panel = _build_header("test query", "plan")
        text = _render_to_str(panel)
        assert "Research Agent" in text

    def test_contains_step_name(self) -> None:
        panel = _build_header("my query", "search")
        text = _render_to_str(panel)
        assert "search" in text

    def test_contains_query_text(self) -> None:
        panel = _build_header("latest advances in RAG", "plan")
        text = _render_to_str(panel)
        assert "latest advances in RAG" in text

    def test_empty_query(self) -> None:
        panel = _build_header("", "idle")
        text = _render_to_str(panel)
        assert "Dashboard" in text


# ---------------------------------------------------------------------------
# _build_metrics_table tests
# ---------------------------------------------------------------------------


class TestBuildMetricsTable:
    """Metrics table rendering."""

    def test_contains_elapsed(self) -> None:
        collector = MetricsCollector()
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "Elapsed" in text

    def test_contains_tokens(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("m", input_tokens=100, output_tokens=50)
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "Tokens" in text
        assert "150" in text

    def test_contains_cost(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("m", cost_usd=0.0123)
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "Cost" in text
        assert "0.0123" in text

    def test_contains_budget_used(self) -> None:
        collector = MetricsCollector(budget_usd=1.0)
        collector.record_llm_call("m", cost_usd=0.5)
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "50.0%" in text

    def test_contains_sources_and_findings(self) -> None:
        collector = MetricsCollector()
        collector.record_sources(5)
        collector.record_findings(3)
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "5" in text
        assert "3" in text

    def test_contains_errors(self) -> None:
        collector = MetricsCollector()
        collector.record_error()
        table = _build_metrics_table(collector)
        text = _render_to_str(table)
        assert "Errors" in text


# ---------------------------------------------------------------------------
# _build_subtopic_progress tests
# ---------------------------------------------------------------------------


class TestBuildSubtopicProgress:
    """Subtopic progress panel rendering."""

    def test_renders_with_no_subtopics(self) -> None:
        collector = MetricsCollector()
        panel = _build_subtopic_progress(collector)
        text = _render_to_str(panel)
        assert "Progress" in text

    def test_renders_with_partial_progress(self) -> None:
        collector = MetricsCollector()
        collector.set_subtopics(4)
        collector.complete_subtopic()
        collector.complete_subtopic()
        panel = _build_subtopic_progress(collector)
        text = _render_to_str(panel)
        assert "Subtopics" in text

    def test_renders_with_full_progress(self) -> None:
        collector = MetricsCollector()
        collector.set_subtopics(2)
        collector.complete_subtopic()
        collector.complete_subtopic()
        panel = _build_subtopic_progress(collector)
        text = _render_to_str(panel)
        assert "100%" in text


# ---------------------------------------------------------------------------
# _build_model_usage tests
# ---------------------------------------------------------------------------


class TestBuildModelUsage:
    """Model usage panel rendering."""

    def test_no_calls_shows_placeholder(self) -> None:
        collector = MetricsCollector()
        panel = _build_model_usage(collector)
        text = _render_to_str(panel)
        assert "no calls yet" in text

    def test_single_model_shown(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("claude-sonnet")
        panel = _build_model_usage(collector)
        text = _render_to_str(panel)
        assert "claude-sonnet" in text

    def test_multiple_models_shown(self) -> None:
        collector = MetricsCollector()
        collector.record_llm_call("claude-sonnet")
        collector.record_llm_call("claude-haiku")
        collector.record_llm_call("claude-sonnet")
        panel = _build_model_usage(collector)
        text = _render_to_str(panel)
        assert "claude-sonnet" in text
        assert "claude-haiku" in text

    def test_shows_call_count(self) -> None:
        collector = MetricsCollector()
        for _ in range(5):
            collector.record_llm_call("gpt-4o")
        panel = _build_model_usage(collector)
        text = _render_to_str(panel)
        assert "5" in text


# ---------------------------------------------------------------------------
# _build_steps_table tests
# ---------------------------------------------------------------------------


class TestBuildStepsTable:
    """Pipeline steps panel rendering."""

    def test_no_steps_shows_waiting(self) -> None:
        collector = MetricsCollector()
        panel = _build_steps_table(collector)
        text = _render_to_str(panel)
        assert "waiting" in text

    def test_running_step_shows_running(self) -> None:
        collector = MetricsCollector()
        collector.start_step("search")
        panel = _build_steps_table(collector)
        text = _render_to_str(panel)
        assert "search" in text
        assert "Running" in text

    def test_completed_step_shows_done(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("plan")
        collector.finish_step(step)
        panel = _build_steps_table(collector)
        text = _render_to_str(panel)
        assert "plan" in text
        assert "Done" in text

    def test_multiple_steps_listed(self) -> None:
        collector = MetricsCollector()
        for name in ["plan", "search", "scrape"]:
            s = collector.start_step(name)
            collector.finish_step(s)
        panel = _build_steps_table(collector)
        text = _render_to_str(panel)
        assert "plan" in text
        assert "search" in text
        assert "scrape" in text

    def test_step_tokens_displayed(self) -> None:
        collector = MetricsCollector()
        step = collector.start_step("plan")
        collector.record_llm_call("m", input_tokens=500, output_tokens=200)
        collector.finish_step(step)
        panel = _build_steps_table(collector)
        text = _render_to_str(panel)
        assert "700" in text


# ---------------------------------------------------------------------------
# build_dashboard tests
# ---------------------------------------------------------------------------


class TestBuildDashboard:
    """Full dashboard layout construction."""

    def test_returns_layout(self) -> None:
        collector = MetricsCollector()
        layout = build_dashboard(collector, query="test")
        assert layout is not None

    def test_layout_has_named_regions(self) -> None:
        collector = MetricsCollector()
        layout = build_dashboard(collector, query="test")
        # Layout should have header, body, footer regions
        assert layout["header"] is not None
        assert layout["body"] is not None
        assert layout["footer"] is not None

    def test_layout_renders_without_error(self) -> None:
        collector = MetricsCollector()
        collector.start_step("plan")
        collector.record_llm_call("m", input_tokens=100, cost_usd=0.01)
        layout = build_dashboard(collector, query="test query")
        text = _render_to_str(layout)
        assert len(text) > 0

    def test_empty_query_renders(self) -> None:
        collector = MetricsCollector()
        layout = build_dashboard(collector)
        text = _render_to_str(layout)
        assert "Dashboard" in text
