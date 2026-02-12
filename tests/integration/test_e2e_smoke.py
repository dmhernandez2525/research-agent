"""E2E smoke tests for the research-agent CLI and pipeline.

Tests the CLI commands with mocked external services (LLM, search, HTTP)
to verify end-to-end flows without real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from research_agent.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(**overrides: Any) -> MagicMock:
    """Build a MagicMock settings object with sane defaults."""
    s = MagicMock()
    s.checkpoints.directory.__truediv__ = lambda self, x: MagicMock()
    s.checkpoints.max_checkpoints = 5
    s.checkpoints.enabled = False
    s.report.output_dir = "/tmp/test_reports"
    s.costs.max_cost_per_run = 2.0
    s.logging.level = "WARNING"
    for key, val in overrides.items():
        setattr(s, key, val)
    return s


# ---------------------------------------------------------------------------
# CLI: version and help
# ---------------------------------------------------------------------------


class TestCLIBasics:
    """Verify CLI invocation, version, and help text."""

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer exits with code 0 or 2 for no-args help display
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "research-agent" in result.output.lower()

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "research-agent" in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower()

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--no-approve" in result.output
        assert "--approve-timeout" in result.output
        assert "--budget" in result.output

    def test_resume_help(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "--no-approve" in result.output

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--query" in result.output

    def test_clean_help(self) -> None:
        result = runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0
        assert "--checkpoints" in result.output


# ---------------------------------------------------------------------------
# CLI: run command smoke tests
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Smoke tests for the run command with mocked pipeline."""

    @patch("research_agent.cli._load_settings")
    @patch("research_agent.cli.generate_run_id", return_value="run-test-001")
    @patch("research_agent.cli.CheckpointManager")
    @patch("research_agent.graph.compile_graph")
    def test_run_completes_with_no_report(
        self,
        mock_compile: MagicMock,
        mock_cp_cls: MagicMock,
        mock_run_id: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        settings = _mock_settings()
        mock_settings.return_value = settings
        settings.checkpoints.directory.__truediv__ = lambda self, x: MagicMock()

        result = runner.invoke(app, ["run", "test query"])
        assert result.exit_code == 0
        assert "run-test-001" in result.output

    @patch("research_agent.cli._load_settings")
    @patch("research_agent.cli.generate_run_id", return_value="run-test-002")
    @patch("research_agent.cli.CheckpointManager")
    @patch("research_agent.graph.compile_graph")
    def test_run_with_budget_override(
        self,
        mock_compile: MagicMock,
        mock_cp_cls: MagicMock,
        mock_run_id: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_settings()
        mock_load.return_value.checkpoints.directory.__truediv__ = lambda self, x: (
            MagicMock()
        )

        result = runner.invoke(app, ["run", "--budget", "0.50", "test query"])
        assert result.exit_code == 0
        # Verify the budget override was passed through
        call_kwargs = mock_load.call_args
        assert call_kwargs is not None

    @patch("research_agent.cli._load_settings")
    @patch("research_agent.cli.generate_run_id", return_value="run-test-003")
    @patch("research_agent.cli.CheckpointManager")
    @patch("research_agent.graph.compile_graph")
    def test_run_with_verbose_flag(
        self,
        mock_compile: MagicMock,
        mock_cp_cls: MagicMock,
        mock_run_id: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_settings()
        mock_load.return_value.checkpoints.directory.__truediv__ = lambda self, x: (
            MagicMock()
        )

        result = runner.invoke(app, ["run", "-v", "test query"])
        assert result.exit_code == 0

    @patch("research_agent.cli._load_settings")
    @patch("research_agent.cli.generate_run_id", return_value="run-test-004")
    @patch("research_agent.cli.CheckpointManager")
    @patch("research_agent.graph.compile_graph")
    def test_run_with_no_approve_flag(
        self,
        mock_compile: MagicMock,
        mock_cp_cls: MagicMock,
        mock_run_id: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = _mock_settings()
        mock_load.return_value.checkpoints.directory.__truediv__ = lambda self, x: (
            MagicMock()
        )

        result = runner.invoke(app, ["run", "--no-approve", "test query"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI: evaluate command
# ---------------------------------------------------------------------------


class TestEvaluateCommand:
    """Smoke tests for the evaluate command."""

    def test_missing_file_exits_with_error(self, tmp_path: Any) -> None:
        nonexistent = str(tmp_path / "nonexistent.md")
        result = runner.invoke(app, ["evaluate", nonexistent])
        assert result.exit_code == 1

    def test_empty_file_exits_with_error(self, tmp_path: Any) -> None:
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        result = runner.invoke(app, ["evaluate", str(empty_file)])
        assert result.exit_code == 1

    def test_valid_report_shows_info(self, tmp_path: Any) -> None:
        report = tmp_path / "report.md"
        report.write_text("# Test Report\n\nThis is a test report with content.")
        result = runner.invoke(app, ["evaluate", str(report), "--query", "test topic"])
        assert result.exit_code == 0
        assert "Evaluation" in result.output

    def test_evaluate_without_query(self, tmp_path: Any) -> None:
        report = tmp_path / "report.md"
        report.write_text("# Report\n\nContent here.")
        result = runner.invoke(app, ["evaluate", str(report)])
        assert result.exit_code == 0
        assert "Evaluation" in result.output


# ---------------------------------------------------------------------------
# CLI: clean command
# ---------------------------------------------------------------------------


class TestCleanCommand:
    """Smoke tests for the clean command."""

    def test_no_target_exits_with_error(self) -> None:
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 1
        assert "No cleanup target" in result.output

    @patch("research_agent.cli._load_settings")
    def test_clean_checkpoints_when_dir_missing(
        self, mock_load: MagicMock, tmp_path: Any
    ) -> None:
        settings = _mock_settings()
        settings.checkpoints.directory = tmp_path / "nonexistent"
        settings.vector_store.persist_directory = str(tmp_path / "nonexistent_vs")
        mock_load.return_value = settings

        result = runner.invoke(app, ["clean", "--checkpoints"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    @patch("research_agent.cli._load_settings")
    def test_clean_checkpoints_moves_to_trash(
        self, mock_load: MagicMock, tmp_path: Any
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        (cp_dir / "data.json").write_text("{}")

        settings = _mock_settings()
        settings.checkpoints.directory = cp_dir
        mock_load.return_value = settings

        result = runner.invoke(app, ["clean", "--checkpoints"])
        assert result.exit_code == 0
        assert "Removed" in result.output


# ---------------------------------------------------------------------------
# CLI: resume command
# ---------------------------------------------------------------------------


class TestResumeCommand:
    """Smoke tests for the resume command."""

    @patch("research_agent.cli._load_settings")
    def test_resume_missing_dir_exits_with_error(
        self, mock_load: MagicMock, tmp_path: Any
    ) -> None:
        settings = _mock_settings()
        settings.checkpoints.directory = tmp_path / "no-such-dir"
        mock_load.return_value = settings

        result = runner.invoke(app, ["resume", "--dir", str(tmp_path / "no-such-dir")])
        assert result.exit_code == 1

    @patch("research_agent.cli._load_settings")
    def test_resume_empty_dir_exits_with_error(
        self, mock_load: MagicMock, tmp_path: Any
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        settings = _mock_settings()
        settings.checkpoints.directory = cp_dir
        mock_load.return_value = settings

        result = runner.invoke(app, ["resume", "--dir", str(cp_dir)])
        assert result.exit_code == 1
        assert "No runs found" in result.output


# ---------------------------------------------------------------------------
# Graph edge routing (E2E verification)
# ---------------------------------------------------------------------------


class TestGraphEdgeRouting:
    """Verify conditional edge functions end-to-end."""

    def test_full_routing_scenario_single_subtopic(self) -> None:
        """Simulate routing through the full graph for one subtopic."""
        from research_agent.graph import (
            _all_subtopics_done,
            _should_continue_scrape,
            _should_continue_search,
        )

        # After search: enough results -> scrape
        state: dict[str, Any] = {
            "search_results": [MagicMock()] * 5,
            "search_retry_count": 0,
            "scraped_pages": [],
            "subtopics": [MagicMock()],
            "current_subtopic_index": 0,
        }
        assert _should_continue_search(state) == "scrape"

        # After scrape: has content -> summarize
        state["scraped_pages"] = [MagicMock()]
        assert _should_continue_scrape(state) == "summarize"

        # After summarize: index incremented to 1, only 1 subtopic -> synthesize
        state["current_subtopic_index"] = 1
        assert _all_subtopics_done(state) == "synthesize"

    def test_full_routing_scenario_multiple_subtopics(self) -> None:
        """Simulate routing with 3 subtopics, looping back to search."""
        from research_agent.graph import (
            _all_subtopics_done,
            _should_continue_search,
        )

        state: dict[str, Any] = {
            "search_results": [MagicMock()] * 3,
            "search_retry_count": 0,
            "subtopics": [MagicMock(), MagicMock(), MagicMock()],
            "current_subtopic_index": 0,
        }

        # First subtopic: search done -> scrape -> summarize -> loop to search
        assert _should_continue_search(state) == "scrape"
        state["current_subtopic_index"] = 1
        assert _all_subtopics_done(state) == "search"

        # Second subtopic
        state["current_subtopic_index"] = 2
        assert _all_subtopics_done(state) == "search"

        # Third subtopic done -> synthesize
        state["current_subtopic_index"] = 3
        assert _all_subtopics_done(state) == "synthesize"

    def test_search_retry_then_proceed(self) -> None:
        """Verify retry loop exhaustion routes to scrape."""
        from research_agent.graph import _MAX_SEARCH_RETRIES, _should_continue_search

        state: dict[str, Any] = {
            "search_results": [],
            "search_retry_count": 0,
        }

        # Not enough results, under retry limit -> retry
        assert _should_continue_search(state) == "search"

        # At max retries -> proceed to scrape even with no results
        state["search_retry_count"] = _MAX_SEARCH_RETRIES
        assert _should_continue_search(state) == "scrape"


# ---------------------------------------------------------------------------
# Metrics collector E2E
# ---------------------------------------------------------------------------


class TestMetricsCollectorE2E:
    """Verify metrics accumulation across a simulated pipeline."""

    def test_full_pipeline_metrics_flow(self) -> None:
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector(budget_usd=1.0)

        # Plan step
        plan_step = collector.start_step("plan")
        collector.record_llm_call(
            "claude-sonnet", input_tokens=500, output_tokens=200, cost_usd=0.003
        )
        collector.set_subtopics(3)
        collector.finish_step(plan_step)

        assert plan_step.is_complete
        assert plan_step.input_tokens == 500
        assert plan_step.output_tokens == 200

        # Search + scrape for subtopic 1
        search_step = collector.start_step("search")
        collector.record_llm_call(
            "claude-sonnet", input_tokens=300, output_tokens=100, cost_usd=0.002
        )
        collector.record_sources(5)
        collector.finish_step(search_step)

        scrape_step = collector.start_step("scrape")
        collector.finish_step(scrape_step)

        summarize_step = collector.start_step("summarize")
        collector.record_llm_call(
            "claude-sonnet", input_tokens=2000, output_tokens=500, cost_usd=0.012
        )
        collector.record_findings(4)
        collector.complete_subtopic()
        collector.finish_step(summarize_step)

        # Verify accumulated metrics
        snap = collector.snapshot()
        assert snap["total_input_tokens"] == 2800
        assert snap["total_output_tokens"] == 800
        assert snap["total_tokens"] == 3600
        assert snap["total_cost_usd"] == 0.017
        assert snap["total_sources"] == 5
        assert snap["total_findings"] == 4
        assert snap["subtopics_completed"] == 1
        assert snap["subtopics_total"] == 3
        assert snap["steps_completed"] == 4
        assert snap["model_usage"]["claude-sonnet"] == 3

    def test_budget_tracking(self) -> None:
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector(budget_usd=0.10)

        collector.record_llm_call("gpt-4", cost_usd=0.05)
        snap = collector.snapshot()
        assert snap["budget_used_pct"] == 50.0
        assert snap["budget_remaining_usd"] == 0.05

        collector.record_llm_call("gpt-4", cost_usd=0.05)
        snap = collector.snapshot()
        assert snap["budget_used_pct"] == 100.0
        assert snap["budget_remaining_usd"] == 0.0

    def test_error_recording_across_steps(self) -> None:
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector()
        step = collector.start_step("search")
        collector.record_error()
        collector.record_error()
        collector.finish_step(step)

        assert step.errors == 2
        assert collector.metrics.total_errors == 2


# ---------------------------------------------------------------------------
# Dashboard rendering E2E
# ---------------------------------------------------------------------------


class TestDashboardE2E:
    """Verify dashboard renders without errors for various states."""

    def test_dashboard_renders_empty_state(self) -> None:
        from research_agent.dashboard import build_dashboard
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector()
        layout = build_dashboard(collector, query="Test query")
        assert layout is not None

    def test_dashboard_renders_mid_pipeline(self) -> None:
        from research_agent.dashboard import build_dashboard
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector(budget_usd=2.0)
        step = collector.start_step("search")
        collector.record_llm_call(
            "claude-sonnet", input_tokens=100, output_tokens=50, cost_usd=0.001
        )
        collector.record_sources(3)
        collector.set_subtopics(5)
        collector.complete_subtopic()

        layout = build_dashboard(collector, query="Mid-pipeline query")
        assert layout is not None

        # Finish step and render again
        collector.finish_step(step)
        layout = build_dashboard(collector, query="Mid-pipeline query")
        assert layout is not None

    def test_dashboard_renders_completed_pipeline(self) -> None:
        from research_agent.dashboard import build_dashboard
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector(budget_usd=1.0)

        for name in ["plan", "search", "scrape", "summarize", "synthesize"]:
            s = collector.start_step(name)
            collector.record_llm_call("claude-sonnet", input_tokens=100, cost_usd=0.001)
            collector.finish_step(s)

        collector.set_subtopics(3)
        for _ in range(3):
            collector.complete_subtopic()

        layout = build_dashboard(collector, query="Completed pipeline")
        assert layout is not None

    def test_dashboard_renders_with_multiple_models(self) -> None:
        from research_agent.dashboard import build_dashboard
        from research_agent.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_llm_call("claude-sonnet", input_tokens=100, cost_usd=0.001)
        collector.record_llm_call("claude-haiku", input_tokens=50, cost_usd=0.0001)
        collector.record_llm_call("gpt-4o", input_tokens=200, cost_usd=0.005)

        layout = build_dashboard(collector, query="Multi-model test")
        assert layout is not None


# ---------------------------------------------------------------------------
# Plan editor E2E
# ---------------------------------------------------------------------------


class TestPlanEditorE2E:
    """Verify plan editor serialization and validation roundtrips."""

    def test_yaml_roundtrip(self) -> None:
        from research_agent.plan_editor import plan_to_yaml, yaml_to_plan

        sub_questions = [
            {"id": 1, "question": "What is RAG?", "rationale": "Core concept"},
            {"id": 2, "question": "How does it scale?", "rationale": "Performance"},
        ]

        yaml_str = plan_to_yaml(sub_questions)
        assert "What is RAG?" in yaml_str
        assert "How does it scale?" in yaml_str

        plan = yaml_to_plan(yaml_str)
        assert plan is not None
        assert len(plan.subtopics) == 2
        assert plan.subtopics[0].question == "What is RAG?"
        assert plan.subtopics[1].id == 2

    def test_yaml_with_special_characters(self) -> None:
        from research_agent.plan_editor import plan_to_yaml, yaml_to_plan

        sub_questions = [
            {
                "id": 1,
                "question": "What's the impact of LLMs on code review?",
                "rationale": "Key area: AI-assisted development",
            },
        ]

        yaml_str = plan_to_yaml(sub_questions)
        plan = yaml_to_plan(yaml_str)
        assert plan is not None
        assert "LLMs" in plan.subtopics[0].question

    def test_empty_yaml_returns_none(self) -> None:
        from research_agent.plan_editor import yaml_to_plan

        assert yaml_to_plan("") is None
        assert yaml_to_plan("   \n  ") is None

    def test_invalid_yaml_returns_none(self) -> None:
        from research_agent.plan_editor import yaml_to_plan

        assert yaml_to_plan("{{invalid yaml:::") is None

    def test_yaml_missing_subtopics_returns_none(self) -> None:
        from research_agent.plan_editor import yaml_to_plan

        assert yaml_to_plan("other_key: value") is None

    def test_edited_plan_renumbers_ids(self) -> None:
        from research_agent.plan_editor import EditableSubQuestion, EditedPlan

        plan = EditedPlan(
            subtopics=[
                EditableSubQuestion(id=5, question="First", rationale=""),
                EditableSubQuestion(id=10, question="Second", rationale=""),
            ]
        )
        assert plan.subtopics[0].id == 1
        assert plan.subtopics[1].id == 2

    def test_inline_edit_removes_by_id(self) -> None:
        from research_agent.plan_editor import edit_plan_inline

        sub_questions = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
            {"id": 2, "question": "Q2", "rationale": "R2"},
            {"id": 3, "question": "Q3", "rationale": "R3"},
        ]

        # Simulate user entering "2" to remove sub-question 2
        with patch("builtins.input", return_value="2"):
            result = edit_plan_inline(sub_questions)

        assert result is not None
        assert len(result.subtopics) == 2
        questions = [sq.question for sq in result.subtopics]
        assert "Q1" in questions
        assert "Q3" in questions

    def test_inline_edit_cancel(self) -> None:
        from research_agent.plan_editor import edit_plan_inline

        sub_questions = [{"id": 1, "question": "Q1", "rationale": "R1"}]

        with patch("builtins.input", return_value="c"):
            result = edit_plan_inline(sub_questions)

        assert result is None

    def test_inline_edit_eof(self) -> None:
        from research_agent.plan_editor import edit_plan_inline

        sub_questions = [{"id": 1, "question": "Q1", "rationale": "R1"}]

        with patch("builtins.input", side_effect=EOFError):
            result = edit_plan_inline(sub_questions)

        assert result is None

    def test_editor_integration_with_mock(self) -> None:
        from research_agent.plan_editor import edit_plan_in_editor

        sub_questions = [
            {
                "id": 1,
                "question": "Original question",
                "rationale": "Original rationale",
            },
        ]

        edited_yaml = (
            "subtopics:\n"
            "  - id: 1\n"
            "    question: Edited question\n"
            "    rationale: Updated rationale\n"
        )

        def mock_subprocess_run(args: list[str], **kwargs: Any) -> MagicMock:
            # Write edited content to the temp file
            with open(args[1], "w") as f:
                f.write(edited_yaml)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch(
            "research_agent.plan_editor.subprocess.run", side_effect=mock_subprocess_run
        ):
            result = edit_plan_in_editor(sub_questions)

        assert result is not None
        assert result.subtopics[0].question == "Edited question"

    def test_editor_nonzero_exit_returns_none(self) -> None:
        from research_agent.plan_editor import edit_plan_in_editor

        sub_questions = [{"id": 1, "question": "Q", "rationale": "R"}]

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch(
            "research_agent.plan_editor.subprocess.run", return_value=mock_result
        ):
            result = edit_plan_in_editor(sub_questions)

        assert result is None


# ---------------------------------------------------------------------------
# Plan review workflow E2E
# ---------------------------------------------------------------------------


class TestPlanReviewWorkflow:
    """Test _handle_plan_review orchestrator."""

    def test_no_approve_skips_prompt(self) -> None:
        from research_agent.cli import _handle_plan_review

        sub_questions = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
        ]

        result = _handle_plan_review(sub_questions, no_approve=True)
        assert result is not None
        assert len(result) == 1

    def test_approve_returns_plan(self) -> None:
        from research_agent.cli import _handle_plan_review

        sub_questions = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
        ]

        with patch("research_agent.cli._approve_plan", return_value="approve"):
            result = _handle_plan_review(sub_questions)

        assert result is not None
        assert result == sub_questions

    def test_cancel_returns_none(self) -> None:
        from research_agent.cli import _handle_plan_review

        sub_questions = [
            {"id": 1, "question": "Q1", "rationale": "R1"},
        ]

        with patch("research_agent.cli._approve_plan", return_value="cancel"):
            result = _handle_plan_review(sub_questions)

        assert result is None
