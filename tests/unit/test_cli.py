"""Unit tests for research_agent.cli - argument parsing, version, commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import click
import pytest
import typer
from typer.testing import CliRunner

from research_agent import __version__
from research_agent.cli import (
    _approve_plan,
    _create_progress,
    _display_error_with_resume,
    _display_plan,
    _handle_plan_review,
    _handle_sigint,
    app,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# ---- Version and help -------------------------------------------------------


class TestVersionAndHelp:
    """Version flag and help text output."""

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_short_version_flag(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer returns exit code 2 for no_args_is_help
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "run" in result.output

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "resume" in result.output
        assert "evaluate" in result.output
        assert "clean" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--output" in result.output
        assert "--resume" in result.output
        assert "--budget" in result.output
        assert "--verbose" in result.output

    def test_resume_help(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "--dir" in result.output

    def test_evaluate_help(self) -> None:
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--query" in result.output

    def test_clean_help(self) -> None:
        result = runner.invoke(app, ["clean", "--help"])
        assert result.exit_code == 0
        assert "--checkpoints" in result.output
        assert "--cache" in result.output
        assert "--all" in result.output


# ---- Run command -------------------------------------------------------------


class TestRunCommand:
    """The `run` command argument parsing and basic behavior."""

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_with_query(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["run", "What is RAG?"])
        assert result.exit_code == 0
        assert "Research Agent" in result.output

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_shows_run_id(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["run", "test query"])
        assert result.exit_code == 0
        assert "run-" in result.output

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_with_budget(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["run", "test", "--budget", "0.50"])
        assert result.exit_code == 0
        mock_settings.assert_called_once()
        call_kwargs = mock_settings.call_args
        assert call_kwargs[1]["costs"] == {"max_cost_per_run": 0.50}

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_with_verbose(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["run", "test", "--verbose"])
        assert result.exit_code == 0
        call_kwargs = mock_settings.call_args
        assert call_kwargs[1]["logging"] == {"level": "DEBUG"}

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_with_resume_flag_no_checkpoint(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["run", "test", "--resume"])
        assert result.exit_code == 0
        assert "starting fresh" in result.output.lower() or "No report" in result.output

    @patch("research_agent.graph.compile_graph", side_effect=RuntimeError("boom"))
    @patch("research_agent.cli._load_settings")
    def test_run_error_shows_resume_instructions(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        mock_settings.return_value = settings

        result = runner.invoke(app, ["run", "test"])
        assert result.exit_code == 1


# ---- Resume command ----------------------------------------------------------


class TestResumeCommand:
    """The `resume` command behavior."""

    @patch("research_agent.cli._load_settings")
    def test_resume_missing_dir(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "nonexistent"
        mock_settings.return_value = settings

        result = runner.invoke(app, ["resume"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("research_agent.cli._load_settings")
    def test_resume_no_runs(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        settings = MagicMock()
        settings.checkpoints.directory = cp_dir
        mock_settings.return_value = settings

        result = runner.invoke(app, ["resume"])
        assert result.exit_code == 1
        assert "No runs" in result.output

    @patch("research_agent.cli._load_settings")
    def test_resume_no_checkpoints_in_run(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        (cp_dir / "run-abc123").mkdir()

        settings = MagicMock()
        settings.checkpoints.directory = cp_dir
        mock_settings.return_value = settings

        result = runner.invoke(app, ["resume"])
        assert result.exit_code == 1
        assert "No checkpoints" in result.output

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_resume_with_valid_checkpoint(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        from research_agent.checkpoints import CheckpointManager

        cp_dir = tmp_path / "checkpoints"
        run_dir = cp_dir / "run-test123"
        run_dir.mkdir(parents=True)

        mgr = CheckpointManager(directory=run_dir)
        mgr.save("run-test123-step-1", {"query": "test", "step": "plan"}, step_index=1)

        settings = MagicMock()
        settings.checkpoints.directory = cp_dir
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        result = runner.invoke(app, ["resume", "--dir", str(cp_dir)])
        assert result.exit_code == 0
        assert "Resume" in result.output or "Resuming" in result.output


# ---- Clean command -----------------------------------------------------------


class TestCleanCommand:
    """The `clean` command behavior."""

    def test_clean_no_flags_exits_with_error(self) -> None:
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 1
        assert "No cleanup target" in result.output

    @patch("research_agent.cli._load_settings")
    def test_clean_checkpoints(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        (cp_dir / "test.json").write_text("{}")

        settings = MagicMock()
        settings.checkpoints.directory = cp_dir
        settings.vector_store.persist_directory = str(tmp_path / "nonexistent")
        mock_settings.return_value = settings

        result = runner.invoke(app, ["clean", "--checkpoints"])
        assert result.exit_code == 0
        assert "Checkpoints" in result.output
        assert "Trash" in result.output

    @patch("research_agent.cli._load_settings")
    def test_clean_cache(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        cache_dir = tmp_path / "chromadb"
        cache_dir.mkdir()
        (cache_dir / "data.bin").write_text("x")

        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "nonexistent"
        settings.vector_store.persist_directory = str(cache_dir)
        mock_settings.return_value = settings

        result = runner.invoke(app, ["clean", "--cache"])
        assert result.exit_code == 0
        assert "Cache" in result.output

    @patch("research_agent.cli._load_settings")
    def test_clean_nothing_to_clean(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "nonexistent"
        settings.vector_store.persist_directory = str(tmp_path / "also-nonexistent")
        mock_settings.return_value = settings

        result = runner.invoke(app, ["clean", "--all"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output


# ---- Evaluate command --------------------------------------------------------


class TestEvaluateCommand:
    """The `evaluate` command behavior."""

    def test_evaluate_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["evaluate", str(tmp_path / "nonexistent.md")])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_evaluate_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.md"
        empty.write_text("")
        result = runner.invoke(app, ["evaluate", str(empty)])
        assert result.exit_code == 1
        assert "empty" in result.output

    def test_evaluate_valid_report(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        report.write_text("# Report\n\nSome findings here.")
        result = runner.invoke(app, ["evaluate", str(report), "--query", "test query"])
        assert result.exit_code == 0
        assert "Evaluation" in result.output

    def test_evaluate_without_query(self, tmp_path: Path) -> None:
        report = tmp_path / "report.md"
        report.write_text("# Report\n\nContent.")
        result = runner.invoke(app, ["evaluate", str(report)])
        assert result.exit_code == 0
        assert "Evaluation" in result.output


# ---- Signal handler ----------------------------------------------------------


class TestSignalHandler:
    """Graceful Ctrl+C handling."""

    def test_first_interrupt_no_state(self) -> None:
        import research_agent.cli as cli_mod

        cli_mod._interrupt_count = 0
        cli_mod._checkpoint_mgr = None
        cli_mod._current_state = None
        cli_mod._current_run_id = None

        _handle_sigint(2, None)
        assert cli_mod._interrupt_count == 1

    def test_first_interrupt_with_state_saves(self, tmp_path: Path) -> None:
        import research_agent.cli as cli_mod
        from research_agent.checkpoints import CheckpointManager

        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=10)
        cli_mod._interrupt_count = 0
        cli_mod._checkpoint_mgr = mgr
        cli_mod._current_state = {"step": "search", "step_index": 2}
        cli_mod._current_run_id = "run-test123"

        _handle_sigint(2, None)
        assert cli_mod._interrupt_count == 1
        # Checkpoint should have been saved
        assert mgr.latest() is not None

        # Cleanup
        cli_mod._checkpoint_mgr = None
        cli_mod._current_state = None
        cli_mod._current_run_id = None

    def test_first_interrupt_save_failure(self) -> None:
        import research_agent.cli as cli_mod

        mock_mgr = MagicMock()
        mock_mgr.save.side_effect = OSError("disk full")
        cli_mod._interrupt_count = 0
        cli_mod._checkpoint_mgr = mock_mgr
        cli_mod._current_state = {"step": "plan", "step_index": 0}
        cli_mod._current_run_id = "run-fail"

        # Should not raise, just print error
        _handle_sigint(2, None)
        assert cli_mod._interrupt_count == 1

        cli_mod._checkpoint_mgr = None
        cli_mod._current_state = None
        cli_mod._current_run_id = None

    def test_second_interrupt_exits(self) -> None:
        import research_agent.cli as cli_mod

        cli_mod._interrupt_count = 1

        with pytest.raises(SystemExit) as exc_info:
            _handle_sigint(2, None)
        assert exc_info.value.code == 130

        cli_mod._interrupt_count = 0


# ---- Approve plan ------------------------------------------------------------


class TestApprovePlan:
    """Plan approval workflow."""

    def test_approve_returns_approve(self) -> None:
        from research_agent.cli import _approve_plan

        with patch("research_agent.cli.typer.prompt", return_value="a"):
            assert _approve_plan() == "approve"

    def test_approve_returns_edit(self) -> None:
        from research_agent.cli import _approve_plan

        with patch("research_agent.cli.typer.prompt", return_value="e"):
            assert _approve_plan() == "edit"

    def test_approve_returns_cancel(self) -> None:
        from research_agent.cli import _approve_plan

        with patch("research_agent.cli.typer.prompt", return_value="c"):
            assert _approve_plan() == "cancel"

    def test_approve_full_word(self) -> None:
        from research_agent.cli import _approve_plan

        with patch("research_agent.cli.typer.prompt", return_value="approve"):
            assert _approve_plan() == "approve"


# ---- Load settings -----------------------------------------------------------


class TestLoadSettings:
    """Settings loading with error handling."""

    @patch("research_agent.cli.Settings.load")
    def test_load_settings_success(self, mock_load: MagicMock) -> None:
        from research_agent.cli import _load_settings

        mock_load.return_value = MagicMock()
        result = _load_settings()
        assert result is not None

    @patch("research_agent.cli.Settings.load")
    def test_load_settings_validation_error(self, mock_load: MagicMock) -> None:
        from pydantic import ValidationError

        from research_agent.cli import _load_settings

        mock_load.side_effect = ValidationError.from_exception_data(
            "Settings",
            [
                {
                    "type": "missing",
                    "loc": ("llm", "model"),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )
        with pytest.raises(click.exceptions.Exit):
            _load_settings()


# ---- Progress helper ---------------------------------------------------------


class TestProgressHelper:
    """Rich progress bar creation."""

    def test_create_progress_returns_progress_object(self) -> None:
        progress = _create_progress()
        assert progress is not None

    def test_progress_has_spinner(self) -> None:
        progress = _create_progress()
        column_types = [type(c).__name__ for c in progress.columns]
        assert "SpinnerColumn" in column_types

    def test_progress_has_bar(self) -> None:
        progress = _create_progress()
        column_types = [type(c).__name__ for c in progress.columns]
        assert "BarColumn" in column_types

    def test_progress_has_time(self) -> None:
        progress = _create_progress()
        column_types = [type(c).__name__ for c in progress.columns]
        assert "TimeElapsedColumn" in column_types


# ---- Plan display ------------------------------------------------------------


class TestPlanDisplay:
    """Plan approval workflow display."""

    def test_display_plan_no_error(self) -> None:
        subtopics = [
            {"id": 1, "question": "What is RAG?", "rationale": "Core concept"},
            {"id": 2, "question": "How does it work?", "rationale": "Implementation"},
        ]
        _display_plan(subtopics)

    def test_display_plan_empty(self) -> None:
        _display_plan([])

    def test_display_plan_missing_fields(self) -> None:
        _display_plan([{"id": 1}])


# ---- Error display -----------------------------------------------------------


class TestErrorDisplay:
    """Error display with resume instructions."""

    def test_error_display_with_run_id(self) -> None:
        exc = RuntimeError("Something failed")
        _display_error_with_resume(exc, run_id="run-abc123")

    def test_error_display_without_run_id(self) -> None:
        exc = RuntimeError("Something failed")
        _display_error_with_resume(exc, run_id=None)


# ---- Handle plan review edit flow -------------------------------------------


class TestHandlePlanReviewEdit:
    """Test the edit flow in _handle_plan_review (lines 206-232)."""

    def test_edit_then_approve(self) -> None:
        """When user chooses 'edit' then 'approve', the edited subtopics are
        returned after being processed by edit_plan_in_editor."""
        from research_agent.plan_editor import EditableSubQuestion, EditedPlan

        subtopics = [
            {"id": 1, "question": "What is RAG?", "rationale": "Core concept"},
        ]
        edited_plan = EditedPlan(
            subtopics=[
                EditableSubQuestion(
                    id=1, question="What is RAG architecture?", rationale="Updated"
                ),
                EditableSubQuestion(
                    id=2, question="New subtopic", rationale="Added"
                ),
            ]
        )

        with (
            patch(
                "research_agent.cli._approve_plan", side_effect=["edit", "approve"]
            ),
            patch(
                "research_agent.cli.edit_plan_in_editor", return_value=edited_plan
            ),
        ):
            result = _handle_plan_review(subtopics)

        assert result is not None
        assert len(result) == 2
        assert result[0]["question"] == "What is RAG architecture?"
        assert result[1]["question"] == "New subtopic"

    def test_edit_editor_fails_falls_back_to_inline(self) -> None:
        """When edit_plan_in_editor returns None, fall back to edit_plan_inline."""
        from research_agent.plan_editor import EditableSubQuestion, EditedPlan

        subtopics = [
            {"id": 1, "question": "What is RAG?", "rationale": "Core concept"},
        ]
        inline_plan = EditedPlan(
            subtopics=[
                EditableSubQuestion(
                    id=1, question="Inline edited question", rationale="Inline"
                ),
            ]
        )

        with (
            patch(
                "research_agent.cli._approve_plan", side_effect=["edit", "approve"]
            ),
            patch("research_agent.cli.edit_plan_in_editor", return_value=None),
            patch("research_agent.cli.edit_plan_inline", return_value=inline_plan),
        ):
            result = _handle_plan_review(subtopics)

        assert result is not None
        assert len(result) == 1
        assert result[0]["question"] == "Inline edited question"

    def test_edit_both_fail_returns_to_approval(self) -> None:
        """When both editors return None, the loop continues and
        the user can cancel."""
        subtopics = [
            {"id": 1, "question": "What is RAG?", "rationale": "Core concept"},
        ]

        with (
            patch(
                "research_agent.cli._approve_plan", side_effect=["edit", "cancel"]
            ),
            patch("research_agent.cli.edit_plan_in_editor", return_value=None),
            patch("research_agent.cli.edit_plan_inline", return_value=None),
        ):
            result = _handle_plan_review(subtopics)

        assert result is None


# ---- Approve plan timeout ---------------------------------------------------


class TestApprovePlanTimeout:
    """Test timeout-based auto-approval (lines 151-155)."""

    def test_auto_approve_on_timeout(self) -> None:
        """When stdin is a tty and select returns no ready input,
        _approve_plan auto-approves."""
        with (
            patch("research_agent.cli.sys.stdin") as mock_stdin,
            patch("research_agent.cli.select.select", return_value=([], [], [])),
        ):
            mock_stdin.isatty.return_value = True
            result = _approve_plan(timeout_seconds=5)

        assert result == "approve"


# ---- Run command output path ------------------------------------------------


class TestRunCommandOutputPath:
    """Test report file writing (lines 407-411)."""

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_writes_report_to_output_dir(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """When _current_state has a non-empty final_report after the pipeline
        loop, the report is written to the output directory."""
        import research_agent.cli as cli_mod

        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings

        # Make compile_graph return a mock, then patch _current_state
        # to have a final_report after the progress loop runs.
        mock_compile.return_value = MagicMock()

        out_dir = tmp_path / "output"

        # We need to inject a final_report into the state during the run.
        # The easiest way is to patch the progress loop to set the report.
        original_create_progress = cli_mod._create_progress

        def patched_progress():
            progress = original_create_progress()
            return progress

        # Instead, patch at a lower level: intercept after compile_graph
        # and set the state's final_report via a side effect.
        def set_report(*args, **kwargs):
            cli_mod._current_state["final_report"] = "# Test Report\n\nFindings."
            return MagicMock()

        mock_compile.side_effect = set_report

        result = runner.invoke(
            app, ["run", "test query", "--output", str(out_dir)]
        )
        assert result.exit_code == 0
        assert "Report saved" in result.output

        # Verify a report file was written in the output dir
        report_files = list(out_dir.glob("run-*.md"))
        assert len(report_files) == 1
        assert "Test Report" in report_files[0].read_text()


# ---- Resume with verbose flag -----------------------------------------------


class TestResumeVerbose:
    """Test verbose flag in resume command (line 454)."""

    @patch("research_agent.cli._load_settings")
    def test_resume_verbose_sets_debug(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Passing --verbose to resume should pass logging level DEBUG
        to _load_settings."""
        from research_agent.checkpoints import CheckpointManager

        cp_dir = tmp_path / "checkpoints"
        run_dir = cp_dir / "run-test456"
        run_dir.mkdir(parents=True)

        mgr = CheckpointManager(directory=run_dir)
        mgr.save(
            "run-test456-step-1", {"query": "test", "step": "plan"}, step_index=1
        )

        settings = MagicMock()
        settings.checkpoints.directory = cp_dir
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings

        with patch("research_agent.graph.compile_graph", return_value=MagicMock()):
            result = runner.invoke(
                app, ["resume", "--dir", str(cp_dir), "--verbose"]
            )

        assert result.exit_code == 0
        # _load_settings is called twice: once for resume, once for run.
        # The first call (from resume) should have logging debug override.
        first_call_kwargs = mock_settings.call_args_list[0]
        assert first_call_kwargs[1]["logging"] == {"level": "DEBUG"}


# ---- Main entrypoint --------------------------------------------------------


class TestMainEntrypoint:
    """Test the main() function (line 597)."""

    def test_main_calls_app(self) -> None:
        """main() should invoke the Typer app."""
        with patch("research_agent.cli.app") as mock_app:
            main()
        mock_app.assert_called_once()


# ---- Run command with resume_flag finding checkpoint -------------------------


class TestRunResumeWithCheckpoint:
    """Test resume_flag path in the run command (lines 379-380)."""

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_run_with_resume_flag_loads_checkpoint(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """When --resume is passed and a checkpoint exists, the state is
        loaded from the checkpoint."""
        from research_agent.checkpoints import CheckpointManager

        settings = MagicMock()
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")

        # The run command creates cp_dir = settings.checkpoints.directory / run_id
        # but we need the checkpoint to exist *before* the run starts.
        # Since run_id is generated dynamically, we need to mock generate_run_id.
        mock_settings.return_value = settings
        mock_compile.return_value = MagicMock()

        # Pre-create a checkpoint directory. The run command does:
        #   cp_dir = settings.checkpoints.directory / run_id
        # We mock generate_run_id to return a known value.
        run_id = "run-resume-test"
        cp_dir = tmp_path / "cp" / run_id
        cp_dir.mkdir(parents=True)

        mgr = CheckpointManager(directory=cp_dir, max_checkpoints=5)
        mgr.save(
            f"{run_id}-step-2",
            {"query": "resumable query", "step": "search", "step_index": 2},
            step_index=2,
        )

        settings.checkpoints.directory = tmp_path / "cp"

        with patch(
            "research_agent.cli.generate_run_id", return_value=run_id
        ):
            result = runner.invoke(app, ["run", "resumable query", "--resume"])

        assert result.exit_code == 0
        assert "Resuming from checkpoint" in result.output


# ---- Typer.Exit re-raise (line 421) -----------------------------------------


class TestRunTyperExitReRaise:
    """Test that typer.Exit is re-raised, not caught as a generic exception."""

    @patch("research_agent.graph.compile_graph")
    @patch("research_agent.cli._load_settings")
    def test_typer_exit_is_reraised(
        self, mock_settings: MagicMock, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """If a typer.Exit is raised during the pipeline, it should propagate
        directly instead of being caught by the generic Exception handler."""
        settings = MagicMock()
        settings.checkpoints.directory = tmp_path / "cp"
        settings.checkpoints.max_checkpoints = 5
        settings.report.output_dir = str(tmp_path / "reports")
        mock_settings.return_value = settings

        # Make compile_graph raise typer.Exit
        mock_compile.side_effect = typer.Exit(code=0)

        result = runner.invoke(app, ["run", "test query"])
        # typer.Exit(0) should not trigger the error display path
        assert result.exit_code == 0
