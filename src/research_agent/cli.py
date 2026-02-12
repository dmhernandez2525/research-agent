"""Typer CLI entry point for the research-agent."""

from __future__ import annotations

import select
import shutil
import signal
import sys
from pathlib import Path
from typing import Annotated, Any

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from research_agent import __version__
from research_agent.checkpoints import CheckpointManager, generate_run_id
from research_agent.config import Settings, format_validation_error
from research_agent.plan_editor import edit_plan_in_editor, edit_plan_inline

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="research-agent",
    help="Crash-resilient deep research agent for the Apps That Build Apps ecosystem.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Signal handling for graceful Ctrl+C
# ---------------------------------------------------------------------------

_checkpoint_mgr: CheckpointManager | None = None
_current_state: dict[str, Any] | None = None
_current_run_id: str | None = None
_interrupt_count = 0


def _handle_sigint(signum: int, frame: Any) -> None:
    """Handle Ctrl+C gracefully: first press saves checkpoint, second exits."""
    global _interrupt_count
    _interrupt_count += 1

    if _interrupt_count == 1:
        err_console.print("\n[yellow]Interrupt received. Saving checkpoint...[/yellow]")
        if _checkpoint_mgr and _current_state and _current_run_id:
            try:
                step_idx = _current_state.get("step_index", 0)
                step_name = _current_state.get("step", "interrupted")
                cp_id = f"{_current_run_id}-interrupt-{step_idx}"
                _checkpoint_mgr.save(
                    cp_id,
                    _current_state,
                    step_index=step_idx,
                    step_name=step_name,
                )
                err_console.print(
                    f"[green]Checkpoint saved:[/green] {cp_id}\n"
                    f"Resume with: [bold]research-agent resume[/bold]"
                )
            except Exception as exc:
                err_console.print(f"[red]Failed to save checkpoint:[/red] {exc}")
        else:
            err_console.print("[dim]No active state to checkpoint.[/dim]")
        err_console.print("[yellow]Press Ctrl+C again to exit immediately.[/yellow]")
    else:
        err_console.print("\n[red]Forced exit.[/red]")
        sys.exit(130)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings(
    config_path: Path | None = None,
    **overrides: Any,
) -> Settings:
    """Load settings with error handling and user-friendly messages."""
    from pydantic import ValidationError

    try:
        return Settings.load(config_path=config_path, **overrides)
    except ValidationError as exc:
        err_console.print(
            Panel(
                format_validation_error(exc),
                title="Configuration Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _create_progress() -> Progress:
    """Create a Rich progress bar with spinner, text, bar, and time columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def _display_plan(subtopics: list[dict[str, Any]]) -> None:
    """Display the research plan as a Rich table."""
    table = Table(title="Research Plan", show_lines=True)
    table.add_column("#", style="cyan", justify="right", width=4)
    table.add_column("Sub-Question", style="white")
    table.add_column("Rationale", style="dim")

    for sq in subtopics:
        table.add_row(
            str(sq.get("id", "")),
            sq.get("question", ""),
            sq.get("rationale", ""),
        )

    console.print(table)


def _approve_plan(timeout_seconds: int = 0) -> str:
    """Prompt user to approve, edit, or cancel the research plan.

    Args:
        timeout_seconds: If > 0, auto-approve after this many seconds
            of inactivity. Useful for batch/CI environments.

    Returns:
        One of "approve", "edit", or "cancel".
    """
    console.print()

    if timeout_seconds > 0 and sys.stdin.isatty():
        console.print(f"[dim]Auto-approving in {timeout_seconds}s if no input...[/dim]")
        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
        if not ready:
            console.print("[green]Auto-approved (timeout).[/green]")
            return "approve"

    choice = typer.prompt(
        "Approve plan? [a]pprove / [e]dit / [c]ancel",
        default="a",
    )
    choice = choice.strip().lower()
    if choice in ("a", "approve"):
        return "approve"
    if choice in ("e", "edit"):
        return "edit"
    return "cancel"


def _handle_plan_review(
    subtopics: list[dict[str, Any]],
    no_approve: bool = False,
    approve_timeout: int = 0,
) -> list[dict[str, Any]] | None:
    """Run the full plan review workflow.

    Displays the plan table, prompts for approval, and handles the edit
    flow when requested. Returns the (possibly edited) subtopics,
    or None if the user cancelled.

    Args:
        subtopics: List of subtopic dicts from the planner.
        no_approve: If True, skip approval and proceed immediately.
        approve_timeout: Seconds before auto-approval (0 = no timeout).

    Returns:
        Final list of subtopic dicts, or None if cancelled.
    """
    _display_plan(subtopics)

    if no_approve:
        console.print("[dim]Plan auto-approved (--no-approve).[/dim]")
        return subtopics

    while True:
        decision = _approve_plan(timeout_seconds=approve_timeout)

        if decision == "approve":
            console.print("[green]Plan approved.[/green]")
            return subtopics

        if decision == "cancel":
            console.print("[yellow]Plan cancelled by user.[/yellow]")
            return None

        # decision == "edit"
        console.print("[cyan]Opening plan editor...[/cyan]")

        # Try $EDITOR first, fall back to inline editing
        edited = edit_plan_in_editor(subtopics)
        if edited is None:
            console.print(
                "[yellow]Editor returned no changes. Trying inline editing...[/yellow]"
            )
            edited = edit_plan_inline(subtopics)

        if edited is None:
            console.print("[yellow]Edit cancelled. Returning to approval.[/yellow]")
            continue

        # Update subtopics with the edited version
        subtopics = [
            {
                "id": sq.id,
                "question": sq.question,
                "rationale": sq.rationale,
            }
            for sq in edited.subtopics
        ]
        console.print(
            f"[green]Plan updated ({len(subtopics)} sub-questions).[/green]"
        )
        _display_plan(subtopics)


def _display_error_with_resume(
    exc: Exception,
    run_id: str | None = None,
) -> None:
    """Display an error with resume instructions."""
    err_console.print(
        Panel(
            f"[red bold]Pipeline Error[/red bold]\n\n{exc}",
            title="Error",
            border_style="red",
        )
    )
    if run_id:
        err_console.print(
            "\nTo resume this run:\n"
            "  [bold]research-agent resume --dir data/checkpoints[/bold]\n"
        )


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]research-agent[/bold] {__version__}")
        raise typer.Exit


@app.callback()
def common(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Research-agent global options."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    query: Annotated[str, typer.Argument(help="The research query to investigate.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for the report."),
    ] = None,
    resume_flag: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume from latest checkpoint."),
    ] = False,
    budget: Annotated[
        float | None,
        typer.Option("--budget", "-b", help="Max cost budget in USD."),
    ] = None,
    no_approve: Annotated[
        bool,
        typer.Option("--no-approve", help="Skip plan approval, run immediately."),
    ] = False,
    approve_timeout: Annotated[
        int,
        typer.Option(
            "--approve-timeout",
            help="Auto-approve plan after N seconds of inactivity (0=disabled).",
        ),
    ] = 0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging."),
    ] = False,
) -> None:
    """Run a deep-research pipeline for the given query."""
    global _checkpoint_mgr, _current_state, _current_run_id

    # Load configuration
    overrides: dict[str, Any] = {}
    if budget is not None:
        overrides["costs"] = {"max_cost_per_run": budget}
    if verbose:
        overrides["logging"] = {"level": "DEBUG"}

    settings = _load_settings(config, **overrides)

    # Set up signal handler
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    run_id = generate_run_id()
    _current_run_id = run_id

    console.print(
        Panel(
            f"[bold]{query}[/bold]\n\nRun ID: [cyan]{run_id}[/cyan]",
            title="Research Agent",
            border_style="blue",
        )
    )

    try:
        # Set up checkpoint manager
        cp_dir = settings.checkpoints.directory / run_id
        _checkpoint_mgr = CheckpointManager(
            directory=cp_dir,
            max_checkpoints=settings.checkpoints.max_checkpoints,
        )

        # Initialize state
        _current_state = {
            "query": query,
            "step": "plan",
            "step_index": 0,
            "subtopics": [],
            "search_results": [],
            "scraped_pages": [],
            "subtopic_summaries": [],
            "final_report": "",
            "sources": [],
            "error_log": [],
            "cost_so_far": 0.0,
            "llm_call_count": 0,
            "seen_urls": [],
            "current_subtopic_index": 0,
            "search_retry_count": 0,
        }

        if resume_flag:
            latest = _checkpoint_mgr.latest()
            if latest:
                console.print(f"[green]Resuming from checkpoint:[/green] {latest}")
                _current_state = _checkpoint_mgr.load(latest)
            else:
                console.print("[yellow]No checkpoint found, starting fresh.[/yellow]")

        # Compile and run graph
        from research_agent.graph import compile_graph

        compile_graph(settings, checkpoint_db=cp_dir / "langgraph.db")

        steps = ["plan", "search", "scrape", "summarize", "synthesize"]
        with _create_progress() as progress:
            task = progress.add_task("Running pipeline...", total=len(steps))

            for i, step_name in enumerate(steps):
                progress.update(
                    task,
                    description=f"[cyan]{step_name.capitalize()}[/cyan]",
                    completed=i,
                )
                _current_state["step"] = step_name
                _current_state["step_index"] = i

            progress.update(task, completed=len(steps))

        # Output results
        report = _current_state.get("final_report", "")
        if report:
            out_dir = output or Path(settings.report.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / f"{run_id}.md"
            report_path.write_text(report)
            console.print(f"\n[green]Report saved:[/green] {report_path}")
        else:
            console.print(
                "\n[yellow]No report generated. "
                "Pipeline nodes may not be fully implemented yet.[/yellow]"
            )

        console.print(f"[dim]Run ID: {run_id}[/dim]")

    except typer.Exit:
        raise
    except Exception as exc:
        _display_error_with_resume(exc, run_id)
        raise typer.Exit(code=1) from exc
    finally:
        signal.signal(signal.SIGINT, original_handler)
        _checkpoint_mgr = None
        _current_state = None
        _current_run_id = None


@app.command(name="resume")
def resume_cmd(
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Checkpoint directory to resume from."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
    no_approve: Annotated[
        bool,
        typer.Option("--no-approve", help="Skip plan approval on resume."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging."),
    ] = False,
) -> None:
    """Resume a previously interrupted research run from the latest checkpoint."""
    overrides: dict[str, Any] = {}
    if verbose:
        overrides["logging"] = {"level": "DEBUG"}

    settings = _load_settings(config, **overrides)
    cp_dir = checkpoint_dir or settings.checkpoints.directory

    if not cp_dir.exists():
        err_console.print("[red]Checkpoint directory not found.[/red]")
        raise typer.Exit(code=1)

    # Find the most recent run subdirectory
    run_dirs = sorted(
        (d for d in cp_dir.iterdir() if d.is_dir() and d.name.startswith("run-")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not run_dirs:
        err_console.print("[yellow]No runs found in checkpoint directory.[/yellow]")
        raise typer.Exit(code=1)

    latest_dir = run_dirs[0]
    mgr = CheckpointManager(directory=latest_dir)
    latest_cp = mgr.latest()

    if not latest_cp:
        err_console.print(
            f"[yellow]No checkpoints found in {latest_dir.name}.[/yellow]"
        )
        raise typer.Exit(code=1)

    state = mgr.load(latest_cp)
    query = state.get("query", "<unknown>")

    console.print(
        Panel(
            f"[bold]Resuming:[/bold] {query}\n"
            f"Checkpoint: [cyan]{latest_cp}[/cyan]\n"
            f"Run: [dim]{latest_dir.name}[/dim]",
            title="Resume",
            border_style="green",
        )
    )

    # Delegate to run with resume flag
    run(
        query=query,
        config=config,
        resume_flag=True,
        no_approve=no_approve,
        verbose=verbose,
        output=None,
        budget=None,
    )


@app.command()
def evaluate(
    report: Annotated[
        Path,
        typer.Argument(help="Path to the generated report to evaluate."),
    ],
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Original research query for evaluation."),
    ] = "",
) -> None:
    """Evaluate a generated report using LLM-as-judge scoring."""
    if not report.exists():
        err_console.print(f"[red]Report not found:[/red] {report}")
        raise typer.Exit(code=1)

    content = report.read_text()
    if not content.strip():
        err_console.print("[red]Report file is empty.[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Report: [cyan]{report}[/cyan]\n"
            f"Length: {len(content):,} characters\n"
            f"Query: {query or '[not provided]'}",
            title="Evaluation",
            border_style="blue",
        )
    )

    # Evaluation framework is implemented in F7; for now display info
    console.print(
        "[yellow]Full evaluation requires the evaluation framework (Phase 7).[/yellow]"
    )


@app.command()
def clean(
    checkpoints: Annotated[
        bool,
        typer.Option("--checkpoints", help="Remove all checkpoint files."),
    ] = False,
    cache: Annotated[
        bool,
        typer.Option("--cache", help="Remove ChromaDB cache."),
    ] = False,
    all_data: Annotated[
        bool,
        typer.Option("--all", help="Remove all generated data."),
    ] = False,
) -> None:
    """Clean up generated data, checkpoints, and caches."""
    if not (checkpoints or cache or all_data):
        err_console.print(
            "[yellow]No cleanup target specified. "
            "Use --checkpoints, --cache, or --all.[/yellow]"
        )
        raise typer.Exit(code=1)

    settings = _load_settings()
    trash_dir = Path.home() / ".Trash"
    cleaned: list[str] = []

    if checkpoints or all_data:
        cp_dir = settings.checkpoints.directory
        if cp_dir.exists():
            dest = trash_dir / f"research-agent-checkpoints-{generate_run_id()}"
            shutil.move(str(cp_dir), str(dest))
            cleaned.append(f"Checkpoints: {cp_dir}")

    if cache or all_data:
        vs_dir = Path(settings.vector_store.persist_directory)
        if vs_dir.exists():
            dest = trash_dir / f"research-agent-cache-{generate_run_id()}"
            shutil.move(str(vs_dir), str(dest))
            cleaned.append(f"Cache: {vs_dir}")

    if cleaned:
        for item in cleaned:
            console.print(f"[green]Removed:[/green] {item}")
        console.print("[dim]Files moved to Trash.[/dim]")
    else:
        console.print("[yellow]Nothing to clean.[/yellow]")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
