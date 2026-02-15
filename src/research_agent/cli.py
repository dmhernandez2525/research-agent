"""Typer CLI entry point for the research-agent."""

from __future__ import annotations

import json
import select
import shutil
import signal
import sys
import time
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
from research_agent.agentprompts.bridge import run_for_project
from research_agent.agentprompts.registry import ProjectRegistry
from research_agent.agentprompts.templates import (
    list_templates,
    load_template,
    render_template,
)
from research_agent.agentprompts.watch import PromptWatcher
from research_agent.api.auth import APIKeyStore
from research_agent.api.server import run_server
from research_agent.checkpoints import CheckpointManager, generate_run_id
from research_agent.config import Settings, format_validation_error
from research_agent.doctor import CheckStatus, run_doctor
from research_agent.enhance_context import build_project_context
from research_agent.enhance_engine import (
    generate_enhancement_report,
    identify_opportunities,
    persist_findings,
    plan_incremental_research,
)
from research_agent.enhance_models import OpportunityCategory
from research_agent.enhance_store import KnowledgeStore as EnhancementKnowledgeStore
from research_agent.knowledge.io import (
    export_to_json as export_knowledge_json,
)
from research_agent.knowledge.io import (
    export_to_markdown as export_knowledge_markdown,
)
from research_agent.knowledge.io import (
    import_from_json as import_knowledge_json,
)
from research_agent.knowledge.service import KnowledgeService
from research_agent.knowledge.store import KnowledgeStore as ResearchKnowledgeStore
from research_agent.mcp.serve import (
    benchmark_tool_latency,
    run_sse_server,
    run_stdio_server,
)
from research_agent.mcp.server import MCPServer
from research_agent.plan_editor import edit_plan_in_editor, edit_plan_inline

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="research-agent",
    help="Crash-resilient deep research agent for the Apps That Build Apps ecosystem.",
    no_args_is_help=True,
)
knowledge_app = typer.Typer(help="Knowledge graph, synthesis, and export commands.")
projects_app = typer.Typer(help="AgentPrompts project registry commands.")
template_app = typer.Typer(help="AgentPrompts RESEARCH_PROMPT templates.")
mcp_app = typer.Typer(help="MCP protocol server and diagnostics.")

app.add_typer(knowledge_app, name="knowledge")
app.add_typer(projects_app, name="projects")
app.add_typer(template_app, name="template")
app.add_typer(mcp_app, name="mcp")


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
        console.print(f"[green]Plan updated ({len(subtopics)} sub-questions).[/green]")
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


def _parse_focus_areas(raw: str) -> set[OpportunityCategory]:
    """Parse comma-delimited focus area text to enum values."""
    if not raw.strip():
        return set()

    valid = {item.value: item for item in OpportunityCategory}
    result: set[OpportunityCategory] = set()
    unknown: list[str] = []

    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        item = valid.get(key)
        if item is None:
            unknown.append(key)
            continue
        result.add(item)

    if unknown:
        joined = ", ".join(sorted(unknown))
        valid_values = ", ".join(sorted(valid))
        raise typer.BadParameter(
            f"Unknown focus area(s): {joined}. Valid values: {valid_values}"
        )

    return result


def _knowledge_store_path(settings: Settings) -> Path:
    return Path(settings.vector_store.persist_directory) / "knowledge.json"


DEFAULT_PROJECTS_DIR = Path("~/Desktop/Projects/_@agent-prompts").expanduser()
DEFAULT_REGISTRY_PATH = Path("./data/agentprompts-projects.json")


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
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'md' (default) or 'pdf' (requires pymupdf).",
        ),
    ] = "md",
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

            if fmt == "pdf":
                from research_agent.pdf_output import write_pdf_report

                pdf_path = write_pdf_report(report, query, out_dir)
                if pdf_path:
                    console.print(f"\n[green]PDF report saved:[/green] {pdf_path}")
                else:
                    console.print(
                        "[yellow]PDF generation unavailable. "
                        "Install pymupdf: pip install research-agent[pdf][/yellow]"
                    )
                    report_path = out_dir / f"{run_id}.md"
                    report_path.write_text(report)
                    console.print(
                        f"[green]Markdown report saved:[/green] {report_path}"
                    )
            else:
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
def doctor(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
    no_api_probes: Annotated[
        bool,
        typer.Option(
            "--no-api-probes",
            help="Skip external API probe calls (offline mode).",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress table output, use exit code only."),
    ] = False,
) -> None:
    """Run self-diagnostics and health checks for this environment."""
    settings = _load_settings(config)
    report = run_doctor(
        settings=settings,
        config_path=config,
        check_api_probes=not no_api_probes,
    )

    if not quiet:
        table = Table(title="Research Agent Doctor", show_lines=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message")

        status_style = {
            CheckStatus.OK: "[green]OK[/green]",
            CheckStatus.WARN: "[yellow]WARN[/yellow]",
            CheckStatus.FAIL: "[red]FAIL[/red]",
        }

        for check in report.checks:
            table.add_row(
                check.name,
                status_style[check.status],
                check.message,
            )
        console.print(table)

        for check in report.checks:
            if check.details:
                details = ", ".join(f"{k}={v}" for k, v in check.details.items())
                console.print(f"[dim]{check.name}: {details}[/dim]")

    raise typer.Exit(code=report.exit_code)


@app.command()
def enhance(
    project: Annotated[
        Path,
        typer.Option(
            "--project",
            help="Path to the project to analyze for enhancement opportunities.",
        ),
    ],
    focus: Annotated[
        str,
        typer.Option(
            "--focus",
            help="Comma-separated focus categories (security,performance,testing,documentation,architecture,dependencies).",
        ),
    ] = "",
    stale_days: Annotated[
        int,
        typer.Option(
            "--stale-days",
            help="Skip topics researched within this many days unless force refreshed.",
        ),
    ] = 14,
    force_refresh: Annotated[
        bool,
        typer.Option(
            "--force-refresh",
            help="Re-research all topics regardless of staleness.",
        ),
    ] = False,
    apply_to: Annotated[
        Path | None,
        typer.Option(
            "--apply-to",
            help="Optional file path to also write enhancement recommendations into.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path for generated COMPILED_RESEARCH markdown.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Analyze a codebase and generate enhancement-focused research output."""
    settings = _load_settings(config)
    project_path = project.expanduser().resolve()
    if not project_path.exists():
        raise typer.BadParameter(f"Project path does not exist: {project_path}")

    focus_areas = _parse_focus_areas(focus)
    context = build_project_context(project_path)
    opportunities = identify_opportunities(
        context,
        focus_areas=focus_areas or None,
    )

    knowledge_path = Path(settings.vector_store.persist_directory) / "enhancement.json"
    store = EnhancementKnowledgeStore(knowledge_path)
    refresh_targets, delta, shared_entries = plan_incremental_research(
        project_id=context.project_name,
        opportunities=opportunities,
        store=store,
        stale_days=max(stale_days, 1),
        force_refresh=force_refresh,
    )

    if refresh_targets:
        persist_findings(context.project_name, refresh_targets, store)

    report_targets = refresh_targets or opportunities
    report = generate_enhancement_report(
        context=context,
        opportunities=report_targets,
        delta=delta,
        shared_entries=shared_entries,
    )

    output_path = output or (project_path / "COMPILED_RESEARCH.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    if apply_to is not None:
        apply_to.parent.mkdir(parents=True, exist_ok=True)
        apply_to.write_text(report, encoding="utf-8")

    console.print(
        Panel(
            f"Project: [cyan]{context.project_name}[/cyan]\n"
            f"Focus: {focus or 'all categories'}\n"
            f"Opportunities researched: {len(report_targets)}\n"
            f"Output: [green]{output_path}[/green]",
            title="Enhancement Research",
            border_style="magenta",
        )
    )


@app.command(name="for-project")
def for_project(
    project_name: Annotated[
        str,
        typer.Argument(help="AgentPrompts project name."),
    ],
    projects_dir: Annotated[
        Path,
        typer.Option(
            "--projects-dir",
            help="Root directory that contains AgentPrompts project folders.",
        ),
    ] = DEFAULT_PROJECTS_DIR,
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            help="Path to persistent project-name registry JSON.",
        ),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """Generate COMPILED_RESEARCH.md for a project in the AgentPrompts ecosystem."""
    resolved_projects_dir = projects_dir.expanduser().resolve()
    registry = ProjectRegistry(registry_path.expanduser().resolve())
    registry.discover_and_register(resolved_projects_dir)

    try:
        result = run_for_project(
            project_name=project_name,
            projects_dir=resolved_projects_dir,
            registry=registry,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(
        Panel(
            f"Project: [cyan]{result.project_name}[/cyan]\n"
            f"Path: {result.project_path}\n"
            f"Output: [green]{result.output_path}[/green]\n"
            f"Status file: {result.status_path}",
            title="AgentPrompts Bridge",
            border_style="blue",
        )
    )

    if not result.quality_gate_passed:
        err_console.print(
            "[red]Quality gate failed. Missing sections:[/red] "
            + ", ".join(result.missing_sections)
        )
        raise typer.Exit(code=1)


@app.command()
def watch(
    projects_dir: Annotated[
        Path,
        typer.Option(
            "--projects-dir",
            help="Root directory containing AgentPrompts projects.",
        ),
    ] = DEFAULT_PROJECTS_DIR,
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            help="Path to persistent project-name registry JSON.",
        ),
    ] = DEFAULT_REGISTRY_PATH,
    debounce_seconds: Annotated[
        float,
        typer.Option(
            "--debounce-seconds",
            help="Minimum seconds between triggers for the same prompt file.",
        ),
    ] = 2.0,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Polling interval in seconds."),
    ] = 1.0,
    once: Annotated[
        bool,
        typer.Option(
            "--once",
            help="Run one scan iteration and exit (useful for validation).",
        ),
    ] = False,
    no_notify: Annotated[
        bool,
        typer.Option(
            "--no-notify",
            help="Disable desktop notifications when research completes.",
        ),
    ] = False,
) -> None:
    """Watch RESEARCH_PROMPT.md files and auto-trigger research on changes."""
    resolved_projects_dir = projects_dir.expanduser().resolve()
    registry = ProjectRegistry(registry_path.expanduser().resolve())
    registry.discover_and_register(resolved_projects_dir)

    watcher = PromptWatcher(
        projects_dir=resolved_projects_dir,
        registry=registry,
        debounce_seconds=max(debounce_seconds, 0.0),
        poll_interval=max(poll_interval, 0.1),
        notify=not no_notify,
    )

    def report_results(label: str, results: list[Any]) -> None:
        if not results:
            console.print(f"[dim]{label}: no prompt changes detected.[/dim]")
            return
        for item in results:
            console.print(
                f"[green]{label}:[/green] {item.project_name} -> {item.output_path}"
            )

    if once:
        results = watcher.run_once()
        report_results("Watch run", results)
        if any(not item.quality_gate_passed for item in results):
            raise typer.Exit(code=1)
        return

    console.print(
        f"[cyan]Watching[/cyan] {resolved_projects_dir} "
        f"(poll={poll_interval:.1f}s, debounce={debounce_seconds:.1f}s)"
    )
    try:
        while True:
            report_results("Triggered", watcher.run_once())
            time.sleep(max(poll_interval, 0.1))
    except KeyboardInterrupt:
        console.print("[yellow]Watch stopped by user.[/yellow]")


@projects_app.command("list")
def projects_list(
    projects_dir: Annotated[
        Path,
        typer.Option(
            "--projects-dir",
            help="Root directory containing AgentPrompts projects.",
        ),
    ] = DEFAULT_PROJECTS_DIR,
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            help="Path to persistent project-name registry JSON.",
        ),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """List registered AgentPrompts projects."""
    resolved_projects_dir = projects_dir.expanduser().resolve()
    registry = ProjectRegistry(registry_path.expanduser().resolve())
    registry.discover_and_register(resolved_projects_dir)
    projects = registry.list_projects()

    table = Table(title="AgentPrompts Projects", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Path")
    if not projects:
        console.print("[yellow]No projects registered.[/yellow]")
        return
    for name, path in projects:
        table.add_row(name, str(path))
    console.print(table)


@projects_app.command("register")
def projects_register(
    name: Annotated[str, typer.Argument(help="Registry key for the project.")],
    path: Annotated[Path, typer.Argument(help="Absolute or relative project path.")],
    registry_path: Annotated[
        Path,
        typer.Option(
            "--registry-path",
            help="Path to persistent project-name registry JSON.",
        ),
    ] = DEFAULT_REGISTRY_PATH,
) -> None:
    """Register an AgentPrompts project path."""
    project_path = path.expanduser().resolve()
    if not project_path.exists():
        raise typer.BadParameter(f"Path does not exist: {project_path}")
    registry = ProjectRegistry(registry_path.expanduser().resolve())
    registry.register(name, project_path)
    console.print(f"[green]Registered[/green] {name} -> {project_path}")


@template_app.command("list")
def template_list(
    custom_dir: Annotated[
        Path | None,
        typer.Option(
            "--custom-dir",
            help="Optional custom template directory containing *.md files.",
        ),
    ] = None,
) -> None:
    """List built-in and custom research templates."""
    resolved_custom_dir = custom_dir.expanduser().resolve() if custom_dir else None
    names = list_templates(resolved_custom_dir)
    table = Table(title="Research Templates")
    table.add_column("Template", style="cyan")
    for name in names:
        table.add_row(name)
    console.print(table)


@template_app.command("use")
def template_use(
    template_name: Annotated[
        str,
        typer.Argument(help="Template name (built-in or custom file stem)."),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (typically RESEARCH_PROMPT.md).",
        ),
    ] = Path("RESEARCH_PROMPT.md"),
    project_name: Annotated[
        str,
        typer.Option("--project-name", help="Value for {{PROJECT_NAME}}."),
    ] = "my-project",
    language: Annotated[
        str,
        typer.Option("--language", help="Value for {{LANGUAGE}}."),
    ] = "python",
    focus_area: Annotated[
        str,
        typer.Option("--focus-area", help="Value for {{FOCUS_AREA}}."),
    ] = "architecture",
    custom_dir: Annotated[
        Path | None,
        typer.Option(
            "--custom-dir",
            help="Optional custom template directory containing *.md files.",
        ),
    ] = None,
) -> None:
    """Render a template to RESEARCH_PROMPT.md format."""
    resolved_custom_dir = custom_dir.expanduser().resolve() if custom_dir else None
    template_text = load_template(template_name, resolved_custom_dir)
    rendered = render_template(
        template_text,
        {
            "PROJECT_NAME": project_name,
            "LANGUAGE": language,
            "FOCUS_AREA": focus_area,
        },
    )
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Template written:[/green] {output_path}")


@mcp_app.command("serve")
def mcp_serve(
    transport: Annotated[
        str,
        typer.Option(
            "--transport",
            help="MCP transport: stdio or sse.",
        ),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option("--host", help="Host for SSE transport."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Port for SSE transport."),
    ] = 8765,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Serve the MCP protocol over stdio or SSE transport."""
    normalized = transport.strip().lower()
    if normalized not in {"stdio", "sse"}:
        raise typer.BadParameter("Transport must be 'stdio' or 'sse'.")

    settings = _load_settings(config)
    if normalized == "stdio":
        run_stdio_server(settings)
        return

    run_sse_server(settings, host=host, port=port)


@mcp_app.command("benchmark")
def mcp_benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            help="Query used for MCP research tool latency benchmark.",
        ),
    ] = "vector database tradeoffs",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Benchmark MCP research tool latency to first result."""
    settings = _load_settings(config)
    server = MCPServer(settings)
    result = benchmark_tool_latency(server, query=query)

    table = Table(title="MCP Benchmark")
    table.add_column("Query", style="cyan")
    table.add_column("Session ID")
    table.add_column("Latency (ms)", justify="right")
    table.add_row(
        str(result["query"]),
        str(result["session_id"]),
        f"{float(result['latency_ms']):.2f}",
    )
    console.print(table)


@knowledge_app.command("summarize")
def knowledge_summarize(
    topic: Annotated[
        str | None,
        typer.Option("--topic", help="Optional topic filter."),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Refresh trigger threshold used in summary diagnostics.",
        ),
    ] = 0.45,
    refresh_days: Annotated[
        int,
        typer.Option(
            "--refresh-days",
            help="Default refresh cadence in days used in diagnostics.",
        ),
    ] = 30,
    mermaid_out: Annotated[
        Path | None,
        typer.Option(
            "--mermaid-out",
            help="Optional output path for Mermaid relationship graph.",
        ),
    ] = None,
    graph_json_out: Annotated[
        Path | None,
        typer.Option(
            "--graph-json-out",
            help="Optional output path for JSON graph export.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Summarize consolidated knowledge and confidence status."""
    settings = _load_settings(config)
    store = ResearchKnowledgeStore(_knowledge_store_path(settings))
    service = KnowledgeService(store)

    relationship_count = service.rebuild_relationships(topic)
    summary = service.summarize(
        topic=topic,
        threshold=max(0.0, min(1.0, threshold)),
        refresh_days=max(1, refresh_days),
    )

    table = Table(title="Knowledge Summary", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Findings", str(len(summary.findings)))
    table.add_row("Conflicts", str(len(summary.conflicts)))
    table.add_row("Due for refresh", str(len(summary.due_for_refresh_ids)))
    table.add_row("Relationships", str(relationship_count))
    console.print(table)

    if summary.cluster_summaries:
        console.print("[bold]Cluster Summaries[/bold]")
        for cluster, text in summary.cluster_summaries.items():
            console.print(f"[cyan]{cluster}[/cyan]\n{text}")

    if mermaid_out is not None:
        mermaid_text = service.to_mermaid(topic)
        mermaid_path = mermaid_out.expanduser().resolve()
        mermaid_path.parent.mkdir(parents=True, exist_ok=True)
        mermaid_path.write_text(mermaid_text + "\n", encoding="utf-8")
        console.print(f"[green]Mermaid graph written:[/green] {mermaid_path}")

    if graph_json_out is not None:
        graph_payload = service.to_json_graph(topic)
        graph_path = graph_json_out.expanduser().resolve()
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
        console.print(f"[green]JSON graph written:[/green] {graph_path}")


@knowledge_app.command("refresh")
def knowledge_refresh(
    topic: Annotated[
        str,
        typer.Option("--topic", help='Topic to refresh (e.g., "AI news").'),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Refresh trigger confidence threshold."),
    ] = 0.45,
    refresh_days: Annotated[
        int,
        typer.Option("--refresh-days", help="Refresh schedule in days for this topic."),
    ] = 30,
    statement: Annotated[
        str | None,
        typer.Option(
            "--statement",
            help="Optional replacement statement for refreshed findings.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Refresh decayed or stale knowledge findings for a topic."""
    settings = _load_settings(config)
    store = ResearchKnowledgeStore(_knowledge_store_path(settings))
    service = KnowledgeService(store)
    refreshed = service.refresh_topic(
        topic=topic,
        threshold=max(0.0, min(1.0, threshold)),
        refresh_days=max(1, refresh_days),
        new_statement=statement,
    )
    service.rebuild_relationships(topic)
    console.print(
        f"[green]Refreshed findings:[/green] {refreshed} for topic '{topic}'."
    )


@knowledge_app.command("query")
def knowledge_query(
    topic: Annotated[str, typer.Argument(help="Topic phrase to query.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Query known findings and relationships for a topic."""
    settings = _load_settings(config)
    store = ResearchKnowledgeStore(_knowledge_store_path(settings))
    service = KnowledgeService(store)
    if not store.load().relationships:
        service.rebuild_relationships()

    result = service.query_topic(topic)
    console.print(Panel(topic, title="Knowledge Query", border_style="cyan"))

    if result["findings"]:
        console.print("[bold]Findings[/bold]")
        for line in result["findings"]:
            console.print(f"- {line}")
    else:
        console.print("[yellow]No findings matched this topic.[/yellow]")

    if result["relationships"]:
        console.print("[bold]Relationships[/bold]")
        for rel in result["relationships"]:
            console.print(f"- {rel}")


@knowledge_app.command("export")
def knowledge_export(
    output: Annotated[
        Path,
        typer.Argument(help="Destination export file (.json or .md)."),
    ],
    export_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Export format: json or md.",
        ),
    ] = "json",
    topic: Annotated[
        str | None,
        typer.Option("--topic", help="Optional topic filter."),
    ] = None,
    date_from: Annotated[
        str | None,
        typer.Option("--date-from", help="ISO timestamp lower bound."),
    ] = None,
    date_to: Annotated[
        str | None,
        typer.Option("--date-to", help="ISO timestamp upper bound."),
    ] = None,
    min_confidence: Annotated[
        float,
        typer.Option("--min-confidence", help="Minimum confidence threshold."),
    ] = 0.0,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Export knowledge base with optional selective filters."""
    normalized_format = export_format.strip().lower()
    if normalized_format not in {"json", "md"}:
        raise typer.BadParameter("Format must be 'json' or 'md'.")

    settings = _load_settings(config)
    store = ResearchKnowledgeStore(_knowledge_store_path(settings))
    payload = store.export_filtered(
        topic=topic,
        date_from=date_from,
        date_to=date_to,
        min_confidence=max(0.0, min(1.0, min_confidence)),
    )

    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if normalized_format == "json":
        export_knowledge_json(output_path, payload)
    else:
        output_path.write_text(export_knowledge_markdown(payload), encoding="utf-8")

    console.print(f"[green]Knowledge exported:[/green] {output_path}")


@knowledge_app.command("import")
def knowledge_import(
    source: Annotated[
        Path,
        typer.Argument(help="Knowledge export JSON file to import."),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Import knowledge payload and merge with conflict resolution."""
    source_path = source.expanduser().resolve()
    if not source_path.exists():
        raise typer.BadParameter(f"Import file not found: {source_path}")

    settings = _load_settings(config)
    store = ResearchKnowledgeStore(_knowledge_store_path(settings))
    payload = import_knowledge_json(source_path)
    summary = store.import_payload(payload)

    service = KnowledgeService(store)
    relationship_count = service.rebuild_relationships()

    console.print(
        Panel(
            f"Merged findings: {summary['merged']}\n"
            f"Conflicts resolved: {summary['conflicts']}\n"
            f"Relationships rebuilt: {relationship_count}",
            title="Knowledge Import",
            border_style="green",
        )
    )


@app.command()
def serve(
    port: Annotated[
        int,
        typer.Option("--port", help="Port to bind the FastAPI server."),
    ] = 8000,
    host: Annotated[
        str,
        typer.Option("--host", help="Host/interface to bind the FastAPI server."),
    ] = "0.0.0.0",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Run the research-agent FastAPI server."""
    settings = _load_settings(
        config,
        api={"enabled": True, "port": port, "host": host},
    )
    run_server(settings)


@app.command(name="api-keys")
def api_keys(
    create: Annotated[
        str | None,
        typer.Option(
            "--create", help="Create a new API key with the given display name."
        ),
    ] = None,
    admin: Annotated[
        bool,
        typer.Option("--admin", help="Mark newly-created API key as admin."),
    ] = False,
    revoke: Annotated[
        str | None,
        typer.Option("--revoke", help="Revoke an API key by key id."),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--list", help="List API keys and usage stats."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file."),
    ] = None,
) -> None:
    """Generate, revoke, and list API keys for API authentication."""
    settings = _load_settings(config)
    store = APIKeyStore(Path(settings.api.api_key_file))

    actions = sum(1 for value in [bool(create), bool(revoke), show] if value)
    if actions > 1:
        raise typer.BadParameter(
            "Use only one action at a time: --create, --revoke, or --list."
        )

    if create:
        record = store.create_key(name=create, admin=admin)
        console.print(
            Panel(
                f"ID: [cyan]{record.id}[/cyan]\n"
                f"Admin: {record.admin}\n"
                f"Key: [green]{record.key}[/green]",
                title="API Key Created",
                border_style="green",
            )
        )
        return

    if revoke:
        if not store.revoke_key(revoke):
            raise typer.BadParameter(f"API key not found: {revoke}")
        console.print(f"[green]Revoked API key:[/green] {revoke}")
        return

    table = Table(title="API Keys", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Admin")
    table.add_column("Revoked")
    table.add_column("Requests", justify="right")
    table.add_column("Sessions", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")

    for record in store.list_keys():
        table.add_row(
            record.id,
            record.name,
            str(record.admin),
            str(record.revoked),
            str(record.requests),
            str(record.sessions_started),
            str(record.tokens_used),
            f"${record.cost_usd:.4f}",
        )

    console.print(table)


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
