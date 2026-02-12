"""Rich live dashboard for real-time pipeline monitoring.

Displays pipeline progress, token usage, cost tracking, source counts,
and per-step timing in a Rich Layout that refreshes during execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from research_agent.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Dashboard panels
# ---------------------------------------------------------------------------


def _build_header(query: str, current_step: str) -> Panel:
    """Build the dashboard header panel.

    Args:
        query: The research query.
        current_step: Currently executing step name.

    Returns:
        A Rich Panel with the header.
    """
    text = Text()
    text.append("Research Agent", style="bold cyan")
    text.append(" | ", style="dim")
    text.append(f"Step: {current_step}", style="bold yellow")
    text.append("\n")
    text.append(query, style="italic")
    return Panel(text, title="Dashboard", border_style="cyan")


def _build_metrics_table(collector: MetricsCollector) -> Table:
    """Build the main metrics table.

    Args:
        collector: The metrics collector with current data.

    Returns:
        A Rich Table with key metrics.
    """
    snap = collector.snapshot()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Elapsed", f"{snap['elapsed_seconds']:.1f}s")
    table.add_row("Tokens", f"{snap['total_tokens']:,}")
    table.add_row(
        "Cost",
        f"${snap['total_cost_usd']:.4f} / ${snap['total_cost_usd'] + snap['budget_remaining_usd']:.2f}",
    )
    table.add_row("Budget Used", f"{snap['budget_used_pct']:.1f}%")
    table.add_row("Sources", str(snap["total_sources"]))
    table.add_row("Findings", str(snap["total_findings"]))
    table.add_row("Errors", str(snap["total_errors"]))
    return table


def _build_subtopic_progress(collector: MetricsCollector) -> Panel:
    """Build subtopic progress display.

    Args:
        collector: The metrics collector.

    Returns:
        A Rich Panel with progress bar and stats.
    """
    m = collector.metrics
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    task_id = progress.add_task(
        "Subtopics",
        total=max(m.subtopics_total, 1),
        completed=m.subtopics_completed,
    )
    # Manually advance to set the completed value
    progress.update(task_id, completed=m.subtopics_completed)
    return Panel(progress, title="Progress", border_style="green")


def _build_model_usage(collector: MetricsCollector) -> Panel:
    """Build model usage breakdown panel.

    Args:
        collector: The metrics collector.

    Returns:
        A Rich Panel showing per-model call counts.
    """
    table = Table(show_header=True, box=None)
    table.add_column("Model", style="cyan")
    table.add_column("Calls", justify="right")

    usage = collector.metrics.model_usage
    if not usage:
        table.add_row("(no calls yet)", "-")
    else:
        for model, count in sorted(usage.items()):
            table.add_row(model, str(count))

    return Panel(table, title="Model Usage", border_style="blue")


def _build_steps_table(collector: MetricsCollector) -> Panel:
    """Build pipeline steps status panel.

    Args:
        collector: The metrics collector.

    Returns:
        A Rich Panel with per-step timing.
    """
    table = Table(show_header=True, box=None)
    table.add_column("Step", style="bold")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")

    for step in collector.steps:
        status = (
            "[green]Done[/green]" if step.is_complete else "[yellow]Running[/yellow]"
        )
        duration = f"{step.duration_seconds:.1f}s"
        tokens = f"{step.input_tokens + step.output_tokens:,}"
        table.add_row(step.step_name, status, duration, tokens)

    if not collector.steps:
        table.add_row("(waiting)", "[dim]Pending[/dim]", "-", "-")

    return Panel(table, title="Pipeline Steps", border_style="magenta")


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------


def build_dashboard(
    collector: MetricsCollector,
    query: str = "",
) -> Layout:
    """Build the complete dashboard layout.

    Assembles all panels into a Rich Layout suitable for use with
    ``rich.live.Live`` for real-time updates.

    Args:
        collector: The metrics collector with current data.
        query: The research query being executed.

    Returns:
        A Rich Layout containing all dashboard panels.
    """
    current_step = collector.metrics.current_step

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=8),
    )

    layout["header"].update(_build_header(query, current_step))

    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="steps", ratio=2),
    )

    layout["metrics"].update(
        Panel(_build_metrics_table(collector), title="Metrics", border_style="green")
    )
    layout["steps"].update(_build_steps_table(collector))

    layout["footer"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="models", ratio=1),
    )

    layout["progress"].update(_build_subtopic_progress(collector))
    layout["models"].update(_build_model_usage(collector))

    return layout
