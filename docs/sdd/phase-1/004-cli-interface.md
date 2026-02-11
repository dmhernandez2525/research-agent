# SDD-004: CLI Interface

## Overview

The CLI is built with Typer for argument parsing and Rich for terminal output. It provides a single primary command for running research, along with subcommands for managing checkpoints and viewing past reports.

## Typer CLI Structure

```python
import typer

app = typer.Typer(
    name="research-agent",
    help="Deep research agent with crash resilience.",
    no_args_is_help=True,
)

@app.command()
def research(
    query: str = typer.Argument(..., help="Research topic or question"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model"),
    max_cost: float | None = typer.Option(None, "--max-cost", help="Max cost in USD"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Resume run by ID"),
    no_approve: bool = typer.Option(False, "--no-approve", help="Skip plan approval"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run a deep research session on the given topic."""
    ...

@app.command()
def list_runs() -> None:
    """List previous research runs."""
    ...

@app.command()
def show(run_id: str) -> None:
    """Display a past research report."""
    ...

def main() -> None:
    app()
```

**Entry point:** `research-agent` (configured in pyproject.toml `[project.scripts]`).

**Primary usage:**
```bash
research-agent "What are the best practices for LLM observability in 2026?"
research-agent --resume abc123          # Resume a previous run
research-agent --max-cost 1.00 "topic"  # Cap spending
research-agent --no-approve "topic"     # Skip plan review
```

## Rich Progress Bars and Tables

### Research Progress Display

```
Research: What are the best practices for LLM observability?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

Phase        Status     Duration   Cost
────────────────────────────────────────────────────────
Plan         Done       2.1s       $0.003
Search       Done       8.4s       $0.006
  Subtopic 1 Done       2.8s       $0.002
  Subtopic 2 Done       2.7s       $0.002
  Subtopic 3 Done       2.9s       $0.002
Scrape       Done       12.3s      --
Summarize    Done       6.2s       $0.045
Synthesize   Done       4.8s       $0.032
────────────────────────────────────────────────────────
Total                   33.8s      $0.086

Report saved to: reports/llm-observability-2026.md
```

### Implementation

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
```

The progress display is updated via callbacks registered on the LangGraph state. Each node entry/exit triggers a progress update.

## Plan Approval Workflow

After the plan node generates subtopics, the CLI pauses for user review (unless `--no-approve` is set):

```
Research Plan for: "LLM observability best practices"

#  Subtopic                          Queries
─────────────────────────────────────────────────────────
1  Tracing and span collection       3 queries
2  Cost and token monitoring         3 queries
3  Evaluation frameworks             3 queries
4  Production alerting patterns      3 queries

Estimated cost: $0.08 - $0.15
Estimated time: 2 - 4 minutes

[A]pprove  [E]dit  [C]ancel  >
```

**Actions:**
- **Approve (a):** Proceeds with the plan as-is.
- **Edit (e):** Opens the subtopic list in `$EDITOR` (or a simple inline editor) for modification. User can remove, reorder, or add subtopics.
- **Cancel (c):** Exits without running.

```python
def prompt_plan_approval(subtopics: list[Subtopic]) -> str:
    display_plan_table(subtopics)
    choice = console.input("\n[A]pprove  [E]dit  [C]ancel  > ").strip().lower()
    if choice in ("a", "approve", ""):
        return "approve"
    if choice in ("e", "edit"):
        return "edit"
    return "cancel"
```

## Graceful Ctrl+C Handling

The agent registers a signal handler for SIGINT that triggers a graceful shutdown:

```python
import signal
import asyncio

class GracefulShutdown:
    def __init__(self):
        self.should_stop = False
        self._original_handler = None

    def install(self) -> None:
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame) -> None:
        if self.should_stop:
            # Second Ctrl+C: force exit
            console.print("\n[red]Force exit. State was checkpointed.[/red]")
            raise SystemExit(1)

        self.should_stop = True
        console.print("\n[yellow]Graceful shutdown requested. "
                      "Finishing current step...[/yellow]")
        console.print("[dim]Press Ctrl+C again to force exit.[/dim]")
```

**Behavior:**
1. First Ctrl+C sets `should_stop = True`. The current node finishes, state is checkpointed, and the agent exits cleanly with a message showing the run ID for resumption.
2. Second Ctrl+C raises `SystemExit` immediately. The last completed checkpoint is still valid.

Each node checks `shutdown.should_stop` at safe points (between loop iterations, before starting a new API call) and returns early if set.

## Token and Cost Display

After each run, a summary is displayed:

```python
def display_cost_summary(metadata: dict) -> None:
    table = Table(title="Token & Cost Summary")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cost", justify="right")

    for entry in metadata["model_usage"]:
        table.add_row(
            entry["provider"],
            entry["model"],
            f"{entry['input_tokens']:,}",
            f"{entry['output_tokens']:,}",
            f"${entry['cost']:.4f}",
        )

    table.add_section()
    table.add_row(
        "Total", "",
        f"{metadata['total_input_tokens']:,}",
        f"{metadata['total_output_tokens']:,}",
        f"[bold]${metadata['total_cost']:.4f}[/bold]",
    )

    console.print(table)
```

## Error Display

Errors are shown inline during execution and summarized at the end:

```
[!] Search failed for subtopic "Evaluation frameworks": Rate limit exceeded. Retrying in 5s...
[!] Scrape failed for https://example.com: Timeout after 30s. Skipping URL.
```

Fatal errors that halt the pipeline show the run ID for resumption:

```
[ERROR] Budget exceeded ($2.00 limit). Generating partial report.
        Resume with: research-agent --resume abc123
```

## File Location

```
src/research_agent/
    cli.py            # Typer app, main(), commands
    display.py        # Rich progress, tables, cost summary
    shutdown.py       # GracefulShutdown handler
```
