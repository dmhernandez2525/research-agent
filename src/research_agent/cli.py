"""Typer CLI entry point for the research-agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from research_agent import __version__

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(
    name="research-agent",
    help="Crash-resilient deep research agent for the Apps That Build Apps ecosystem.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"research-agent {__version__}")
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
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume from latest checkpoint."),
    ] = False,
    budget: Annotated[
        float | None,
        typer.Option("--budget", "-b", help="Max cost budget in USD."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging."),
    ] = False,
) -> None:
    """Run a deep-research pipeline for the given query.

    Args:
        query: The research query to investigate.
        config: Path to config YAML file.
        output: Output directory for the report.
        resume: Resume from latest checkpoint.
        budget: Max cost budget in USD.
        verbose: Enable verbose logging.
    """
    ...


@app.command()
def resume(
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Checkpoint directory to resume from."),
    ] = None,
) -> None:
    """Resume a previously interrupted research run from the latest checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory to resume from.
    """
    ...


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
    """Evaluate a generated report using LLM-as-judge scoring.

    Args:
        report: Path to the generated report to evaluate.
        query: Original research query for evaluation context.
    """
    ...


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
    """Clean up generated data, checkpoints, and caches.

    Args:
        checkpoints: Remove all checkpoint files.
        cache: Remove ChromaDB cache.
        all_data: Remove all generated data.
    """
    ...


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
