"""CLI tests for MCP serve and benchmark commands."""

from __future__ import annotations

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Settings

runner = CliRunner()


def test_mcp_help_lists_commands() -> None:
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.output
    assert "benchmark" in result.output


def test_mcp_benchmark_renders_table(
    monkeypatch: object,
) -> None:
    settings = Settings()
    monkeypatch.setattr("research_agent.cli._load_settings", lambda *_a, **_k: settings)
    monkeypatch.setattr(
        "research_agent.cli.benchmark_tool_latency",
        lambda _server, query: {
            "query": query,
            "session_id": "mcp-bench-1",
            "latency_ms": 12.5,
        },
    )

    result = runner.invoke(app, ["mcp", "benchmark", "--query", "demo query"])
    assert result.exit_code == 0
    assert "MCP Benchmark" in result.output
    assert "mcp-bench-1" in result.output


def test_mcp_serve_stdio_invokes_runtime(
    monkeypatch: object,
) -> None:
    settings = Settings()
    called = {"stdio": False}

    def run_stdio(_settings: Settings) -> None:
        called["stdio"] = True

    monkeypatch.setattr("research_agent.cli._load_settings", lambda *_a, **_k: settings)
    monkeypatch.setattr("research_agent.cli.run_stdio_server", run_stdio)

    result = runner.invoke(app, ["mcp", "serve", "--transport", "stdio"])
    assert result.exit_code == 0
    assert called["stdio"] is True


def test_mcp_serve_sse_invokes_runtime(
    monkeypatch: object,
) -> None:
    settings = Settings()
    called: dict[str, object] = {}

    def run_sse(_settings: Settings, *, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr("research_agent.cli._load_settings", lambda *_a, **_k: settings)
    monkeypatch.setattr("research_agent.cli.run_sse_server", run_sse)

    result = runner.invoke(
        app,
        ["mcp", "serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "9001"],
    )
    assert result.exit_code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9001
