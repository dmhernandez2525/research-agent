"""Tests for MCP serve runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from research_agent.config import Settings
from research_agent.mcp.serve import benchmark_tool_latency, create_sse_app
from research_agent.mcp.server import MCPServer

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.report.output_dir = tmp_path / "reports"
    settings.vector_store.persist_directory = tmp_path / "vector"
    return settings


def test_benchmark_reports_latency_and_session(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))
    result = benchmark_tool_latency(server, query="benchmark query")
    assert result["latency_ms"] >= 0.0
    assert str(result["session_id"]).startswith("mcp-")


def test_sse_app_request_and_event_stream(tmp_path: Path) -> None:
    app = create_sse_app(_settings(tmp_path))

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200

        response = client.post(
            "/mcp/request",
            json={
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocol_version": "2024-11-05",
                    "client_name": "pytest",
                    "client_version": "1.0",
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["error"] is None

        with client.stream("GET", "/mcp/events") as stream:
            assert stream.status_code == 200
            found = False
            for line in stream.iter_lines():
                if "initialize" in line or "serverInfo" in line:
                    found = True
                    break
            assert found is True
