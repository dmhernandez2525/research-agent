"""Tests for MCP protocol server, tools, and resources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.config import Settings
from research_agent.mcp.client_example import MCPClientExample
from research_agent.mcp.server import MCPServer
from research_agent.mcp.transport import SSETransportBuffer, run_stdio_once

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.report.output_dir = tmp_path / "reports"
    settings.vector_store.persist_directory = tmp_path / "vector"
    return settings


def test_initialize_handshake_and_capabilities(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))
    response = server.handle_request(
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "protocol_version": "2024-11-05",
                "client_name": "pytest",
                "client_version": "1.0.0",
            },
        }
    )

    assert response["error"] is None
    result = response["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "research-agent"
    assert "tools" in result["serverInfo"]["capabilities"]


def test_invalid_method_returns_error(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))
    response = server.handle_request({"id": 1, "method": "unknown", "params": {}})
    assert response["error"] is not None
    assert response["error"]["code"] == -32602


def test_tool_listing_and_research_status_roundtrip(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))

    listed = server.handle_request({"id": 2, "method": "tools/list", "params": {}})
    assert listed["error"] is None
    names = {item["name"] for item in listed["result"]["tools"]}
    assert {"research", "recall", "evaluate", "status"}.issubset(names)

    research = server.handle_request(
        {
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "research",
                "arguments": {"query": "MCP integration", "output_format": "md"},
            },
        }
    )
    session_id = research["result"]["content"]["session_id"]

    status = server.handle_request(
        {
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "status",
                "arguments": {"session_id": session_id},
            },
        }
    )
    assert status["result"]["content"]["status"] == "COMPLETED"
    assert status["result"]["content"]["progress"] == 100.0


def test_resource_listing_and_report_read(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))

    _ = server.handle_request(
        {
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "research",
                "arguments": {"query": "resource read", "output_format": "md"},
            },
        }
    )

    listed = server.handle_request(
        {
            "id": 2,
            "method": "resources/list",
            "params": {"uri_prefix": "reports://", "page": 1, "page_size": 10},
        }
    )
    assert listed["error"] is None
    assert listed["result"]["items"][0]["uri"] == "reports://"

    reports = server.handle_request(
        {
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "reports://", "page": 1, "page_size": 10},
        }
    )
    assert reports["error"] is None
    names = reports["result"]["content"]
    assert names

    first = str(names[0])
    detail = server.handle_request(
        {
            "id": 4,
            "method": "resources/read",
            "params": {
                "uri": f"reports://{first}",
                "page": 1,
                "page_size": 10,
                "accept": "text/markdown",
            },
        }
    )
    assert detail["result"]["mime_type"] == "text/markdown"
    assert "MCP Research Report" in detail["result"]["content"]


def test_stdio_sse_and_client_example(tmp_path: Path) -> None:
    server = MCPServer(_settings(tmp_path))
    response_text = run_stdio_once(
        server,
        '{"id":1,"method":"initialize","params":{"client_name":"x","client_version":"1","protocol_version":"2024-11-05"}}',
    )
    assert '"error": null' in response_text

    sse = SSETransportBuffer(max_events=5)
    sse.publish({"method": "tools/list"})
    sse.publish({"method": "resources/list"})
    assert len(sse.stream()) == 2
    assert len(sse.stream(last_event_id=1)) == 1

    client = MCPClientExample(server=server)
    init = client.handshake()
    assert init["error"] is None
