"""Example MCP client for integration testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from research_agent.mcp.server import MCPServer


@dataclass
class MCPClientExample:
    """Tiny in-process MCP client exercising core round-trip flow."""

    server: MCPServer

    def call(
        self, method: str, params: dict[str, Any], request_id: int = 1
    ) -> dict[str, Any]:
        payload = {
            "id": request_id,
            "method": method,
            "params": params,
        }
        return self.server.handle_request(payload)

    def handshake(self) -> dict[str, Any]:
        return self.call(
            "initialize",
            {
                "protocol_version": "2024-11-05",
                "client_name": "example-client",
                "client_version": "0.1.0",
            },
        )


def run_demo() -> str:
    """Run an example handshake + tool list and return JSON text."""
    client = MCPClientExample(server=MCPServer())
    init = client.handshake()
    tools = client.call("tools/list", {}, request_id=2)
    return json.dumps({"initialize": init, "tools": tools}, indent=2)


if __name__ == "__main__":
    print(run_demo())
