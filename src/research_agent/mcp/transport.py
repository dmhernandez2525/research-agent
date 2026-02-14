"""MCP transport helpers for stdio and SSE usage."""

from __future__ import annotations

import json
from collections import deque
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.mcp.server import MCPServer


def run_stdio_once(server: MCPServer, line: str) -> str:
    """Process a single stdio JSON request line."""
    payload = json.loads(line)
    response = server.handle_request(payload)
    return json.dumps(response)


def run_stdio_loop(
    server: MCPServer, input_stream: IO[str], output_stream: IO[str]
) -> None:
    """Process stdio requests until EOF."""
    for line in input_stream:
        stripped = line.strip()
        if not stripped:
            continue
        output_stream.write(run_stdio_once(server, stripped) + "\n")
        output_stream.flush()


class SSETransportBuffer:
    """In-memory SSE buffer for remote MCP clients."""

    def __init__(self, max_events: int = 200) -> None:
        self._events: deque[dict[str, object]] = deque(maxlen=max_events)
        self._next_id = 1

    def publish(self, payload: dict[str, object]) -> dict[str, object]:
        """Store and return an SSE-formatted event envelope."""
        event = {
            "id": self._next_id,
            "event": "message",
            "data": payload,
        }
        self._events.append(event)
        self._next_id += 1
        return event

    def stream(self, last_event_id: int | None = None) -> list[dict[str, object]]:
        """Return buffered events for SSE replay or catch-up."""
        if last_event_id is None:
            return list(self._events)
        return [event for event in self._events if int(event["id"]) > last_event_id]
