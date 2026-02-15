"""Runtime serving utilities for MCP stdio and SSE transports."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from research_agent.mcp.server import MCPServer
from research_agent.mcp.transport import SSETransportBuffer, run_stdio_loop

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from research_agent.config import Settings


def run_stdio_server(settings: Settings) -> None:
    """Run MCP server over stdio transport."""
    import sys

    server = MCPServer(settings)
    run_stdio_loop(server, sys.stdin, sys.stdout)


def create_sse_app(settings: Settings) -> FastAPI:
    """Create FastAPI app for MCP over HTTP + SSE streaming."""
    from fastapi import FastAPI, Header
    from fastapi.responses import StreamingResponse

    server = MCPServer(settings)
    buffer = SSETransportBuffer(max_events=500)
    app = FastAPI(title="research-agent MCP", version="0.1.0")

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def request_endpoint(payload: dict[str, object]) -> dict[str, object]:
        response = server.handle_request(payload)
        buffer.publish(response)
        return response

    async def events(
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        parsed = (
            int(last_event_id) if last_event_id and last_event_id.isdigit() else None
        )

        async def stream() -> AsyncIterator[str]:
            while True:
                if await request.is_disconnected():
                    break
                events_payload = buffer.stream(parsed)
                for event in events_payload:
                    yield (
                        f"id: {event['id']}\n"
                        f"event: {event['event']}\n"
                        f"data: {json.dumps(event['data'])}\n\n"
                    )
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/mcp/request", request_endpoint, methods=["POST"])
    app.add_api_route("/mcp/events", events, methods=["GET"])

    return app


def run_sse_server(settings: Settings, host: str, port: int) -> None:
    """Run MCP server over SSE transport."""
    import uvicorn

    app = create_sse_app(settings)
    uvicorn.run(app, host=host, port=port, log_level="info")


def benchmark_tool_latency(server: MCPServer, query: str) -> dict[str, float | str]:
    """Measure latency from tool call dispatch to first result payload."""
    start = time.perf_counter()
    response = server.handle_request(
        {
            "id": "bench-1",
            "method": "tools/call",
            "params": {"name": "research", "arguments": {"query": query}},
        }
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    error = response.get("error")
    if error is not None:
        raise RuntimeError(f"MCP benchmark failed: {error}")
    content = response.get("result", {}).get("content", {})
    session_id = str(content.get("session_id", ""))
    return {
        "query": query,
        "session_id": session_id,
        "latency_ms": round(elapsed_ms, 2),
    }
