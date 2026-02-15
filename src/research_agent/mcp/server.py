"""MCP protocol server core implementation."""

from __future__ import annotations

from typing import Any

from research_agent import __version__
from research_agent.config import Settings
from research_agent.mcp.models import (
    MCPError,
    MCPInitializeParams,
    MCPRequest,
    MCPResourceListParams,
    MCPResourceReadParams,
    MCPResponse,
    MCPServerInfo,
    MCPToolCallParams,
)
from research_agent.mcp.resources import MCPResourceProvider
from research_agent.mcp.tools import MCPToolRegistry


class MCPServer:
    """Handle MCP protocol handshake, tools, and resources."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.load()
        self._tools = MCPToolRegistry(self._settings)
        self._resources = MCPResourceProvider(self._settings, self._tools)

    def capabilities(self) -> dict[str, Any]:
        """Capabilities advertisement payload."""
        return {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True},
            "transports": ["stdio", "sse"],
        }

    def handle_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Process one MCP request and return response payload."""
        try:
            request = MCPRequest.model_validate(request_payload)
        except Exception as exc:
            return MCPResponse(
                id=None,
                error=MCPError(code=-32600, message=f"Invalid request: {exc}"),
            ).model_dump()

        try:
            result = self._dispatch(request)
            return MCPResponse(id=request.id, result=result).model_dump()
        except ValueError as exc:
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32602, message=str(exc)),
            ).model_dump()
        except Exception as exc:
            return MCPResponse(
                id=request.id,
                error=MCPError(code=-32000, message=f"Server error: {exc}"),
            ).model_dump()

    def _dispatch(self, request: MCPRequest) -> dict[str, Any]:
        method = request.method

        if method == "initialize":
            _ = MCPInitializeParams.model_validate(request.params)
            info = MCPServerInfo(
                version=__version__,
                capabilities=self.capabilities(),
            )
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": info.model_dump(),
            }
        elif method == "tools/list":
            return {"tools": [tool.model_dump() for tool in self._tools.list_tools()]}
        elif method == "tools/call":
            tool_params = MCPToolCallParams.model_validate(request.params)
            return {"content": self._tools.call_tool(tool_params)}
        elif method == "resources/list":
            list_params = MCPResourceListParams.model_validate(request.params)
            return self._resources.list_resources(
                uri_prefix=list_params.uri_prefix,
                page=list_params.page,
                page_size=list_params.page_size,
            )
        elif method == "resources/read":
            read_params = MCPResourceReadParams.model_validate(request.params)
            return self._resources.read_resource(read_params)

        raise ValueError(f"Unsupported method: {method}")
