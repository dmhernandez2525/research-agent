"""MCP protocol models and tool schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPError(BaseModel):
    """JSON-RPC error payload."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class MCPRequest(BaseModel):
    """Incoming MCP request payload."""

    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """Outgoing MCP response payload."""

    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: MCPError | None = None


class ToolInfo(BaseModel):
    """Advertised MCP tool descriptor."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class ResourceInfo(BaseModel):
    """Advertised MCP resource descriptor."""

    uri: str
    name: str
    description: str
    mime_type: str


class MCPInitializeParams(BaseModel):
    """Initialization parameters from client."""

    protocol_version: str = "2024-11-05"
    client_name: str = "unknown"
    client_version: str = "0.0.0"


class MCPServerInfo(BaseModel):
    """Server identity and capability advertisement."""

    name: str = "research-agent"
    version: str
    capabilities: dict[str, Any]


class MCPToolCallParams(BaseModel):
    """Tool call request params."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPResourceReadParams(BaseModel):
    """Resource read request params."""

    uri: str
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=500)
    accept: str = "application/json"


class MCPResourceListParams(BaseModel):
    """Resource list request params."""

    uri_prefix: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=500)


class ResearchToolInput(BaseModel):
    """Input schema for MCP `research` tool."""

    query: str = Field(min_length=1)
    budget: float | None = Field(default=None, gt=0)
    output_format: str = Field(default="md", pattern="^(md|pdf)$")


class ResearchToolOutput(BaseModel):
    """Output schema for MCP `research` tool."""

    session_id: str
    report_path: str
    report_excerpt: str


class RecallToolInput(BaseModel):
    """Input schema for MCP `recall` tool."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)


class RecallToolOutput(BaseModel):
    """Output schema for MCP `recall` tool."""

    entries: list[dict[str, Any]]


class EvaluateToolInput(BaseModel):
    """Input schema for MCP `evaluate` tool."""

    report: str = Field(min_length=1)
    query: str = ""


class EvaluateToolOutput(BaseModel):
    """Output schema for MCP `evaluate` tool."""

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class StatusToolInput(BaseModel):
    """Input schema for MCP `status` tool."""

    session_id: str = Field(min_length=1)


class StatusToolOutput(BaseModel):
    """Output schema for MCP `status` tool."""

    status: str
    progress: float = Field(ge=0.0, le=100.0)
    cost_usd: float = Field(ge=0.0)
