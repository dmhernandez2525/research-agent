"""MCP resource listing and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_agent.mcp.models import MCPResourceReadParams, ResourceInfo

if TYPE_CHECKING:
    from research_agent.config import Settings
    from research_agent.mcp.tools import MCPToolRegistry


class MCPResourceProvider:
    """Expose reports, sessions, and memory over MCP resource URIs."""

    def __init__(self, settings: Settings, tools: MCPToolRegistry) -> None:
        self._settings = settings
        self._tools = tools

    def list_resources(
        self,
        uri_prefix: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """List available MCP resource namespaces with pagination."""
        resources = [
            ResourceInfo(
                uri="reports://",
                name="Reports",
                description="Completed markdown reports from CLI/API/MCP sessions.",
                mime_type="text/markdown",
            ),
            ResourceInfo(
                uri="sessions://",
                name="Sessions",
                description="Session status and metadata exposed as JSON.",
                mime_type="application/json",
            ),
            ResourceInfo(
                uri="memory://",
                name="Memory",
                description="Cross-session knowledge entries and findings.",
                mime_type="application/json",
            ),
        ]

        if uri_prefix:
            resources = [item for item in resources if item.uri.startswith(uri_prefix)]

        start = (page - 1) * page_size
        end = start + page_size
        next_page = page + 1 if end < len(resources) else None

        return {
            "items": [item.model_dump() for item in resources[start:end]],
            "page": page,
            "page_size": page_size,
            "next_page": next_page,
        }

    def read_resource(self, params: MCPResourceReadParams) -> dict[str, Any]:
        """Read a resource URI with pagination and MIME negotiation."""
        uri = params.uri
        if uri.startswith("reports://"):
            return self._read_reports(uri, params.page, params.page_size, params.accept)
        if uri.startswith("sessions://"):
            return self._read_sessions(params.page, params.page_size)
        if uri.startswith("memory://"):
            return self._read_memory(params.page, params.page_size)
        raise ValueError(f"Unknown resource URI: {uri}")

    def _read_reports(
        self,
        uri: str,
        page: int,
        page_size: int,
        accept: str,
    ) -> dict[str, Any]:
        report_dir = Path(self._settings.report.output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            report_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True
        )

        if uri != "reports://":
            filename = uri.replace("reports://", "", 1)
            target = report_dir / filename
            if not target.exists():
                raise ValueError(f"Report not found: {filename}")
            content = target.read_text(encoding="utf-8")
            if accept == "application/json":
                return {
                    "uri": uri,
                    "mime_type": "application/json",
                    "content": {"filename": filename, "content": content},
                }
            return {
                "uri": uri,
                "mime_type": "text/markdown",
                "content": content,
            }

        start = (page - 1) * page_size
        end = start + page_size
        sliced = files[start:end]
        next_page = page + 1 if end < len(files) else None

        return {
            "uri": "reports://",
            "mime_type": "application/json",
            "content": [path.name for path in sliced],
            "page": page,
            "next_page": next_page,
        }

    def _read_sessions(self, page: int, page_size: int) -> dict[str, Any]:
        states = self._tools.session_states()
        session_items = [
            {
                "session_id": session_id,
                "status": state.status,
                "progress": state.progress,
                "cost_usd": state.cost_usd,
            }
            for session_id, state in states.items()
        ]

        start = (page - 1) * page_size
        end = start + page_size
        sliced = session_items[start:end]

        return {
            "uri": "sessions://",
            "mime_type": "application/json",
            "content": sliced,
            "page": page,
            "next_page": page + 1 if end < len(session_items) else None,
        }

    def _read_memory(self, page: int, page_size: int) -> dict[str, Any]:
        memory_path = (
            Path(self._settings.vector_store.persist_directory) / "enhancement.json"
        )
        if not memory_path.exists():
            payload: dict[str, Any] = {"projects": {}}
        else:
            payload = json.loads(memory_path.read_text(encoding="utf-8"))

        items: list[dict[str, Any]] = []
        projects = payload.get("projects", {})
        if isinstance(projects, dict):
            for project_id, topics in projects.items():
                if not isinstance(topics, dict):
                    continue
                for topic, entry in topics.items():
                    if not isinstance(entry, dict):
                        continue
                    items.append(
                        {
                            "project_id": project_id,
                            "topic": topic,
                            "entry": entry,
                        }
                    )

        start = (page - 1) * page_size
        end = start + page_size

        return {
            "uri": "memory://",
            "mime_type": "application/json",
            "content": items[start:end],
            "page": page,
            "next_page": page + 1 if end < len(items) else None,
        }
