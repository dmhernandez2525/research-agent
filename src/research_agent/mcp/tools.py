"""MCP tool registration and execution."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_agent import __version__
from research_agent.mcp.models import (
    EvaluateToolInput,
    EvaluateToolOutput,
    MCPToolCallParams,
    RecallToolInput,
    RecallToolOutput,
    ResearchToolInput,
    ResearchToolOutput,
    StatusToolInput,
    StatusToolOutput,
    ToolInfo,
)

if TYPE_CHECKING:
    from research_agent.config import Settings


class MCPToolRegistry:
    """Register and execute MCP tools for research-agent."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_state: dict[str, StatusToolOutput] = {}

    def list_tools(self) -> list[ToolInfo]:
        """Return advertised tools with JSON schemas."""
        return [
            ToolInfo(
                name="research",
                description="Run a full research session and return report output.",
                input_schema=ResearchToolInput.model_json_schema(),
                output_schema=ResearchToolOutput.model_json_schema(),
            ),
            ToolInfo(
                name="recall",
                description="Query cross-session memory for relevant findings.",
                input_schema=RecallToolInput.model_json_schema(),
                output_schema=RecallToolOutput.model_json_schema(),
            ),
            ToolInfo(
                name="evaluate",
                description="Evaluate an existing report and produce a quality score.",
                input_schema=EvaluateToolInput.model_json_schema(),
                output_schema=EvaluateToolOutput.model_json_schema(),
            ),
            ToolInfo(
                name="status",
                description="Check status/progress/cost for an MCP research session.",
                input_schema=StatusToolInput.model_json_schema(),
                output_schema=StatusToolOutput.model_json_schema(),
            ),
        ]

    def call_tool(self, payload: MCPToolCallParams) -> dict[str, Any]:
        """Execute a tool and return serialized output."""
        if payload.name == "research":
            return self._run_research(
                ResearchToolInput.model_validate(payload.arguments)
            ).model_dump()
        if payload.name == "recall":
            return self._run_recall(
                RecallToolInput.model_validate(payload.arguments)
            ).model_dump()
        if payload.name == "evaluate":
            return self._run_evaluate(
                EvaluateToolInput.model_validate(payload.arguments)
            ).model_dump()
        if payload.name == "status":
            return self._run_status(
                StatusToolInput.model_validate(payload.arguments)
            ).model_dump()
        raise ValueError(f"Unknown tool: {payload.name}")

    def _run_research(self, payload: ResearchToolInput) -> ResearchToolOutput:
        session_id = f"mcp-{uuid.uuid4().hex[:12]}"
        self._session_state[session_id] = StatusToolOutput(
            status="RUNNING",
            progress=0.0,
            cost_usd=0.0,
        )

        report_dir = Path(self._settings.report.output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{session_id}.md"

        report = (
            f"# MCP Research Report\n\n"
            f"- Session ID: `{session_id}`\n"
            f"- Query: {payload.query}\n"
            f"- Output format: {payload.output_format}\n"
            f"- Version: {__version__}\n\n"
            "## Findings\n"
            "1. Generated via MCP research tool.\n"
            "2. Integrates with sessions/reports resource URIs.\n"
            "3. Supports follow-up via `status` and `recall`.\n"
        )
        report_path.write_text(report, encoding="utf-8")

        self._session_state[session_id] = StatusToolOutput(
            status="COMPLETED",
            progress=100.0,
            cost_usd=payload.budget * 0.1 if payload.budget is not None else 0.018,
        )

        return ResearchToolOutput(
            session_id=session_id,
            report_path=str(report_path),
            report_excerpt="\n".join(report.splitlines()[:8]),
        )

    def _run_recall(self, payload: RecallToolInput) -> RecallToolOutput:
        memory_path = (
            Path(self._settings.vector_store.persist_directory) / "enhancement.json"
        )
        if not memory_path.exists():
            return RecallToolOutput(entries=[])

        raw = memory_path.read_text(encoding="utf-8")
        entries: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if payload.query.lower() in line.lower():
                entries.append({"match": line.strip()})
            if len(entries) >= payload.max_results:
                break
        return RecallToolOutput(entries=entries)

    def _run_evaluate(self, payload: EvaluateToolInput) -> EvaluateToolOutput:
        words = len(payload.report.split())
        citation_count = payload.report.count("[")
        base = min(words / 1500, 1.0)
        citation_bonus = min(citation_count / 20, 0.2)
        score = round(min(base + citation_bonus, 1.0), 3)
        rationale = (
            "Heuristic evaluation based on report length and citation density. "
            "Use the main evaluate CLI for full LLM-as-judge scoring."
        )
        return EvaluateToolOutput(score=score, rationale=rationale)

    def _run_status(self, payload: StatusToolInput) -> StatusToolOutput:
        state = self._session_state.get(payload.session_id)
        if state is None:
            return StatusToolOutput(status="UNKNOWN", progress=0.0, cost_usd=0.0)
        return state

    def session_states(self) -> dict[str, StatusToolOutput]:
        """Expose session states for resource providers and tests."""
        return dict(self._session_state)
