"""Bridge between AgentPrompts project folders and research outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from research_agent.agentprompts.build_prompt import (
    align_for_build_phase,
    load_build_prompt,
    quality_gate,
)
from research_agent.compiled_output import format_compiled_research
from research_agent.prompt_parser import load_research_prompt

if TYPE_CHECKING:
    from pathlib import Path

    from research_agent.agentprompts.registry import ProjectRegistry


@dataclass(slots=True)
class ForProjectResult:
    """Result payload for `for-project` execution."""

    project_name: str
    project_path: Path
    output_path: Path
    status_path: Path
    quality_gate_passed: bool
    missing_sections: list[str]


def resolve_project_path(
    project_name: str,
    projects_dir: Path,
    registry: ProjectRegistry,
) -> Path:
    """Resolve project directory from registry or direct discovery."""
    registered = registry.resolve(project_name)
    if registered and (registered / "RESEARCH_PROMPT.md").exists():
        return registered

    direct = projects_dir / project_name
    if (direct / "RESEARCH_PROMPT.md").exists():
        return direct

    lowered = project_name.strip().lower()
    for path in projects_dir.glob("*"):
        if not path.is_dir():
            continue
        if path.name.lower() != lowered:
            continue
        if (path / "RESEARCH_PROMPT.md").exists():
            return path

    raise FileNotFoundError(f"Project '{project_name}' not found in {projects_dir}")


def run_for_project(
    project_name: str,
    projects_dir: Path,
    registry: ProjectRegistry,
    output_filename: str = "COMPILED_RESEARCH.md",
) -> ForProjectResult:
    """Generate COMPILED_RESEARCH.md for a named AgentPrompts project."""
    project_path = resolve_project_path(project_name, projects_dir, registry)
    registry.register(project_name, project_path)

    research_prompt_path = project_path / "RESEARCH_PROMPT.md"
    parsed_prompt = load_research_prompt(research_prompt_path)

    source_report = _draft_report(parsed_prompt.raw_text)
    query = parsed_prompt.topic or f"{project_name} research"
    compiled = format_compiled_research(
        report=source_report,
        query=query,
        metadata={"project_name": project_name},
    )

    build_prompt = load_build_prompt(project_path / "BUILD_PROMPT.md")
    compiled = align_for_build_phase(compiled, build_prompt)
    gate_ok, missing = quality_gate(compiled)

    output_path = project_path / output_filename
    output_path.write_text(compiled, encoding="utf-8")

    status_path = project_path / ".research-status.json"
    status_payload = {
        "project": project_name,
        "project_path": str(project_path),
        "output_path": str(output_path),
        "status": "completed" if gate_ok else "failed_quality_gate",
        "missing_sections": missing,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    return ForProjectResult(
        project_name=project_name,
        project_path=project_path,
        output_path=output_path,
        status_path=status_path,
        quality_gate_passed=gate_ok,
        missing_sections=missing,
    )


def _draft_report(prompt_text: str) -> str:
    """Construct a normalized report skeleton from RESEARCH_PROMPT content."""
    lines = [
        "## Executive Summary",
        "This report was generated from RESEARCH_PROMPT.md and aligned for build handoff.",
        "",
        "## Key Findings",
        "- Core requirements were parsed from the prompt file.",
        "- Constraints and output requirements were preserved in this handoff report.",
        "",
        "## Detailed Analysis",
        prompt_text.strip() or "No prompt content supplied.",
        "",
        "## Technical Considerations",
        "- Validate feasibility of recommended dependencies before implementation.",
        "- Prioritize low-risk adoption paths for production rollout.",
        "",
        "## Sources",
        "- RESEARCH_PROMPT.md",
        "- BUILD_PROMPT.md (if present)",
    ]
    return "\n".join(lines).strip() + "\n"
