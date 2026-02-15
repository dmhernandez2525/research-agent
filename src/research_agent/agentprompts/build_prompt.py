"""BUILD_PROMPT parsing and COMPILED_RESEARCH alignment helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_REQUIRED_COMPILED_SECTIONS = (
    "## Executive Summary",
    "## Key Findings",
    "## Detailed Analysis",
    "## Technical Considerations",
    "## Sources",
    "## Methodology",
)


def load_build_prompt(path: Path) -> str:
    """Load BUILD_PROMPT.md content if present."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def extract_build_tasks(build_prompt: str) -> list[str]:
    """Extract task bullets from BUILD_PROMPT markdown."""
    tasks: list[str] = []
    for line in build_prompt.splitlines():
        stripped = line.strip()
        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)
        if bullet:
            tasks.append(bullet.group(1).strip())
            continue
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ordered:
            tasks.append(ordered.group(1).strip())
    return tasks


def align_for_build_phase(compiled_research: str, build_prompt: str) -> str:
    """Append build-phase alignment details into compiled research."""
    tasks = extract_build_tasks(build_prompt)
    if not tasks:
        return compiled_research

    lines = [compiled_research.rstrip(), "", "## Build Phase Alignment", ""]
    lines.append("This section maps research findings to BUILD_PROMPT execution tasks.")
    lines.append("")
    lines.append("### Referenced Build Tasks")
    for task in tasks[:25]:
        lines.append(f"- {task}")
    lines.append("")
    lines.append("### Implementation-Ready Guidance")
    lines.append("- Apply findings in priority order from Key Findings.")
    lines.append(
        "- Reuse cited dependencies and snippets from Technical Considerations."
    )
    lines.append("- Validate assumptions against Sources before final implementation.")
    return "\n".join(lines).rstrip() + "\n"


def quality_gate(compiled_research: str) -> tuple[bool, list[str]]:
    """Validate required sections exist before build handoff."""
    missing = [
        section
        for section in _REQUIRED_COMPILED_SECTIONS
        if section not in compiled_research
    ]
    return (len(missing) == 0, missing)
