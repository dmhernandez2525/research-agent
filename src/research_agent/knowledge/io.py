"""Knowledge base export and import helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from research_agent.knowledge.models import KnowledgeExportPayload

if TYPE_CHECKING:
    from pathlib import Path


def export_to_json(path: Path, payload: KnowledgeExportPayload) -> None:
    """Write structured knowledge export payload to disk."""
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def import_from_json(path: Path) -> KnowledgeExportPayload:
    """Load knowledge payload from JSON file."""
    content = json.loads(path.read_text(encoding="utf-8"))
    return KnowledgeExportPayload.model_validate(content)


def export_to_markdown(payload: KnowledgeExportPayload) -> str:
    """Render a human-readable markdown knowledge dump."""
    lines: list[str] = ["# Knowledge Base Export", ""]

    lines.append("## Findings")
    if not payload.findings:
        lines.append("- No findings available.")
    for finding in payload.findings:
        lines.append(f"- **{finding.topic}** ({finding.confidence:.2f})")
        lines.append(f"  - {finding.statement}")
        if finding.sources:
            lines.append(f"  - Sources: {', '.join(finding.sources)}")

    lines.extend(["", "## Relationships"])
    if not payload.relationships:
        lines.append("- No relationships available.")
    for rel in payload.relationships:
        lines.append(f"- {rel.source} --{rel.relation.value}--> {rel.target}")

    lines.extend(["", "## Refresh History"])
    if not payload.refresh_history:
        lines.append("- No refresh events available.")
    for record in payload.refresh_history:
        lines.append(
            f"- {record.topic} ({record.triggered_at}): {record.change_summary} "
            f"[{record.old_confidence:.2f} -> {record.new_confidence:.2f}]"
        )

    return "\n".join(lines).strip() + "\n"
