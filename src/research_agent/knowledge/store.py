"""Persistence layer for knowledge findings and graph relationships."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from research_agent.knowledge.models import (
    KnowledgeExportPayload,
    KnowledgeFinding,
    KnowledgeRefreshRecord,
    KnowledgeRelationship,
)

if TYPE_CHECKING:
    from pathlib import Path


class KnowledgeStore:
    """JSON-backed knowledge store with export/import utilities."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> KnowledgeExportPayload:
        if not self._path.exists():
            return KnowledgeExportPayload()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return KnowledgeExportPayload.model_validate(payload)

    def save(self, payload: KnowledgeExportPayload) -> None:
        self._path.write_text(
            payload.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def upsert_findings(self, findings: list[KnowledgeFinding]) -> None:
        payload = self.load()
        existing = {finding.id: finding for finding in payload.findings}
        for finding in findings:
            existing[finding.id] = finding
        payload.findings = sorted(existing.values(), key=lambda item: item.id)
        self.save(payload)

    def set_relationships(self, relationships: list[KnowledgeRelationship]) -> None:
        payload = self.load()
        payload.relationships = relationships
        self.save(payload)

    def append_refresh(self, record: KnowledgeRefreshRecord) -> None:
        payload = self.load()
        payload.refresh_history.append(record)
        self.save(payload)

    def export_filtered(
        self,
        topic: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_confidence: float = 0.0,
    ) -> KnowledgeExportPayload:
        payload = self.load()
        findings = payload.findings

        if topic:
            lowered = topic.lower()
            findings = [item for item in findings if lowered in item.topic.lower()]

        if date_from:
            start = datetime.fromisoformat(date_from)
            findings = [
                item
                for item in findings
                if datetime.fromisoformat(item.updated_at) >= start
            ]

        if date_to:
            end = datetime.fromisoformat(date_to)
            findings = [
                item
                for item in findings
                if datetime.fromisoformat(item.updated_at) <= end
            ]

        findings = [item for item in findings if item.confidence >= min_confidence]
        ids = {item.id for item in findings}
        relationships = [
            rel
            for rel in payload.relationships
            if rel.source in ids and rel.target in ids
        ]

        return KnowledgeExportPayload(
            findings=findings,
            relationships=relationships,
            refresh_history=payload.refresh_history,
        )

    def import_payload(self, incoming: KnowledgeExportPayload) -> dict[str, int]:
        payload = self.load()
        findings = {item.id: item for item in payload.findings}
        merged = 0
        conflicts = 0

        for finding in incoming.findings:
            existing = findings.get(finding.id)
            if existing is None:
                findings[finding.id] = finding
                merged += 1
                continue

            if existing.statement != finding.statement:
                conflicts += 1
                if finding.confidence >= existing.confidence:
                    findings[finding.id] = finding
            else:
                if finding.confidence > existing.confidence:
                    findings[finding.id] = finding

        relationship_keys = {
            (rel.source, rel.target, rel.relation.value): rel
            for rel in payload.relationships
        }
        for rel in incoming.relationships:
            relationship_keys[(rel.source, rel.target, rel.relation.value)] = rel

        payload.findings = sorted(findings.values(), key=lambda item: item.id)
        payload.relationships = list(relationship_keys.values())
        payload.refresh_history.extend(incoming.refresh_history)
        self.save(payload)

        return {"merged": merged, "conflicts": conflicts}
