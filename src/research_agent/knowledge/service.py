"""Service layer for knowledge synthesis, refresh, and query workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from research_agent.knowledge.decay import (
    apply_confidence_decay,
    create_refresh_record,
    should_trigger_research,
)
from research_agent.knowledge.models import (
    KnowledgeFinding,
    KnowledgeRelationship,
    RelationshipType,
)
from research_agent.knowledge.synthesis import (
    consolidate_findings,
    detect_conflicts,
    score_confidence,
    summarize_by_cluster,
)

if TYPE_CHECKING:
    from research_agent.knowledge.store import KnowledgeStore

_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")


@dataclass(slots=True)
class KnowledgeSummary:
    """Summarized knowledge view for CLI rendering."""

    findings: list[KnowledgeFinding]
    conflicts: list[tuple[KnowledgeFinding, KnowledgeFinding]]
    cluster_summaries: dict[str, str]
    due_for_refresh_ids: list[str]


class KnowledgeService:
    """High-level knowledge operations used by CLI commands."""

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    def summarize(
        self,
        topic: str | None = None,
        threshold: float = 0.45,
        refresh_days: int = 30,
    ) -> KnowledgeSummary:
        """Consolidate, rescore, and summarize knowledge findings."""
        payload = self._store.load()
        filtered = [item for item in payload.findings if self._topic_match(item, topic)]
        consolidated = consolidate_findings(filtered)

        rescored = [
            finding.model_copy(update={"confidence": score_confidence(finding)})
            for finding in consolidated
        ]
        conflicts = detect_conflicts(rescored)

        refresh_schedule = {finding.topic.lower(): refresh_days for finding in rescored}
        due_ids = [
            finding.id
            for finding in rescored
            if should_trigger_research(finding, threshold, refresh_schedule)
        ]

        return KnowledgeSummary(
            findings=rescored,
            conflicts=conflicts,
            cluster_summaries=summarize_by_cluster(rescored),
            due_for_refresh_ids=due_ids,
        )

    def refresh_topic(
        self,
        topic: str,
        threshold: float = 0.45,
        refresh_days: int = 30,
        new_statement: str | None = None,
    ) -> int:
        """Refresh findings for a topic when decay or schedule triggers."""
        payload = self._store.load()
        decayed = apply_confidence_decay(payload.findings)
        payload.findings = decayed

        refreshed_count = 0
        now = datetime.now(tz=UTC).isoformat()
        schedule = {topic.lower(): refresh_days}

        for idx, finding in enumerate(payload.findings):
            if not self._topic_match(finding, topic):
                continue
            if not should_trigger_research(finding, threshold, schedule):
                continue

            old_statement = finding.statement
            old_confidence = finding.confidence
            boosted_confidence = min(1.0, max(finding.confidence, threshold + 0.1))

            updated = finding.model_copy(
                update={
                    "statement": new_statement if new_statement else finding.statement,
                    "confidence": round(boosted_confidence, 3),
                    "updated_at": now,
                }
            )
            payload.findings[idx] = updated
            payload.refresh_history.append(
                create_refresh_record(
                    topic=updated.topic,
                    old_confidence=old_confidence,
                    new_confidence=updated.confidence,
                    old_statement=old_statement,
                    new_statement=updated.statement,
                )
            )
            refreshed_count += 1

        self._store.save(payload)
        return refreshed_count

    def rebuild_relationships(self, topic: str | None = None) -> int:
        """Regenerate relationships from findings and persist them."""
        payload = self._store.load()
        findings = [item for item in payload.findings if self._topic_match(item, topic)]
        relationships = self._map_relationships(findings)
        payload.relationships = relationships
        self._store.save(payload)
        return len(relationships)

    def query_topic(self, topic: str) -> dict[str, list[str]]:
        """Return findings and linked relationships for a topic query."""
        payload = self._store.load()
        findings = [item for item in payload.findings if self._topic_match(item, topic)]
        finding_ids = {item.id for item in findings}

        relationships = []
        for rel in payload.relationships:
            if rel.source in finding_ids or rel.target in finding_ids:
                relationships.append(f"{rel.source} {rel.relation.value} {rel.target}")

        return {
            "findings": [item.statement for item in findings],
            "relationships": relationships,
        }

    def to_mermaid(self, topic: str | None = None) -> str:
        """Render persisted relationships as Mermaid graph text."""
        payload = self._store.load()
        relationships = payload.relationships
        if topic:
            finding_ids = {
                item.id for item in payload.findings if self._topic_match(item, topic)
            }
            relationships = [
                rel
                for rel in relationships
                if rel.source in finding_ids or rel.target in finding_ids
            ]

        lines = ["graph TD"]
        for rel in relationships:
            label = rel.relation.value.replace("_", " ")
            lines.append(f"  {rel.source} -->|{label}| {rel.target}")
        return "\n".join(lines)

    def to_json_graph(self, topic: str | None = None) -> dict[str, object]:
        """Render persisted relationships as JSON graph format."""
        payload = self._store.load()
        findings = [item for item in payload.findings if self._topic_match(item, topic)]
        finding_ids = {item.id for item in findings}
        relationships = [
            rel
            for rel in payload.relationships
            if not topic or rel.source in finding_ids or rel.target in finding_ids
        ]

        nodes = [
            {
                "id": item.id,
                "topic": item.topic,
                "confidence": item.confidence,
                "updated_at": item.updated_at,
            }
            for item in findings
        ]
        edges = [
            {
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation.value,
                "evidence": rel.evidence,
            }
            for rel in relationships
        ]

        return {"nodes": nodes, "edges": edges}

    def _map_relationships(
        self,
        findings: list[KnowledgeFinding],
    ) -> list[KnowledgeRelationship]:
        relationships: list[KnowledgeRelationship] = []
        for finding in findings:
            relation = self._infer_relation(finding.statement.lower())
            if relation is None:
                continue

            entities = sorted(
                {match.group(1) for match in _ENTITY_RE.finditer(finding.statement)}
            )
            if len(entities) < 2:
                continue

            relationships.append(
                KnowledgeRelationship(
                    source=finding.id,
                    target=f"entity:{entities[1].lower()}",
                    relation=relation,
                    evidence=finding.statement,
                )
            )
        return relationships

    def _infer_relation(self, statement: str) -> RelationshipType | None:
        if "depends on" in statement:
            return RelationshipType.DEPENDS_ON
        if "contradict" in statement or "conflict" in statement:
            return RelationshipType.CONTRADICTS
        if "extends" in statement or "builds on" in statement:
            return RelationshipType.EXTENDS
        return None

    def _topic_match(self, finding: KnowledgeFinding, topic: str | None) -> bool:
        if topic is None:
            return True
        return topic.lower() in finding.topic.lower()
