"""Knowledge graph construction and traversal."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from research_agent.embeddings import EmbeddingDocument, ResearchEmbeddings
from research_agent.knowledge.models import (
    KnowledgeExportPayload,
    KnowledgeFinding,
    KnowledgeRelationship,
    RelationshipType,
)

if TYPE_CHECKING:
    from pathlib import Path

_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")


class KnowledgeGraphEngine:
    """Build and query graph relationships from knowledge findings."""

    def __init__(self, persist_directory: Path) -> None:
        self._persist_directory = persist_directory
        self._embeddings = ResearchEmbeddings(
            collection_name="knowledge_graph",
            persist_directory=str(persist_directory),
        )

    def extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from free text."""
        entities = sorted({match.group(1) for match in _ENTITY_RE.finditer(text)})
        return entities[:30]

    def map_relationships(
        self, findings: list[KnowledgeFinding]
    ) -> list[KnowledgeRelationship]:
        """Infer relationships (depends_on/contradicts/extends) from statements."""
        relationships: list[KnowledgeRelationship] = []
        for finding in findings:
            statement = finding.statement.lower()
            relation = self._infer_relation(statement)
            if relation is None:
                continue

            entities = self.extract_entities(finding.statement)
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

    def store_relationships(self, relationships: list[KnowledgeRelationship]) -> int:
        """Store relationship metadata in ChromaDB embeddings collection."""
        docs: list[EmbeddingDocument] = []
        for index, rel in enumerate(relationships):
            docs.append(
                EmbeddingDocument(
                    id=f"rel-{index}-{rel.source}",
                    content=f"{rel.source} {rel.relation.value} {rel.target}",
                    metadata={
                        "source": rel.source,
                        "target": rel.target,
                        "relation": rel.relation.value,
                        "evidence": rel.evidence,
                    },
                )
            )
        return self._embeddings.add_documents(docs)

    def query(
        self, payload: KnowledgeExportPayload, topic: str
    ) -> dict[str, list[str]]:
        """Traverse graph context for a topic question."""
        topic_lower = topic.lower()
        matching = [
            finding
            for finding in payload.findings
            if topic_lower in finding.topic.lower()
        ]
        ids = {finding.id for finding in matching}

        related: list[str] = []
        for rel in payload.relationships:
            if rel.source in ids or rel.target in ids:
                related.append(f"{rel.source} {rel.relation.value} {rel.target}")

        return {
            "findings": [finding.statement for finding in matching],
            "relationships": related,
        }

    def to_mermaid(self, relationships: list[KnowledgeRelationship]) -> str:
        """Export relationships as Mermaid graph syntax."""
        lines = ["graph TD"]
        for rel in relationships:
            relation = rel.relation.value.replace("_", " ")
            lines.append(f"  {rel.source} -->|{relation}| {rel.target}")
        return "\n".join(lines)

    def _infer_relation(self, statement: str) -> RelationshipType | None:
        if "depends on" in statement:
            return RelationshipType.DEPENDS_ON
        if "contradict" in statement or "conflict" in statement:
            return RelationshipType.CONTRADICTS
        if "extends" in statement or "builds on" in statement:
            return RelationshipType.EXTENDS
        return None
