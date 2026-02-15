"""Models for advanced knowledge management workflows."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RelationshipType(StrEnum):
    """Supported knowledge graph relationship types."""

    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"


class KnowledgeFinding(BaseModel):
    """Atomic knowledge statement with confidence metadata."""

    id: str
    topic: str
    statement: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cluster: str = "general"
    updated_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class KnowledgeRelationship(BaseModel):
    """Directed relationship between two entities/findings."""

    source: str
    target: str
    relation: RelationshipType
    evidence: str = ""


class KnowledgeRefreshRecord(BaseModel):
    """Refresh run record for change history tracking."""

    topic: str
    triggered_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    old_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    new_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    change_summary: str = ""


class KnowledgeExportPayload(BaseModel):
    """Serializable knowledge bundle for sharing/import."""

    findings: list[KnowledgeFinding] = Field(default_factory=list)
    relationships: list[KnowledgeRelationship] = Field(default_factory=list)
    refresh_history: list[KnowledgeRefreshRecord] = Field(default_factory=list)
