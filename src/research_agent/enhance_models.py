"""Models for enhancement-mode project analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class OpportunityCategory(StrEnum):
    """Enhancement opportunity categories."""

    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEPENDENCIES = "dependencies"


class ProjectDependency(BaseModel):
    """Dependency metadata parsed from project files."""

    name: str
    version: str = ""
    source: str = ""


class FileSummary(BaseModel):
    """Summarized project file content."""

    path: str
    size_bytes: int = Field(default=0, ge=0)
    summary: str
    truncated: bool = False


class ProjectContext(BaseModel):
    """Structured project context used for enhancement analysis."""

    project_path: str
    project_name: str
    description: str = ""
    languages: list[str] = Field(default_factory=list)
    framework: str = "unknown"
    dependencies: list[ProjectDependency] = Field(default_factory=list)
    key_files: list[FileSummary] = Field(default_factory=list)
    scanned_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class EnhancementOpportunity(BaseModel):
    """An identified project improvement opportunity."""

    title: str
    category: OpportunityCategory
    impact_score: int = Field(ge=1, le=5)
    effort_score: int = Field(ge=1, le=5)
    rationale: str
    suggested_query: str


class KnowledgeEntry(BaseModel):
    """A stored enhancement finding for incremental refresh."""

    topic: str
    category: OpportunityCategory
    updated_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    finding: str
    query: str


class DeltaSummary(BaseModel):
    """Delta report details between prior and current enhancement findings."""

    new_topics: list[str] = Field(default_factory=list)
    skipped_recent_topics: list[str] = Field(default_factory=list)
    shared_topics: list[str] = Field(default_factory=list)
