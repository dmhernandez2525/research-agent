"""API request/response models and session state models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    """Lifecycle states for API research sessions."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SessionCreateRequest(BaseModel):
    """Request payload for creating a research session."""

    query: str = Field(min_length=1)
    budget: float | None = Field(default=None, gt=0)
    output_format: str = Field(default="md", pattern="^(md|pdf)$")


class SessionSource(BaseModel):
    """Normalized source metadata attached to a session."""

    id: str
    domain: str
    title: str
    freshness: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    subtopic: str
    query: str
    content_preview: str


class SessionRecord(BaseModel):
    """State persisted for each API session."""

    id: str
    query: str
    status: SessionStatus = SessionStatus.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: str = "queued"
    cost_usd: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    report_path: str | None = None
    duration_seconds: float = Field(default=0.0, ge=0.0)
    sources: list[SessionSource] = Field(default_factory=list)
    error: str | None = None
    queued_position: int | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class SessionListResponse(BaseModel):
    """Session listing response."""

    sessions: list[SessionRecord]


class APIKeyRecord(BaseModel):
    """Persisted API key metadata."""

    id: str
    key: str
    name: str
    admin: bool = False
    revoked: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    requests: int = 0
    sessions_started: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0


class APIKeyCreateResponse(BaseModel):
    """CLI/API response for new API keys."""

    id: str
    key: str
    admin: bool


class SessionEvent(BaseModel):
    """Event stream payload for websocket/SSE subscribers."""

    id: int
    session_id: str
    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    payload: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
