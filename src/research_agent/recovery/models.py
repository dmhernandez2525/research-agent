"""Models used by recovery orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    """Retry policy for a graph node."""

    attempts: int = Field(default=3, ge=1, le=10)
    backoff_initial_seconds: float = Field(default=0.5, gt=0.0)
    backoff_max_seconds: float = Field(default=8.0, gt=0.0)


class DeadLetterEntry(BaseModel):
    """Unrecoverable node failure captured by the orchestrator."""

    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    node: str
    error_type: str
    message: str
    attempts: int = Field(default=1, ge=1)
    reason: str = Field(default="retry_exhausted")


class RecoveryMetrics(BaseModel):
    """Session-level recovery telemetry."""

    retries_attempted: int = Field(default=0, ge=0)
    recovered_failures: int = Field(default=0, ge=0)
    retry_exhausted: int = Field(default=0, ge=0)
    circuit_breaker_opened: int = Field(default=0, ge=0)
    circuit_breaker_skips: int = Field(default=0, ge=0)
    dead_letter_count: int = Field(default=0, ge=0)
