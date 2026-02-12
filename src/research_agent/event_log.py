"""Append-only JSONL event log for research run auditing and provenance.

Records node_enter, node_exit, error, result, and llm_call events
with timestamps, step IDs, and parent IDs for provenance chain tracking.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(StrEnum):
    """Types of events recorded in the JSONL log."""

    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    ERROR = "error"
    RESULT = "result"
    LLM_CALL = "llm_call"


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """A single event in the research run log."""

    ts: str = Field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(),
        description="ISO-8601 timestamp.",
    )
    step_id: str = Field(description="Unique step identifier.")
    parent_id: str = Field(default="", description="Parent step for provenance.")
    event: EventType = Field(description="Event type.")
    node: str = Field(default="", description="Graph node name.")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary event-specific data.",
    )


# ---------------------------------------------------------------------------
# Step ID generation
# ---------------------------------------------------------------------------


def generate_step_id(node_name: str = "") -> str:
    """Generate a unique step ID with an optional node prefix.

    Args:
        node_name: Graph node name to use as prefix.

    Returns:
        A unique, filesystem-safe step identifier.
    """
    short = uuid.uuid4().hex[:8]
    if node_name:
        return f"{node_name}-{short}"
    return f"step-{short}"


# ---------------------------------------------------------------------------
# Event Log
# ---------------------------------------------------------------------------


class EventLog:
    """Append-only JSONL event logger for a research run.

    Each event is serialized as a single JSON line and flushed
    immediately to ensure durability on crash.

    Attributes:
        path: Path to the events.jsonl file.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the event log.

        Args:
            path: Path to the JSONL log file (created if needed).
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        """Append a single event to the log.

        Args:
            event: The event to record.
        """
        line = event.model_dump_json() + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

        logger.debug(
            "event_logged",
            event_type=event.event.value,
            step_id=event.step_id,
            node=event.node,
        )

    def log_node_enter(
        self,
        node: str,
        step_id: str,
        parent_id: str = "",
        **extra: Any,
    ) -> None:
        """Log a node_enter event.

        Args:
            node: Graph node name.
            step_id: Unique step identifier.
            parent_id: Parent step ID for provenance.
            **extra: Additional payload fields.
        """
        self.append(
            Event(
                step_id=step_id,
                parent_id=parent_id,
                event=EventType.NODE_ENTER,
                node=node,
                payload=extra,
            )
        )

    def log_node_exit(
        self,
        node: str,
        step_id: str,
        parent_id: str = "",
        **extra: Any,
    ) -> None:
        """Log a node_exit event.

        Args:
            node: Graph node name.
            step_id: Unique step identifier.
            parent_id: Parent step ID for provenance.
            **extra: Additional payload fields.
        """
        self.append(
            Event(
                step_id=step_id,
                parent_id=parent_id,
                event=EventType.NODE_EXIT,
                node=node,
                payload=extra,
            )
        )

    def log_error(
        self,
        node: str,
        step_id: str,
        message: str,
        parent_id: str = "",
        **extra: Any,
    ) -> None:
        """Log an error event.

        Args:
            node: Graph node where the error occurred.
            step_id: Unique step identifier.
            message: Error description.
            parent_id: Parent step ID for provenance.
            **extra: Additional payload fields.
        """
        self.append(
            Event(
                step_id=step_id,
                parent_id=parent_id,
                event=EventType.ERROR,
                node=node,
                payload={"message": message, **extra},
            )
        )

    def log_result(
        self,
        node: str,
        step_id: str,
        parent_id: str = "",
        **extra: Any,
    ) -> None:
        """Log a result event.

        Args:
            node: Graph node that produced the result.
            step_id: Unique step identifier.
            parent_id: Parent step ID for provenance.
            **extra: Additional payload fields.
        """
        self.append(
            Event(
                step_id=step_id,
                parent_id=parent_id,
                event=EventType.RESULT,
                node=node,
                payload=extra,
            )
        )

    def log_llm_call(
        self,
        node: str,
        step_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        parent_id: str = "",
    ) -> None:
        """Log an LLM API call with cost and latency.

        Args:
            node: Graph node that made the call.
            step_id: Unique step identifier.
            model: Model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_usd: Cost in USD.
            latency_ms: Latency in milliseconds.
            parent_id: Parent step ID for provenance.
        """
        self.append(
            Event(
                step_id=step_id,
                parent_id=parent_id,
                event=EventType.LLM_CALL,
                node=node,
                payload={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                },
            )
        )

    def read_events(self) -> list[Event]:
        """Read all events from the log file.

        Returns:
            List of Event objects in chronological order.
        """
        if not self.path.exists():
            return []

        events: list[Event] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                events.append(Event.model_validate_json(stripped))
        return events

    def read_events_for_step(self, step_id: str) -> list[Event]:
        """Read events filtered to a specific step.

        Args:
            step_id: The step ID to filter by.

        Returns:
            Events matching the given step_id.
        """
        return [e for e in self.read_events() if e.step_id == step_id]

    def provenance_chain(self, step_id: str) -> list[Event]:
        """Build the provenance chain for a step by following parent_id links.

        Walks backward from the given step to the root, collecting the
        enter event for each ancestor step.

        Args:
            step_id: The step ID to trace.

        Returns:
            List of events from root to the given step (inclusive).
        """
        all_events = self.read_events()

        # Index: step_id -> list of events for that step
        by_step: dict[str, list[Event]] = {}
        for ev in all_events:
            by_step.setdefault(ev.step_id, []).append(ev)

        chain: list[Event] = []
        current_id = step_id

        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            step_events = by_step.get(current_id, [])
            if not step_events:
                break
            # Use the first event (typically node_enter) as the representative
            chain.append(step_events[0])
            current_id = step_events[0].parent_id

        chain.reverse()
        return chain
