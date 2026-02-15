"""In-process event bus for websocket and SSE progress streaming."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from research_agent.api.models import SessionEvent

if TYPE_CHECKING:
    from pathlib import Path


class EventBus:
    """Publish/subscribe event bus with buffering and JSONL persistence."""

    def __init__(self, data_dir: Path, buffer_size: int = 200) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._next_id = 1
        self._buffers: dict[str, deque[SessionEvent]] = defaultdict(
            lambda: deque(maxlen=self._buffer_size)
        )
        self._subscribers: dict[str, set[asyncio.Queue[SessionEvent]]] = defaultdict(
            set
        )

    def publish(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, str | int | float | bool | None] | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            id=self._next_id,
            session_id=session_id,
            event_type=event_type,
            payload=payload or {},
        )
        self._next_id += 1

        self._buffers[session_id].append(event)
        self._persist(event)

        for queue in list(self._subscribers.get(session_id, set())):
            queue.put_nowait(event)

        return event

    def subscribe(
        self,
        session_id: str,
        last_event_id: int | None = None,
    ) -> asyncio.Queue[SessionEvent]:
        queue: asyncio.Queue[SessionEvent] = asyncio.Queue()

        for event in self.recent_events(session_id, last_event_id=last_event_id):
            queue.put_nowait(event)

        self._subscribers[session_id].add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[SessionEvent]) -> None:
        self._subscribers[session_id].discard(queue)

    def recent_events(
        self,
        session_id: str,
        last_event_id: int | None = None,
    ) -> list[SessionEvent]:
        events = list(self._buffers.get(session_id, []))
        if last_event_id is None:
            return events
        return [event for event in events if event.id > last_event_id]

    def _persist(self, event: SessionEvent) -> None:
        path = self._data_dir / f"{event.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
