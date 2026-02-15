"""Background session manager for API-triggered research runs."""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from research_agent.api.models import (
    SessionCreateRequest,
    SessionRecord,
    SessionSource,
    SessionStatus,
)

if TYPE_CHECKING:
    from research_agent.api.auth import APIKeyStore
    from research_agent.api.events import EventBus
    from research_agent.api.notifications import NotificationDispatcher
    from research_agent.config import Settings


class QueueOverflowError(Exception):
    """Raised when session queue limit is reached."""


class SessionManager:
    """Manage queued/running/completed research API sessions."""

    def __init__(
        self,
        settings: Settings,
        event_bus: EventBus,
        api_keys: APIKeyStore,
        notifications: NotificationDispatcher | None = None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._api_keys = api_keys
        self._notifications = notifications

        self._sessions: dict[str, SessionRecord] = {}
        self._queue: deque[str] = deque()
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def start_session(
        self,
        request: SessionCreateRequest,
        *,
        api_key_id: str,
    ) -> SessionRecord:
        async with self._lock:
            session_id = f"session-{uuid.uuid4().hex[:12]}"
            session = SessionRecord(id=session_id, query=request.query)
            self._sessions[session_id] = session

            if self._running_count() < self._settings.api.max_concurrent_sessions:
                self._launch_task(session_id)
            else:
                if len(self._queue) >= self._settings.api.queue_limit:
                    self._sessions.pop(session_id, None)
                    raise QueueOverflowError("Session queue limit reached")
                self._queue.append(session_id)
                session.queued_position = len(self._queue)

            self._api_keys.update_usage(api_key_id, requests=1, sessions_started=1)
            return session

    def list_sessions(self) -> list[SessionRecord]:
        return sorted(
            self._sessions.values(),
            key=lambda session: session.created_at,
            reverse=True,
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    async def cancel_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            if session_id in self._queue:
                self._queue.remove(session_id)
                session.status = SessionStatus.CANCELLED
                session.current_step = "cancelled"
                session.updated_at = datetime.now(tz=UTC).isoformat()
                self._event_bus.publish(session_id, "error", {"message": "Cancelled"})
                return True

            task = self._running_tasks.get(session_id)
            if task is not None and not task.done():
                self._cancel_requested.add(session_id)
                return True

            return False

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            if session_id in self._queue:
                self._queue.remove(session_id)
            task = self._running_tasks.pop(session_id, None)
            if task is not None and not task.done():
                task.cancel()
            if session.report_path:
                Path(session.report_path).unlink(missing_ok=True)
            return True

    async def cleanup_stale(self, stale_hours: int = 24) -> int:
        cutoff = datetime.now(tz=UTC) - timedelta(hours=stale_hours)
        removed = 0
        for session in list(self._sessions.values()):
            if session.status in {SessionStatus.RUNNING, SessionStatus.QUEUED}:
                continue
            created = datetime.fromisoformat(session.created_at)
            if created < cutoff and await self.delete_session(session.id):
                removed += 1
        return removed

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = [task for task in self._running_tasks.values() if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _running_count(self) -> int:
        return sum(1 for task in self._running_tasks.values() if not task.done())

    def _launch_task(self, session_id: str) -> None:
        task = asyncio.create_task(self._run_session(session_id))
        self._running_tasks[session_id] = task

    def _mock_sources(self, query: str) -> list[SessionSource]:
        return [
            SessionSource(
                id="src-1",
                domain="docs.example.com",
                title=f"{query} practical guide",
                freshness=0.81,
                quality_score=0.79,
                subtopic="Implementation approaches",
                query=query,
                content_preview=(
                    "Example implementation notes captured from documentation-style "
                    "sources."
                ),
            ),
            SessionSource(
                id="src-2",
                domain="research.example.org",
                title=f"{query} benchmarking analysis",
                freshness=0.67,
                quality_score=0.71,
                subtopic="Performance and tradeoffs",
                query=query,
                content_preview=(
                    "Synthetic benchmark summary used for API/UI integration tests."
                ),
            ),
        ]

    async def _run_session(self, session_id: str) -> None:
        session = self._sessions[session_id]
        started_at = datetime.now(tz=UTC)
        session.status = SessionStatus.RUNNING
        session.current_step = "plan"
        session.queued_position = None
        session.updated_at = datetime.now(tz=UTC).isoformat()
        self._event_bus.publish(session_id, "plan_ready", {"step": "plan"})

        steps: list[tuple[str, str]] = [
            ("plan", "plan_ready"),
            ("search", "search_complete"),
            ("scrape", "scrape_progress"),
            ("summarize", "summary_ready"),
            ("synthesize", "synthesis_complete"),
        ]

        try:
            for index, (step_name, event_type) in enumerate(steps, start=1):
                if session_id in self._cancel_requested:
                    raise asyncio.CancelledError

                session.current_step = step_name
                session.progress = round(index / len(steps) * 100, 1)
                session.tokens_used += 120 * index
                session.cost_usd += 0.003 * index
                session.updated_at = datetime.now(tz=UTC).isoformat()

                await asyncio.sleep(0.01)

                self._event_bus.publish(
                    session_id,
                    event_type,
                    {
                        "step": step_name,
                        "progress": session.progress,
                    },
                )
                self._event_bus.publish(
                    session_id,
                    "cost_update",
                    {
                        "tokens": session.tokens_used,
                        "cost_usd": round(session.cost_usd, 4),
                    },
                )

            report_dir = Path(self._settings.report.output_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{session_id}.md"
            report_path.write_text(
                f"# API Research Report\n\nSession: {session_id}\n\nQuery: {session.query}\n",
                encoding="utf-8",
            )
            session.report_path = str(report_path)
            session.sources = self._mock_sources(session.query)
            session.status = SessionStatus.COMPLETED
            session.current_step = "done"

            if self._notifications is not None:
                await self._notifications.notify(
                    "completion",
                    session,
                    "Research session completed.",
                )
        except asyncio.CancelledError:
            session.status = SessionStatus.CANCELLED
            session.current_step = "cancelled"
            self._event_bus.publish(
                session_id, "error", {"message": "Session cancelled"}
            )
            if self._notifications is not None:
                await self._notifications.notify("error", session, "Session cancelled")
        except Exception as exc:
            session.status = SessionStatus.FAILED
            session.error = str(exc)
            session.current_step = "failed"
            self._event_bus.publish(session_id, "error", {"message": str(exc)})
            if self._notifications is not None:
                await self._notifications.notify("error", session, str(exc))
        finally:
            session.duration_seconds = max(
                0.0,
                (datetime.now(tz=UTC) - started_at).total_seconds(),
            )
            session.updated_at = datetime.now(tz=UTC).isoformat()
            self._running_tasks.pop(session_id, None)
            self._cancel_requested.discard(session_id)
            await self._start_next_queued()

    async def _start_next_queued(self) -> None:
        async with self._lock:
            if self._shutting_down:
                return
            while (
                self._queue
                and self._running_count() < self._settings.api.max_concurrent_sessions
            ):
                next_session_id = self._queue.popleft()
                self._launch_task(next_session_id)
                for index, queued_id in enumerate(self._queue, start=1):
                    self._sessions[queued_id].queued_position = index
