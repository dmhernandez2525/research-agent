"""Tests for API session manager state transitions and queueing."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from research_agent.api.auth import APIKeyStore
from research_agent.api.events import EventBus
from research_agent.api.manager import QueueOverflowError, SessionManager
from research_agent.api.models import SessionCreateRequest, SessionStatus
from research_agent.config import Settings

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.report.output_dir = tmp_path / "reports"
    settings.checkpoints.directory = tmp_path / "checkpoints"
    settings.api.max_concurrent_sessions = 1
    settings.api.queue_limit = 5
    settings.api.api_key_file = tmp_path / "keys.json"
    return settings


@pytest.mark.asyncio
async def test_session_manager_completes_session(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = APIKeyStore(settings.api.api_key_file)
    key = store.create_key("client")
    manager = SessionManager(settings, EventBus(tmp_path / "events"), store)

    session = await manager.start_session(
        SessionCreateRequest(query="demo"), api_key_id=key.id
    )

    await asyncio.sleep(0.2)
    updated = manager.get_session(session.id)
    assert updated is not None
    assert updated.status == SessionStatus.COMPLETED
    assert updated.report_path is not None
    assert updated.duration_seconds >= 0
    assert len(updated.sources) >= 1
    assert os.path.exists(updated.report_path)


@pytest.mark.asyncio
async def test_session_manager_queue_limit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.api.queue_limit = 0
    store = APIKeyStore(settings.api.api_key_file)
    key = store.create_key("client")
    manager = SessionManager(settings, EventBus(tmp_path / "events"), store)

    await manager.start_session(SessionCreateRequest(query="first"), api_key_id=key.id)
    with pytest.raises(QueueOverflowError):
        await manager.start_session(
            SessionCreateRequest(query="second"), api_key_id=key.id
        )


@pytest.mark.asyncio
async def test_cancel_queued_session(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = APIKeyStore(settings.api.api_key_file)
    key = store.create_key("client")
    manager = SessionManager(settings, EventBus(tmp_path / "events"), store)

    first = await manager.start_session(
        SessionCreateRequest(query="first"), api_key_id=key.id
    )
    second = await manager.start_session(
        SessionCreateRequest(query="second"), api_key_id=key.id
    )

    assert first.id != second.id
    cancelled = await manager.cancel_session(second.id)
    assert cancelled is True

    queued = manager.get_session(second.id)
    assert queued is not None
    assert queued.status == SessionStatus.CANCELLED


@pytest.mark.asyncio
async def test_shutdown_waits_for_running_tasks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = APIKeyStore(settings.api.api_key_file)
    key = store.create_key("client")
    manager = SessionManager(settings, EventBus(tmp_path / "events"), store)

    session = await manager.start_session(
        SessionCreateRequest(query="demo"), api_key_id=key.id
    )
    await manager.shutdown()

    updated = manager.get_session(session.id)
    assert updated is not None
    assert updated.status in {SessionStatus.COMPLETED, SessionStatus.CANCELLED}
