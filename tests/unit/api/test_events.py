"""Tests for API event bus and buffering."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from research_agent.api.events import EventBus

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_event_bus_publish_subscribe_and_buffer(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "events")

    queue = bus.subscribe("s1")
    event = bus.publish("s1", "plan_ready", {"step": "plan"})

    received = await asyncio.wait_for(queue.get(), timeout=1)
    assert received.id == event.id
    assert received.event_type == "plan_ready"

    buffered = bus.recent_events("s1")
    assert buffered

    bus.unsubscribe("s1", queue)


def test_event_bus_last_event_id_filter(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "events")

    first = bus.publish("s2", "a", {})
    second = bus.publish("s2", "b", {})

    recent = bus.recent_events("s2", last_event_id=first.id)
    assert [ev.id for ev in recent] == [second.id]


def test_event_bus_persists_jsonl(tmp_path: Path) -> None:
    bus = EventBus(tmp_path / "events")
    bus.publish("s3", "summary_ready", {"ok": True})

    path = tmp_path / "events" / "s3.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
