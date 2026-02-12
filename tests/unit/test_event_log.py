"""Unit tests for research_agent.event_log - JSONL event logging."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from research_agent.event_log import (
    Event,
    EventLog,
    EventType,
    generate_step_id,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestEventType:
    """EventType enum values."""

    def test_node_enter(self) -> None:
        assert EventType.NODE_ENTER == "node_enter"

    def test_node_exit(self) -> None:
        assert EventType.NODE_EXIT == "node_exit"

    def test_error(self) -> None:
        assert EventType.ERROR == "error"

    def test_result(self) -> None:
        assert EventType.RESULT == "result"

    def test_llm_call(self) -> None:
        assert EventType.LLM_CALL == "llm_call"


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class TestEventModel:
    """Event pydantic model validation."""

    def test_minimal_construction(self) -> None:
        ev = Event(step_id="s-001", event=EventType.NODE_ENTER)
        assert ev.step_id == "s-001"
        assert ev.event == EventType.NODE_ENTER
        assert ev.parent_id == ""
        assert ev.node == ""
        assert ev.payload == {}

    def test_timestamp_auto_populated(self) -> None:
        ev = Event(step_id="s-001", event=EventType.NODE_ENTER)
        assert ev.ts != ""
        assert "T" in ev.ts

    def test_full_construction(self) -> None:
        ev = Event(
            step_id="search-abc",
            parent_id="plan-000",
            event=EventType.NODE_EXIT,
            node="search",
            payload={"results_count": 8},
        )
        assert ev.node == "search"
        assert ev.parent_id == "plan-000"
        assert ev.payload["results_count"] == 8

    def test_serialization_round_trip(self) -> None:
        ev = Event(
            step_id="s-001",
            event=EventType.LLM_CALL,
            node="summarize",
            payload={"model": "claude", "cost": 0.01},
        )
        dumped = ev.model_dump_json()
        restored = Event.model_validate_json(dumped)
        assert restored.step_id == ev.step_id
        assert restored.payload["cost"] == 0.01


# ---------------------------------------------------------------------------
# generate_step_id
# ---------------------------------------------------------------------------


class TestGenerateStepId:
    """Step ID generation."""

    def test_returns_string(self) -> None:
        assert isinstance(generate_step_id(), str)

    def test_no_prefix(self) -> None:
        sid = generate_step_id()
        assert sid.startswith("step-")

    def test_with_node_prefix(self) -> None:
        sid = generate_step_id("search")
        assert sid.startswith("search-")

    def test_unique_ids(self) -> None:
        ids = {generate_step_id("test") for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# EventLog - basic append
# ---------------------------------------------------------------------------


class TestEventLogAppend:
    """EventLog.append writes JSONL lines."""

    def test_creates_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "events.jsonl"
        log = EventLog(log_path)
        log.append(Event(step_id="s-001", event=EventType.NODE_ENTER, node="plan"))
        assert log_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "sub" / "dir" / "events.jsonl"
        log = EventLog(log_path)
        log.append(Event(step_id="s-001", event=EventType.NODE_ENTER))
        assert log_path.exists()

    def test_appends_valid_json_line(self, tmp_path: Path) -> None:
        log_path = tmp_path / "events.jsonl"
        log = EventLog(log_path)
        log.append(Event(step_id="s-001", event=EventType.NODE_ENTER, node="plan"))
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["step_id"] == "s-001"
        assert parsed["event"] == "node_enter"

    def test_multiple_appends(self, tmp_path: Path) -> None:
        log_path = tmp_path / "events.jsonl"
        log = EventLog(log_path)
        for i in range(5):
            log.append(Event(step_id=f"s-{i:03d}", event=EventType.NODE_ENTER))
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# EventLog - convenience methods
# ---------------------------------------------------------------------------


class TestEventLogNodeEnter:
    """log_node_enter convenience method."""

    def test_writes_node_enter_event(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("search", "search-001", parent_id="plan-000")
        events = log.read_events()
        assert len(events) == 1
        assert events[0].event == EventType.NODE_ENTER
        assert events[0].node == "search"
        assert events[0].parent_id == "plan-000"

    def test_extra_payload(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("search", "s-001", subtopic_id="st-1")
        events = log.read_events()
        assert events[0].payload["subtopic_id"] == "st-1"


class TestEventLogNodeExit:
    """log_node_exit convenience method."""

    def test_writes_node_exit_event(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_exit("search", "search-001", results_count=8)
        events = log.read_events()
        assert events[0].event == EventType.NODE_EXIT
        assert events[0].payload["results_count"] == 8


class TestEventLogError:
    """log_error convenience method."""

    def test_writes_error_event(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_error("scrape", "scrape-001", "Connection timeout", recoverable=True)
        events = log.read_events()
        assert events[0].event == EventType.ERROR
        assert events[0].payload["message"] == "Connection timeout"
        assert events[0].payload["recoverable"] is True


class TestEventLogResult:
    """log_result convenience method."""

    def test_writes_result_event(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_result("summarize", "sum-001", summary_length=500)
        events = log.read_events()
        assert events[0].event == EventType.RESULT
        assert events[0].payload["summary_length"] == 500


class TestEventLogLlmCall:
    """log_llm_call records cost and latency."""

    def test_writes_llm_call_event(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_llm_call(
            node="summarize",
            step_id="sum-001",
            model="claude-sonnet-4-5-20250929",
            input_tokens=1500,
            output_tokens=300,
            cost_usd=0.009,
            latency_ms=1200.5,
        )
        events = log.read_events()
        assert events[0].event == EventType.LLM_CALL
        assert events[0].payload["model"] == "claude-sonnet-4-5-20250929"
        assert events[0].payload["input_tokens"] == 1500
        assert events[0].payload["output_tokens"] == 300
        assert events[0].payload["cost_usd"] == 0.009
        assert events[0].payload["latency_ms"] == 1200.5

    def test_llm_call_with_parent(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_llm_call(
            node="summarize",
            step_id="sum-001",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=500.0,
            parent_id="search-001",
        )
        events = log.read_events()
        assert events[0].parent_id == "search-001"


# ---------------------------------------------------------------------------
# EventLog - read operations
# ---------------------------------------------------------------------------


class TestEventLogRead:
    """Reading events from the log."""

    def test_read_empty_file(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        assert log.read_events() == []

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        # Don't write anything; file doesn't exist yet
        events = log.read_events()
        assert events == []

    def test_read_events_round_trip(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("plan", "plan-001")
        log.log_node_exit("plan", "plan-001")
        events = log.read_events()
        assert len(events) == 2
        assert events[0].event == EventType.NODE_ENTER
        assert events[1].event == EventType.NODE_EXIT

    def test_read_events_for_step(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("search", "s-001")
        log.log_node_enter("search", "s-002")
        log.log_node_exit("search", "s-001")
        filtered = log.read_events_for_step("s-001")
        assert len(filtered) == 2
        assert all(e.step_id == "s-001" for e in filtered)


# ---------------------------------------------------------------------------
# EventLog - provenance chain
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """provenance_chain traces step ancestry."""

    def test_single_step(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("plan", "plan-001")
        chain = log.provenance_chain("plan-001")
        assert len(chain) == 1
        assert chain[0].step_id == "plan-001"

    def test_two_level_chain(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("plan", "plan-001")
        log.log_node_enter("search", "search-001", parent_id="plan-001")
        chain = log.provenance_chain("search-001")
        assert len(chain) == 2
        assert chain[0].step_id == "plan-001"
        assert chain[1].step_id == "search-001"

    def test_three_level_chain(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("plan", "plan-001")
        log.log_node_enter("search", "search-001", parent_id="plan-001")
        log.log_node_enter("scrape", "scrape-001", parent_id="search-001")
        chain = log.provenance_chain("scrape-001")
        assert len(chain) == 3
        assert chain[0].step_id == "plan-001"
        assert chain[1].step_id == "search-001"
        assert chain[2].step_id == "scrape-001"

    def test_missing_step_returns_empty(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        log.log_node_enter("plan", "plan-001")
        chain = log.provenance_chain("nonexistent")
        assert chain == []

    def test_broken_chain_stops_gracefully(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        # search-001 references a parent that doesn't exist
        log.log_node_enter("search", "search-001", parent_id="missing-parent")
        chain = log.provenance_chain("search-001")
        assert len(chain) == 1
        assert chain[0].step_id == "search-001"

    def test_no_infinite_loop_on_cycle(self, tmp_path: Path) -> None:
        log = EventLog(tmp_path / "events.jsonl")
        # Create a cycle: A -> B -> A
        log.append(
            Event(
                step_id="a",
                parent_id="b",
                event=EventType.NODE_ENTER,
                node="test",
            )
        )
        log.append(
            Event(
                step_id="b",
                parent_id="a",
                event=EventType.NODE_ENTER,
                node="test",
            )
        )
        chain = log.provenance_chain("a")
        # Should terminate without infinite loop
        assert len(chain) <= 2
