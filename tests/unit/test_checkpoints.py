"""Unit tests for research_agent.checkpoints - atomic writes, recovery, rotation."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from research_agent.checkpoints import (
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointManager,
    CheckpointMetadata,
    generate_run_id,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---- generate_run_id --------------------------------------------------------


class TestGenerateRunId:
    """Run ID generation for checkpoint scoping."""

    def test_returns_string(self) -> None:
        run_id = generate_run_id()
        assert isinstance(run_id, str)

    def test_starts_with_run_prefix(self) -> None:
        run_id = generate_run_id()
        assert run_id.startswith("run-")

    def test_unique_ids(self) -> None:
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100


# ---- CheckpointMetadata -----------------------------------------------------


class TestCheckpointMetadata:
    """Metadata model for checkpoints."""

    def test_default_construction(self) -> None:
        meta = CheckpointMetadata(checkpoint_id="cp-001")
        assert meta.checkpoint_id == "cp-001"
        assert meta.step_index == 0
        assert meta.step_name == ""
        assert meta.sha256 == ""
        assert meta.state_size_bytes == 0

    def test_created_at_auto_populated(self) -> None:
        meta = CheckpointMetadata(checkpoint_id="cp-001")
        assert meta.created_at != ""
        assert "T" in meta.created_at


# ---- Atomic writes -----------------------------------------------------------


class TestAtomicWrites:
    """Checkpoint writes must be atomic to survive mid-write crashes."""

    def test_save_creates_checkpoint_file(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"query": "test"})
        assert (tmp_path / "cp-001.json").exists()

    def test_save_creates_metadata_file(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"query": "test"})
        assert (tmp_path / "cp-001.meta.json").exists()

    def test_no_temp_files_remain(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"query": "test"})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_checkpoint_contains_valid_json(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"query": "test", "count": 42})
        data = json.loads((tmp_path / "cp-001.json").read_text())
        assert data["query"] == "test"
        assert data["count"] == 42

    def test_metadata_contains_checksum(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"query": "test"})
        meta = json.loads((tmp_path / "cp-001.meta.json").read_text())
        assert "sha256" in meta
        assert len(meta["sha256"]) == 64

    def test_concurrent_saves_no_corruption(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-001", {"v": 1})
        mgr.save("cp-001", {"v": 2})
        data = json.loads((tmp_path / "cp-001.json").read_text())
        assert data["v"] == 2

    def test_save_returns_metadata(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        meta = mgr.save("cp-001", {"query": "test"}, step_index=2, step_name="search")
        assert meta.checkpoint_id == "cp-001"
        assert meta.step_index == 2
        assert meta.step_name == "search"
        assert meta.state_size_bytes > 0


# ---- Load and verify --------------------------------------------------------


class TestLoadCheckpoint:
    """Loading checkpoints with integrity verification."""

    def test_round_trip(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        original: dict[str, Any] = {"query": "test", "index": 3, "cost": 0.05}
        mgr.save("cp-001", original)
        loaded = mgr.load("cp-001")
        assert loaded["query"] == "test"
        assert loaded["index"] == 3
        assert loaded["cost"] == 0.05

    def test_nested_state_preserved(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        state: dict[str, Any] = {
            "subtopics": [{"id": 1, "q": "A"}, {"id": 2, "q": "B"}],
            "errors": [],
        }
        mgr.save("cp-nested", state)
        loaded = mgr.load("cp-nested")
        assert len(loaded["subtopics"]) == 2
        assert loaded["subtopics"][0]["id"] == 1

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        with pytest.raises(CheckpointError, match="not found"):
            mgr.load("nonexistent")

    def test_corrupted_checkpoint_raises(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-corrupt", {"query": "test"})
        cp_path = tmp_path / "cp-corrupt.json"
        cp_path.write_text("CORRUPTED DATA")
        with pytest.raises(CheckpointCorruptionError):
            mgr.load("cp-corrupt")

    def test_load_without_metadata_succeeds(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-nometa", {"query": "test"})
        (tmp_path / "cp-nometa.meta.json").unlink()
        loaded = mgr.load("cp-nometa")
        assert loaded["query"] == "test"


# ---- Latest checkpoint -------------------------------------------------------


class TestLatest:
    """Finding the most recent checkpoint."""

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        assert mgr.latest() is None

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        mgr.save("cp-old", {"step": 1})
        time.sleep(0.05)
        mgr.save("cp-new", {"step": 2})
        assert mgr.latest() == "cp-new"


# ---- List checkpoints -------------------------------------------------------


class TestListCheckpoints:
    """Listing all available checkpoints."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path)
        assert mgr.list_checkpoints() == []

    def test_lists_all(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=10)
        for i in range(3):
            mgr.save(f"cp-{i:03d}", {"i": i})
            time.sleep(0.02)
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 3

    def test_newest_first(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=10)
        mgr.save("cp-first", {"i": 0})
        time.sleep(0.05)
        mgr.save("cp-second", {"i": 1})
        checkpoints = mgr.list_checkpoints()
        assert checkpoints[0].checkpoint_id == "cp-second"


# ---- Rotation ----------------------------------------------------------------


class TestRotation:
    """Old checkpoints should be rotated to cap disk usage."""

    def test_rotation_keeps_max(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=3)
        for i in range(6):
            mgr.save(f"cp-{i:03d}", {"i": i})
            time.sleep(0.02)
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 3

    def test_rotation_preserves_newest(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=2)
        mgr.save("cp-old", {"v": 1})
        time.sleep(0.05)
        mgr.save("cp-mid", {"v": 2})
        time.sleep(0.05)
        mgr.save("cp-new", {"v": 3})
        remaining_ids = [c.checkpoint_id for c in mgr.list_checkpoints()]
        assert "cp-new" in remaining_ids
        assert "cp-mid" in remaining_ids

    def test_minimum_two_retained(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(directory=tmp_path, max_checkpoints=1)
        for i in range(5):
            mgr.save(f"cp-{i:03d}", {"i": i})
            time.sleep(0.02)
        remaining = mgr.list_checkpoints()
        assert len(remaining) >= 2
