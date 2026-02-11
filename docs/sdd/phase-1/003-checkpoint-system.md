# SDD-003: Checkpoint System

## Overview

The checkpoint system provides crash resilience through three complementary persistence layers. If the agent process terminates at any point, work completed up to the last checkpoint is preserved and can be resumed.

## Three Persistence Layers

### Layer 1: JSONL Event Log

An append-only log of every significant event during a research run.

**Format:**
```json
{"ts": "2026-02-11T10:30:00Z", "step_id": "search-001", "parent_id": "plan-000", "event": "node_enter", "node": "search", "subtopic_id": "st-1"}
{"ts": "2026-02-11T10:30:05Z", "step_id": "search-001", "parent_id": "plan-000", "event": "node_exit", "node": "search", "results_count": 8, "cost": 0.002}
{"ts": "2026-02-11T10:30:06Z", "step_id": "scrape-001", "parent_id": "search-001", "event": "node_enter", "node": "scrape", "urls_count": 8}
```

**Purpose:**
- Debugging and auditing.
- Provenance tracking (which search produced which scrape).
- Cost accounting (each LLM call logs token counts and cost).

**Implementation:**
```python
import json
from pathlib import Path

class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        line = json.dumps(event, default=str) + "\n"
        with self.path.open("a") as f:
            f.write(line)
            f.flush()
```

**Location:** `data/checkpoints/{run_id}/events.jsonl`

### Layer 2: Atomic Checkpoint Files

Full state snapshots written after each node completes.

**Atomic Write Pattern:**

```python
import hashlib
import json
import os
import tempfile
from pathlib import Path

def atomic_write_checkpoint(state: dict, path: Path) -> str:
    """Write checkpoint atomically. Returns SHA-256 hash."""
    data = json.dumps(state, default=str, indent=2).encode()
    sha256 = hashlib.sha256(data).hexdigest()

    # 1. Write to temp file in same directory (same filesystem)
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        os.write(fd, data)
        # 2. fsync to ensure data is on disk
        os.fsync(fd)
        os.close(fd)
        # 3. Atomic rename (same filesystem guarantees atomicity)
        os.replace(tmp_path, str(path))
    except Exception:
        os.close(fd)
        os.unlink(tmp_path)
        raise

    # 4. Write hash sidecar
    hash_path = path.with_suffix(".sha256")
    hash_path.write_text(sha256)

    return sha256
```

**Key properties:**
- `tempfile.mkstemp` in the same directory ensures same-filesystem rename.
- `os.fsync` forces data to disk before the rename.
- `os.replace` is atomic on POSIX systems -- the checkpoint file either has the old content or the new content, never a partial write.
- The `.sha256` sidecar enables integrity verification on load.

**File naming:** `checkpoint_{step_number:04d}.json` (e.g., `checkpoint_0003.json`)

**Location:** `data/checkpoints/{run_id}/checkpoint_NNNN.json`

### Layer 3: Progressive Markdown

A human-readable file updated as each subtopic completes.

```python
def append_progress(path: Path, subtopic: Subtopic, summary: SubtopicSummary) -> None:
    with path.open("a") as f:
        f.write(f"\n## {subtopic['title']}\n\n")
        f.write(summary["summary"])
        f.write("\n\n**Sources:**\n")
        for citation in summary["citations"]:
            f.write(f"- {citation}\n")
        f.write("\n---\n")
```

**Purpose:** Even if synthesis never runs, the user has a readable partial report.

**Location:** `data/checkpoints/{run_id}/progress.md`

## SHA-256 Integrity Verification

On resume, each checkpoint is verified before use:

```python
def load_checkpoint(path: Path) -> dict:
    data = path.read_bytes()
    actual_hash = hashlib.sha256(data).hexdigest()

    hash_path = path.with_suffix(".sha256")
    if hash_path.exists():
        expected_hash = hash_path.read_text().strip()
        if actual_hash != expected_hash:
            raise CheckpointCorruptionError(
                f"Checkpoint {path.name}: expected {expected_hash}, got {actual_hash}"
            )

    return json.loads(data)
```

## Schema Evolution with Lazy Migration

As `ResearchState` changes across versions, checkpoints from older versions must still load. The system uses lazy migration rather than relying on LangGraph's built-in checkpoint schema management, which has known limitations around schema evolution (see LangGraph GitHub Issue #536 -- schema evolution remains unsolved in the framework's native `SqliteSaver`/`PostgresSaver`). A custom checkpoint layer gives us full control over migration logic and avoids coupling to LangGraph's internal serialization format.

```python
CURRENT_SCHEMA_VERSION = 1

def migrate_state(state: dict) -> dict:
    version = state.get("_schema_version", 0)

    if version < 1:
        # v0 -> v1: Added 'seen_urls' field
        state.setdefault("seen_urls", set())
        state["_schema_version"] = 1

    # Future migrations added here as elif blocks

    return state

def load_and_migrate(path: Path) -> dict:
    state = load_checkpoint(path)
    return migrate_state(state)
```

**Rules:**
- Migrations are additive only (add fields with defaults, never remove).
- Each migration is a simple function that transforms the state dict.
- The `_schema_version` field is always updated after migration.

## Corruption Recovery

If a checkpoint fails integrity verification:

```python
def recover_checkpoint(run_dir: Path) -> dict | None:
    """Find the latest valid checkpoint, quarantine corrupted ones."""
    checkpoints = sorted(run_dir.glob("checkpoint_*.json"), reverse=True)

    for cp_path in checkpoints:
        try:
            state = load_checkpoint(cp_path)
            return migrate_state(state)
        except CheckpointCorruptionError:
            # Move corrupted checkpoint to quarantine
            quarantine_dir = run_dir / "quarantine"
            quarantine_dir.mkdir(exist_ok=True)
            cp_path.rename(quarantine_dir / cp_path.name)
            hash_path = cp_path.with_suffix(".sha256")
            if hash_path.exists():
                hash_path.rename(quarantine_dir / hash_path.name)

    return None  # No valid checkpoints found
```

**Quarantine behavior:**
- Corrupted files are moved to `{run_dir}/quarantine/`, not deleted.
- The system falls back to the next-oldest checkpoint.
- If no valid checkpoints exist, the run restarts from scratch.

## Checkpoint Rotation

To avoid unbounded disk usage, only the N most recent checkpoints are retained (default: 5, configured via `checkpoints.max_checkpoints`):

**Safety rule:** Always keep >= 2 checkpoints to prevent total state loss. If the most recent checkpoint is corrupted (partial write during crash, disk error), the system must have at least one older checkpoint to fall back to. The rotation logic enforces this minimum regardless of configuration.

```python
def rotate_checkpoints(run_dir: Path, max_keep: int = 5) -> None:
    # Safety: never keep fewer than 2 checkpoints
    max_keep = max(max_keep, 2)
    checkpoints = sorted(run_dir.glob("checkpoint_*.json"))
    to_remove = checkpoints[:-max_keep] if len(checkpoints) > max_keep else []
    for cp_path in to_remove:
        cp_path.unlink()
        hash_path = cp_path.with_suffix(".sha256")
        if hash_path.exists():
            hash_path.unlink()
```

## Resume Flow

```
Start
  |
  v
Run directory exists?
  |
  +-- No --> Fresh start (step 0)
  |
  +-- Yes --> Load latest valid checkpoint
                |
                +-- Success --> Verify hash -> Migrate schema -> Resume from step N
                |
                +-- All corrupted --> Log warning -> Fresh start (step 0)
```

## File Location

```
src/research_agent/
    checkpoint.py     # atomic_write_checkpoint, load_checkpoint, recovery
    event_log.py      # EventLog class
    progress.py       # append_progress
```
