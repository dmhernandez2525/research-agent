"""Integration tests for crash simulation + checkpoint resume verification."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from research_agent.checkpoints import (
    CheckpointManager,
    generate_run_id,
)
from research_agent.exceptions import CheckpointCorruptionError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture()
def mgr(tmp_checkpoint_dir: Path) -> CheckpointManager:
    """Return a CheckpointManager backed by the tmp_checkpoint_dir fixture."""
    return CheckpointManager(directory=tmp_checkpoint_dir, max_checkpoints=10)


# ---- Crash simulation -------------------------------------------------------


class TestCrashSimulation:
    """Simulate crashes at various points and verify checkpoint integrity."""

    def test_crash_after_planning_resumes_from_plan(
        self,
        mgr: CheckpointManager,
        sample_state: dict[str, Any],
    ) -> None:
        """If a crash occurs after planning, resume should preserve sub_questions."""
        # Simulate planner output
        sample_state["step"] = "plan"
        sample_state["step_index"] = 1
        sample_state["sub_questions"] = [
            {"id": 1, "question": "What is RAG?"},
            {"id": 2, "question": "How does adaptive retrieval work?"},
        ]

        run_id = generate_run_id()
        cp_id = f"{run_id}-step-1"
        mgr.save(cp_id, sample_state, step_index=1, step_name="plan")

        # "Crash" - create a fresh manager (simulates new process)
        restored_mgr = CheckpointManager(directory=mgr.directory, max_checkpoints=10)
        loaded = restored_mgr.load(cp_id)

        assert loaded["step"] == "plan"
        assert loaded["step_index"] == 1
        assert len(loaded["sub_questions"]) == 2
        assert loaded["sub_questions"][0]["question"] == "What is RAG?"

    def test_crash_after_search_resumes_with_results(
        self,
        mgr: CheckpointManager,
        sample_state: dict[str, Any],
    ) -> None:
        """If a crash occurs after search, resume should have search results."""
        sample_state["step"] = "search"
        sample_state["step_index"] = 2
        sample_state["search_results"] = [
            {"sub_question_id": 1, "query": "RAG", "url": "https://example.com/1"},
            {"sub_question_id": 1, "query": "RAG", "url": "https://example.com/2"},
        ]

        cp_id = f"{generate_run_id()}-step-2"
        mgr.save(cp_id, sample_state, step_index=2, step_name="search")

        restored_mgr = CheckpointManager(directory=mgr.directory, max_checkpoints=10)
        loaded = restored_mgr.load(cp_id)

        assert loaded["step"] == "search"
        assert len(loaded["search_results"]) == 2
        assert loaded["search_results"][1]["url"] == "https://example.com/2"

    def test_crash_mid_scraping_preserves_partial_results(
        self,
        mgr: CheckpointManager,
        sample_state: dict[str, Any],
    ) -> None:
        """A crash during scraping should preserve already-scraped content."""
        # Simulate scraping 2 of 5 URLs before crash
        sample_state["step"] = "scrape"
        sample_state["step_index"] = 3
        sample_state["scraped_content"] = [
            {"url": "https://example.com/1", "content": "Content A", "word_count": 100},
            {"url": "https://example.com/2", "content": "Content B", "word_count": 200},
        ]
        sample_state["seen_urls"] = [
            "https://example.com/1",
            "https://example.com/2",
        ]

        cp_id = f"{generate_run_id()}-step-3"
        mgr.save(cp_id, sample_state, step_index=3, step_name="scrape")

        restored_mgr = CheckpointManager(directory=mgr.directory, max_checkpoints=10)
        loaded = restored_mgr.load(cp_id)

        assert len(loaded["scraped_content"]) == 2
        assert len(loaded["seen_urls"]) == 2
        assert loaded["scraped_content"][0]["content"] == "Content A"

    def test_corrupted_checkpoint_detected_on_resume(
        self,
        mgr: CheckpointManager,
        sample_state: dict[str, Any],
    ) -> None:
        """Corruption introduced between save and load is detected."""
        cp_id = f"{generate_run_id()}-corrupt"
        mgr.save(cp_id, sample_state, step_index=1, step_name="plan")

        # Corrupt the checkpoint file
        cp_path = mgr.directory / f"{cp_id}.json"
        cp_path.write_text("CORRUPTED BYTES")

        restored_mgr = CheckpointManager(directory=mgr.directory, max_checkpoints=10)
        with pytest.raises(CheckpointCorruptionError):
            restored_mgr.load(cp_id)


# ---- Resume verification ----------------------------------------------------


class TestResumeVerification:
    """Verify that resumed runs produce correct state continuity."""

    def test_multi_step_resume_preserves_full_state(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """Saving at multiple steps and loading the latest gives correct state."""
        run_id = generate_run_id()

        # Step 1: plan
        state_v1: dict[str, Any] = {
            "query": "test",
            "step": "plan",
            "step_index": 1,
            "sub_questions": [{"id": 1, "question": "Q1"}],
        }
        mgr.save(f"{run_id}-step-1", state_v1, step_index=1, step_name="plan")
        time.sleep(0.05)

        # Step 2: search
        state_v2: dict[str, Any] = {
            **state_v1,
            "step": "search",
            "step_index": 2,
            "search_results": [{"url": "https://a.com"}],
        }
        mgr.save(f"{run_id}-step-2", state_v2, step_index=2, step_name="search")

        # Find latest and load
        latest_id = mgr.latest()
        assert latest_id == f"{run_id}-step-2"

        loaded = mgr.load(latest_id)
        assert loaded["step"] == "search"
        assert loaded["step_index"] == 2
        assert len(loaded["sub_questions"]) == 1
        assert len(loaded["search_results"]) == 1

    def test_cost_accumulates_across_resume(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """Cost tracking should persist across checkpoint save/load."""
        run_id = generate_run_id()

        state: dict[str, Any] = {
            "query": "test",
            "cost_so_far": 0.05,
            "llm_call_count": 3,
        }
        mgr.save(f"{run_id}-step-2", state, step_index=2, step_name="search")

        loaded = mgr.load(f"{run_id}-step-2")
        assert loaded["cost_so_far"] == 0.05
        assert loaded["llm_call_count"] == 3

        # Simulate resumed run adding more cost
        loaded["cost_so_far"] += 0.03
        loaded["llm_call_count"] += 2
        mgr.save(f"{run_id}-step-3", loaded, step_index=3, step_name="scrape")

        final = mgr.load(f"{run_id}-step-3")
        assert final["cost_so_far"] == pytest.approx(0.08)
        assert final["llm_call_count"] == 5

    def test_iteration_counter_continues_after_resume(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """The iteration counter should continue from the checkpoint value."""
        run_id = generate_run_id()

        state: dict[str, Any] = {
            "query": "test",
            "current_subtopic_index": 2,
            "sub_questions": [{"id": i} for i in range(5)],
        }
        mgr.save(f"{run_id}-iter-2", state, step_index=2, step_name="summarize")

        loaded = mgr.load(f"{run_id}-iter-2")
        assert loaded["current_subtopic_index"] == 2

        # Simulate resumed iteration
        loaded["current_subtopic_index"] = 3
        mgr.save(f"{run_id}-iter-3", loaded, step_index=3, step_name="summarize")

        resumed = mgr.load(f"{run_id}-iter-3")
        assert resumed["current_subtopic_index"] == 3

    def test_run_id_scoping_isolates_runs(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """Different run IDs should not interfere with each other."""
        run_a = generate_run_id()
        run_b = generate_run_id()

        mgr.save(f"{run_a}-step-1", {"query": "run A"}, step_index=1)
        mgr.save(f"{run_b}-step-1", {"query": "run B"}, step_index=1)

        loaded_a = mgr.load(f"{run_a}-step-1")
        loaded_b = mgr.load(f"{run_b}-step-1")

        assert loaded_a["query"] == "run A"
        assert loaded_b["query"] == "run B"

    def test_large_state_round_trip(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """Large state with nested structures survives save/load."""
        state: dict[str, Any] = {
            "query": "complex query",
            "sub_questions": [
                {"id": i, "question": f"Question {i}", "rationale": f"Because {i}"}
                for i in range(20)
            ],
            "search_results": [
                {"sub_question_id": i, "query": f"q{i}", "url": f"https://site{i}.com"}
                for i in range(50)
            ],
            "scraped_content": [
                {"url": f"https://site{i}.com", "content": f"Content block {i}" * 100}
                for i in range(10)
            ],
            "error_log": [
                {"step": "search", "message": f"Timeout on attempt {i}"}
                for i in range(3)
            ],
        }

        cp_id = f"{generate_run_id()}-large"
        meta = mgr.save(cp_id, state, step_index=5, step_name="summarize")

        assert meta.state_size_bytes > 1000

        loaded = mgr.load(cp_id)
        assert len(loaded["sub_questions"]) == 20
        assert len(loaded["search_results"]) == 50
        assert len(loaded["scraped_content"]) == 10
        assert len(loaded["error_log"]) == 3

    def test_metadata_tracks_step_progression(
        self,
        mgr: CheckpointManager,
    ) -> None:
        """Metadata should accurately reflect step index and name."""
        run_id = generate_run_id()
        steps = [("plan", 1), ("search", 2), ("scrape", 3), ("summarize", 4)]

        for step_name, step_index in steps:
            mgr.save(
                f"{run_id}-step-{step_index}",
                {"step": step_name, "step_index": step_index},
                step_index=step_index,
                step_name=step_name,
            )
            time.sleep(0.02)

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 4
        # Newest first
        assert checkpoints[0].step_name == "summarize"
        assert checkpoints[0].step_index == 4
        assert checkpoints[-1].step_name == "plan"
        assert checkpoints[-1].step_index == 1
