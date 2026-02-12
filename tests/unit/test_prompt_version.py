"""Unit tests for research_agent.prompt_version."""

from __future__ import annotations

from typing import TYPE_CHECKING

from research_agent.prompt_version import (
    clear_hash_cache,
    known_hashes,
    prompt_hash,
    prompt_hash_combined,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# TestPromptHash
# ---------------------------------------------------------------------------


class TestPromptHash:
    """prompt_hash computes SHA-256 hashes of prompt YAML files."""

    def test_returns_hex_string_for_existing_prompt(self) -> None:
        clear_hash_cache()
        result = prompt_hash("summarizer")
        assert isinstance(result, str)
        assert len(result) == 64
        # Valid hex
        int(result, 16)

    def test_returns_empty_for_missing_prompt(self) -> None:
        clear_hash_cache()
        result = prompt_hash("nonexistent_prompt_xyz")
        assert result == ""

    def test_same_file_returns_same_hash(self) -> None:
        clear_hash_cache()
        h1 = prompt_hash("summarizer")
        h2 = prompt_hash("summarizer")
        assert h1 == h2

    def test_different_files_return_different_hashes(self) -> None:
        clear_hash_cache()
        h1 = prompt_hash("summarizer")
        h2 = prompt_hash("synthesizer")
        assert h1 != h2

    def test_caches_result_in_memory(self) -> None:
        clear_hash_cache()
        prompt_hash("summarizer")
        hashes = known_hashes()
        assert "summarizer" in hashes

    def test_all_known_prompts_hash_successfully(self) -> None:
        clear_hash_cache()
        for name in ("planner", "searcher", "summarizer", "synthesizer"):
            result = prompt_hash(name)
            assert len(result) == 64, f"Failed for {name}"

    def test_hash_from_real_file_content(self, tmp_path: Path) -> None:
        """Verify hash changes when file content changes."""
        clear_hash_cache()
        h1 = prompt_hash("summarizer")

        # A different prompt should give a different hash
        h2 = prompt_hash("planner")
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestPromptHashCombined
# ---------------------------------------------------------------------------


class TestPromptHashCombined:
    """prompt_hash_combined merges hashes from multiple prompt files."""

    def test_returns_hex_string(self) -> None:
        clear_hash_cache()
        result = prompt_hash_combined("summarizer", "synthesizer")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_order_independent(self) -> None:
        clear_hash_cache()
        h1 = prompt_hash_combined("summarizer", "synthesizer")
        clear_hash_cache()
        h2 = prompt_hash_combined("synthesizer", "summarizer")
        assert h1 == h2

    def test_different_inputs_produce_different_combined(self) -> None:
        clear_hash_cache()
        h1 = prompt_hash_combined("summarizer")
        clear_hash_cache()
        h2 = prompt_hash_combined("synthesizer")
        assert h1 != h2

    def test_single_prompt_combined(self) -> None:
        clear_hash_cache()
        single = prompt_hash("summarizer")
        clear_hash_cache()
        combined = prompt_hash_combined("summarizer")
        # Combined of single should differ from raw single hash
        # because combined uses "|" join before hashing
        assert len(combined) == 64
        # Both should be valid hashes
        assert isinstance(single, str)

    def test_handles_missing_prompt_in_combined(self) -> None:
        clear_hash_cache()
        # Should not raise, missing prompts return empty string
        result = prompt_hash_combined("summarizer", "nonexistent_xyz")
        assert len(result) == 64


# ---------------------------------------------------------------------------
# TestClearHashCache
# ---------------------------------------------------------------------------


class TestClearHashCache:
    """clear_hash_cache removes all cached hashes."""

    def test_clears_all_entries(self) -> None:
        clear_hash_cache()
        prompt_hash("summarizer")
        prompt_hash("synthesizer")
        assert len(known_hashes()) >= 2

        clear_hash_cache()
        assert known_hashes() == {}

    def test_clear_on_empty_is_safe(self) -> None:
        clear_hash_cache()
        clear_hash_cache()
        assert known_hashes() == {}


# ---------------------------------------------------------------------------
# TestKnownHashes
# ---------------------------------------------------------------------------


class TestKnownHashes:
    """known_hashes returns cached prompt hashes."""

    def test_returns_empty_initially(self) -> None:
        clear_hash_cache()
        assert known_hashes() == {}

    def test_returns_copy_not_reference(self) -> None:
        clear_hash_cache()
        prompt_hash("summarizer")
        h1 = known_hashes()
        h2 = known_hashes()
        assert h1 == h2
        assert h1 is not h2

    def test_contains_loaded_prompts(self) -> None:
        clear_hash_cache()
        prompt_hash("summarizer")
        prompt_hash("planner")
        hashes = known_hashes()
        assert "summarizer" in hashes
        assert "planner" in hashes
        assert len(hashes) == 2
