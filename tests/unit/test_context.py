"""Unit tests for research_agent.context - rolling window context manager."""

from __future__ import annotations

from research_agent.context import (
    CompactionResult,
    ContextManager,
    Turn,
)

# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------


class TestTurn:
    """Turn pydantic model."""

    def test_minimal_construction(self) -> None:
        turn = Turn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.token_count == 0
        assert turn.step_name == ""
        assert turn.masked is False

    def test_tool_turn(self) -> None:
        turn = Turn(role="tool", content="result data", token_count=500, step_name="search")
        assert turn.role == "tool"
        assert turn.token_count == 500
        assert turn.step_name == "search"

    def test_all_roles_valid(self) -> None:
        for role in ("system", "user", "assistant", "tool"):
            turn = Turn(role=role, content="test")
            assert turn.role == role

    def test_masked_default_false(self) -> None:
        turn = Turn(role="user", content="test")
        assert turn.masked is False


# ---------------------------------------------------------------------------
# CompactionResult model
# ---------------------------------------------------------------------------


class TestCompactionResult:
    """CompactionResult pydantic model."""

    def test_construction(self) -> None:
        result = CompactionResult(
            original_tokens=5000,
            compacted_tokens=2000,
            turns_masked=3,
            turns_total=15,
        )
        assert result.original_tokens == 5000
        assert result.compacted_tokens == 2000
        assert result.turns_masked == 3
        assert result.turns_total == 15


# ---------------------------------------------------------------------------
# ContextManager initialization
# ---------------------------------------------------------------------------


class TestContextManagerInit:
    """ContextManager initialization and defaults."""

    def test_default_window_size(self) -> None:
        mgr = ContextManager()
        assert mgr.window_size == 10

    def test_default_max_tokens(self) -> None:
        mgr = ContextManager()
        assert mgr.max_tokens == 100_000

    def test_custom_window_size(self) -> None:
        mgr = ContextManager(window_size=5)
        assert mgr.window_size == 5

    def test_custom_max_tokens(self) -> None:
        mgr = ContextManager(max_tokens=50_000)
        assert mgr.max_tokens == 50_000

    def test_custom_cooldown(self) -> None:
        mgr = ContextManager(compaction_cooldown_turns=5)
        assert mgr.compaction_cooldown_turns == 5

    def test_starts_empty(self) -> None:
        mgr = ContextManager()
        assert mgr.turn_count == 0
        assert mgr.total_tokens == 0
        assert mgr.turns == []


# ---------------------------------------------------------------------------
# Adding turns
# ---------------------------------------------------------------------------


class TestAddTurn:
    """Adding turns to the context manager."""

    def test_add_single_turn(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="Hello", token_count=5))
        assert mgr.turn_count == 1
        assert mgr.total_tokens == 5

    def test_add_multiple_turns(self) -> None:
        mgr = ContextManager()
        for i in range(5):
            mgr.add_turn(Turn(role="user", content=f"Turn {i}", token_count=10))
        assert mgr.turn_count == 5
        assert mgr.total_tokens == 50

    def test_turns_returns_copy(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="Hello"))
        turns = mgr.turns
        turns.append(Turn(role="user", content="Extra"))
        assert mgr.turn_count == 1  # original unchanged


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Total token count across all turns."""

    def test_zero_with_no_turns(self) -> None:
        mgr = ContextManager()
        assert mgr.total_tokens == 0

    def test_accumulates_tokens(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="a", token_count=100))
        mgr.add_turn(Turn(role="assistant", content="b", token_count=200))
        assert mgr.total_tokens == 300

    def test_tokens_updated_after_compaction(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=1_000_000)
        for i in range(5):
            mgr.add_turn(
                Turn(role="tool", content="x" * 100, token_count=500, step_name=f"step-{i}")
            )
        original = mgr.total_tokens
        mgr.compact()
        assert mgr.total_tokens < original


# ---------------------------------------------------------------------------
# Window sizing
# ---------------------------------------------------------------------------


class TestWindowSizing:
    """Window size controls which turns remain unmasked."""

    def test_within_window_not_masked(self) -> None:
        mgr = ContextManager(window_size=5, max_tokens=1_000_000)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        assert all(not t.masked for t in mgr.turns)

    def test_beyond_window_tool_turns_masked(self) -> None:
        mgr = ContextManager(window_size=3, max_tokens=1_000_000)
        for i in range(6):
            mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) == 3

    def test_only_tool_turns_masked(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=1_000_000)
        mgr.add_turn(Turn(role="user", content="question", token_count=50))
        mgr.add_turn(Turn(role="tool", content="answer data", token_count=500, step_name="search"))
        mgr.add_turn(Turn(role="assistant", content="summary", token_count=100))
        mgr.add_turn(Turn(role="user", content="followup", token_count=50))
        mgr.compact()
        assert mgr.turns[0].masked is False  # user turn, not maskable
        assert mgr.turns[1].masked is True   # tool turn outside window

    def test_window_size_one(self) -> None:
        mgr = ContextManager(window_size=1, max_tokens=1_000_000)
        for i in range(4):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) == 3

    def test_large_window_no_masking(self) -> None:
        mgr = ContextManager(window_size=100, max_tokens=1_000_000)
        for i in range(10):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 0


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    """compact() masks older tool turns."""

    def test_compact_returns_result(self) -> None:
        mgr = ContextManager(window_size=2)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert isinstance(result, CompactionResult)

    def test_compact_reduces_tokens(self) -> None:
        mgr = ContextManager(window_size=2)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="data", token_count=500, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.compacted_tokens < result.original_tokens

    def test_compact_reports_masked_count(self) -> None:
        mgr = ContextManager(window_size=2)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 3
        assert result.turns_total == 5

    def test_masked_content_includes_step_name(self) -> None:
        mgr = ContextManager(window_size=1)
        mgr.add_turn(Turn(role="tool", content="big output", token_count=1000, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        assert mgr.turns[0].content == "[masked tool output from search]"

    def test_already_masked_not_remasked(self) -> None:
        mgr = ContextManager(window_size=1)
        for i in range(3):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result1 = mgr.compact()
        result2 = mgr.compact()
        assert result1.turns_masked == 2
        assert result2.turns_masked == 0

    def test_masked_token_count_reduced(self) -> None:
        mgr = ContextManager(window_size=1)
        mgr.add_turn(Turn(role="tool", content="big data", token_count=5000, step_name="scrape"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        assert mgr.turns[0].token_count == 10

    def test_compact_empty_context(self) -> None:
        mgr = ContextManager()
        result = mgr.compact()
        assert result.turns_masked == 0
        assert result.turns_total == 0


# ---------------------------------------------------------------------------
# Compaction cooldown
# ---------------------------------------------------------------------------


class TestCompactionCooldown:
    """Cooldown prevents repeated O(n) scans."""

    def test_cooldown_skips_compaction(self) -> None:
        mgr = ContextManager(
            window_size=1,
            max_tokens=50,
            compaction_cooldown_turns=3,
        )
        # Add a non-tool turn that can't be masked
        mgr.add_turn(Turn(role="user", content="big", token_count=100))
        # Compaction triggered but nothing to mask, cooldown starts
        # Add 2 more turns within cooldown period
        mgr.add_turn(Turn(role="user", content="small", token_count=10))
        mgr.add_turn(Turn(role="user", content="small", token_count=10))
        # All turns should still be unmasked (cooldown in effect)
        assert all(not t.masked for t in mgr.turns)

    def test_cooldown_expires_after_threshold(self) -> None:
        mgr = ContextManager(
            window_size=1,
            max_tokens=50,
            compaction_cooldown_turns=2,
        )
        # First turn triggers compaction with nothing to mask
        mgr.add_turn(Turn(role="user", content="big", token_count=100))
        # Within cooldown
        mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name="s1"))
        # Cooldown expires (2 turns since last compaction)
        mgr.add_turn(Turn(role="tool", content="data2", token_count=100, step_name="s2"))
        # The oldest tool turn should be masked
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) >= 1

    def test_manual_compact_resets_cooldown(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=1_000_000)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 3
        assert mgr._compaction_pending is False


# ---------------------------------------------------------------------------
# Context window formatting
# ---------------------------------------------------------------------------


class TestGetContextWindow:
    """get_context_window formats turns for LLM consumption."""

    def test_formats_as_role_content_dicts(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="Hello"))
        mgr.add_turn(Turn(role="assistant", content="Hi"))
        window = mgr.get_context_window()
        assert len(window) == 2
        assert window[0] == {"role": "user", "content": "Hello"}
        assert window[1] == {"role": "assistant", "content": "Hi"}

    def test_includes_masked_turns(self) -> None:
        mgr = ContextManager(window_size=1)
        mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        window = mgr.get_context_window()
        assert window[0]["content"] == "[masked tool output from search]"

    def test_empty_context(self) -> None:
        mgr = ContextManager()
        assert mgr.get_context_window() == []


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    """clear() removes all turns."""

    def test_clear_empties_turns(self) -> None:
        mgr = ContextManager()
        for i in range(5):
            mgr.add_turn(Turn(role="user", content=f"Turn {i}", token_count=10))
        mgr.clear()
        assert mgr.turn_count == 0
        assert mgr.total_tokens == 0
        assert mgr.turns == []

    def test_clear_then_add(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="old", token_count=50))
        mgr.clear()
        mgr.add_turn(Turn(role="user", content="new", token_count=30))
        assert mgr.turn_count == 1
        assert mgr.total_tokens == 30


# ---------------------------------------------------------------------------
# Auto-compaction on add
# ---------------------------------------------------------------------------


class TestAutoCompaction:
    """add_turn auto-triggers compaction when over budget."""

    def test_auto_compacts_when_over_budget(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=100, compaction_cooldown_turns=0)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d" * 50, token_count=50, step_name=f"s-{i}"))
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) >= 1

    def test_no_auto_compact_under_budget(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=100_000)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=10, step_name=f"s-{i}"))
        assert all(not t.masked for t in mgr.turns)


# ---------------------------------------------------------------------------
# Integration: realistic conversation
# ---------------------------------------------------------------------------


class TestRealisticConversation:
    """Simulates a realistic research agent conversation."""

    def test_mixed_role_conversation(self) -> None:
        mgr = ContextManager(window_size=4, max_tokens=1_000_000)
        turns = [
            Turn(role="system", content="You are a researcher.", token_count=50),
            Turn(role="user", content="Research AI safety", token_count=20),
            Turn(role="assistant", content="Planning subtopics...", token_count=100),
            Turn(role="tool", content="Search results: ..." * 100, token_count=2000, step_name="search"),
            Turn(role="assistant", content="Found 8 results.", token_count=50),
            Turn(role="tool", content="Scraped content: ..." * 50, token_count=1500, step_name="scrape"),
            Turn(role="assistant", content="Summarizing...", token_count=200),
            Turn(role="user", content="Continue", token_count=10),
        ]
        for t in turns:
            mgr.add_turn(t)

        result = mgr.compact()
        # Tool turn at index 3 should be masked (outside window of 4)
        assert mgr.turns[3].masked is True
        assert mgr.turns[3].content == "[masked tool output from search]"
        # Scrape turn at index 5 is within window (indices 4-7)
        assert mgr.turns[5].masked is False
        assert result.turns_masked >= 1
