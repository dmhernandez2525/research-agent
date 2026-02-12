"""Unit tests for research_agent.context - rolling window context manager."""

from __future__ import annotations

from research_agent.context import (
    _FILE_POINTER_MIN_CHARS,
    _STAGE_1_THRESHOLD,
    _STAGE_2_THRESHOLD,
    _STAGE_3_THRESHOLD,
    CompactionResult,
    ContextManager,
    MaskingStage,
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
# MaskingStage enum
# ---------------------------------------------------------------------------


class TestMaskingStage:
    """MaskingStage enum ordering."""

    def test_ordering(self) -> None:
        assert MaskingStage.NONE < MaskingStage.STAGE_1
        assert MaskingStage.STAGE_1 < MaskingStage.STAGE_2
        assert MaskingStage.STAGE_2 < MaskingStage.STAGE_3

    def test_values(self) -> None:
        assert MaskingStage.NONE == 0
        assert MaskingStage.STAGE_1 == 1
        assert MaskingStage.STAGE_2 == 2
        assert MaskingStage.STAGE_3 == 3


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

    def test_default_stage(self) -> None:
        result = CompactionResult(
            original_tokens=0, compacted_tokens=0, turns_masked=0, turns_total=0,
        )
        assert result.stage_applied == MaskingStage.NONE

    def test_stage_applied(self) -> None:
        result = CompactionResult(
            original_tokens=100, compacted_tokens=50,
            turns_masked=1, turns_total=2,
            stage_applied=MaskingStage.STAGE_2,
        )
        assert result.stage_applied == MaskingStage.STAGE_2


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
        assert mgr.turn_count == 1


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
        # Use high max_tokens to prevent auto-compaction during add_turn,
        # but total tokens hit 80% when all are added (2500/3125=80% => stage 1)
        mgr = ContextManager(window_size=2, max_tokens=3125)
        for i in range(5):
            mgr.add_turn(
                Turn(role="tool", content="x" * 100, token_count=500, step_name=f"step-{i}")
            )
        original = mgr.total_tokens
        mgr.compact()
        assert mgr.total_tokens < original


# ---------------------------------------------------------------------------
# Utilization and active stage
# ---------------------------------------------------------------------------


class TestUtilization:
    """utilization_percent and active_stage properties."""

    def test_utilization_zero_when_empty(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        assert mgr.utilization_percent == 0.0

    def test_utilization_calculation(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        mgr.add_turn(Turn(role="user", content="a", token_count=500))
        assert mgr.utilization_percent == 50.0

    def test_utilization_over_100(self) -> None:
        mgr = ContextManager(max_tokens=100)
        mgr.add_turn(Turn(role="user", content="a", token_count=200))
        assert mgr.utilization_percent == 200.0

    def test_utilization_zero_max(self) -> None:
        mgr = ContextManager(max_tokens=0)
        assert mgr.utilization_percent == 0.0

    def test_stage_none_below_75(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        mgr.add_turn(Turn(role="user", content="a", token_count=740))
        assert mgr.active_stage == MaskingStage.NONE

    def test_stage_1_at_75(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        mgr.add_turn(Turn(role="user", content="a", token_count=750))
        assert mgr.active_stage == MaskingStage.STAGE_1

    def test_stage_2_at_80(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        mgr.add_turn(Turn(role="user", content="a", token_count=800))
        assert mgr.active_stage == MaskingStage.STAGE_2

    def test_stage_3_at_85(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        mgr.add_turn(Turn(role="user", content="a", token_count=850))
        assert mgr.active_stage == MaskingStage.STAGE_3

    def test_thresholds_are_correct(self) -> None:
        assert _STAGE_1_THRESHOLD == 75.0
        assert _STAGE_2_THRESHOLD == 80.0
        assert _STAGE_3_THRESHOLD == 85.0


# ---------------------------------------------------------------------------
# Window sizing (stage 1 tests use utilization >= 75%)
# ---------------------------------------------------------------------------


class TestWindowSizing:
    """Window size controls which turns remain unmasked."""

    def _make_mgr(self, window_size: int, total_tokens: int) -> ContextManager:
        """Create a ContextManager where utilization is >= 75%."""
        max_tokens = int(total_tokens / 0.80)  # 80% utilization -> stage 1+
        return ContextManager(window_size=window_size, max_tokens=max_tokens)

    def test_within_window_not_masked(self) -> None:
        # 5 turns * 100 tokens = 500; max_tokens=625 => 80% utilization
        mgr = self._make_mgr(window_size=5, total_tokens=500)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        assert all(not t.masked for t in mgr.turns)

    def test_beyond_window_tool_turns_masked(self) -> None:
        mgr = self._make_mgr(window_size=3, total_tokens=600)
        for i in range(6):
            mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) == 3

    def test_only_tool_turns_masked_at_stage_1(self) -> None:
        # 4 turns: total ~700 tokens; max_tokens=875 => 80% utilization (stage 1)
        # Stage 1 only masks tool turns
        mgr = ContextManager(window_size=2, max_tokens=875)
        mgr.add_turn(Turn(role="user", content="question", token_count=50))
        mgr.add_turn(Turn(role="tool", content="answer data", token_count=500, step_name="search"))
        mgr.add_turn(Turn(role="assistant", content="summary", token_count=100))
        mgr.add_turn(Turn(role="user", content="followup", token_count=50))
        mgr.compact()
        assert mgr.turns[0].masked is False  # user turn, not maskable at stage 1
        assert mgr.turns[1].masked is True   # tool turn outside window

    def test_window_size_one(self) -> None:
        mgr = self._make_mgr(window_size=1, total_tokens=400)
        for i in range(4):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) == 3

    def test_large_window_no_masking(self) -> None:
        mgr = self._make_mgr(window_size=100, total_tokens=1000)
        for i in range(10):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 0


# ---------------------------------------------------------------------------
# Compaction (using utilization levels that trigger masking)
# ---------------------------------------------------------------------------


class TestCompaction:
    """compact() masks older tool turns at appropriate stages."""

    def test_compact_returns_result(self) -> None:
        # 5*100=500; max_tokens=625 => 80% => stage 1
        mgr = ContextManager(window_size=2, max_tokens=625)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert isinstance(result, CompactionResult)

    def test_compact_reduces_tokens(self) -> None:
        # 5*500=2500 tokens; max_tokens=3125 => 80% => stage 1
        # High enough to avoid auto-compaction during add_turn
        mgr = ContextManager(window_size=2, max_tokens=3125)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="data", token_count=500, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.compacted_tokens < result.original_tokens

    def test_compact_reports_masked_count(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=625)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 3
        assert result.turns_total == 5

    def test_masked_content_includes_step_name(self) -> None:
        # 1010 tokens; max_tokens=1200 => ~84% => stage 2
        mgr = ContextManager(window_size=1, max_tokens=1200)
        mgr.add_turn(Turn(role="tool", content="big output", token_count=1000, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        assert mgr.turns[0].content == "[masked tool output from search]"

    def test_already_masked_not_remasked(self) -> None:
        mgr = ContextManager(window_size=1, max_tokens=375)
        for i in range(3):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result1 = mgr.compact()
        result2 = mgr.compact()
        assert result1.turns_masked == 2
        assert result2.turns_masked == 0

    def test_masked_token_count_reduced(self) -> None:
        mgr = ContextManager(window_size=1, max_tokens=5500)
        mgr.add_turn(Turn(role="tool", content="big data", token_count=5000, step_name="scrape"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        assert mgr.turns[0].token_count == 10

    def test_compact_empty_context(self) -> None:
        mgr = ContextManager()
        result = mgr.compact()
        assert result.turns_masked == 0
        assert result.turns_total == 0
        assert result.stage_applied == MaskingStage.NONE

    def test_no_masking_below_stage_1_threshold(self) -> None:
        # 500 tokens; max_tokens=1000 => 50% => NONE
        mgr = ContextManager(window_size=2, max_tokens=1000)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        result = mgr.compact()
        assert result.turns_masked == 0
        assert result.stage_applied == MaskingStage.NONE


# ---------------------------------------------------------------------------
# Three-stage masking
# ---------------------------------------------------------------------------


class TestThreeStageMasking:
    """Progressive masking stages activate based on utilization."""

    def test_stage_1_masks_only_tool_turns(self) -> None:
        # 400 tokens; max_tokens=500 => 80% => stage 1 (but < 80 => stage 1)
        # Actually 80% = stage 2. Let's use 76%.
        mgr = ContextManager(window_size=1, max_tokens=526)
        mgr.add_turn(Turn(role="assistant", content="summary", token_count=100, step_name="sum"))
        mgr.add_turn(Turn(role="tool", content="data", token_count=200, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=100))
        # Total=400, 400/526=76% => stage 1
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_1
        # Only tool turn (index 1) is outside window and maskable
        assert mgr.turns[1].masked is True
        # Assistant turn is NOT masked at stage 1
        assert mgr.turns[0].masked is False

    def test_stage_2_also_masks_assistant_turns(self) -> None:
        # Need 80-84.9% utilization
        mgr = ContextManager(window_size=1, max_tokens=500)
        mgr.add_turn(Turn(role="assistant", content="summary", token_count=100, step_name="sum"))
        mgr.add_turn(Turn(role="tool", content="data", token_count=200, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=100))
        # Total=400, 400/500=80% => stage 2
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_2
        assert mgr.turns[0].masked is True  # assistant masked at stage 2
        assert mgr.turns[1].masked is True  # tool masked at stage 1+
        assert mgr.turns[0].content == "[compressed summary from sum]"

    def test_stage_3_replaces_large_text_with_file_pointers(self) -> None:
        # Need >= 85% utilization
        large_content = "x" * (_FILE_POINTER_MIN_CHARS + 50)
        mgr = ContextManager(window_size=1, max_tokens=350)
        mgr.add_turn(Turn(role="user", content=large_content, token_count=200, step_name="input"))
        mgr.add_turn(Turn(role="tool", content="small", token_count=100, step_name="search"))
        # Total=300, 300/350=85.7% => stage 3
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_3
        # User turn has large content and is outside window
        assert mgr.turns[0].masked is True
        assert "content saved to file" in mgr.turns[0].content

    def test_stage_3_skips_small_content(self) -> None:
        # Stage 3 only replaces content >= _FILE_POINTER_MIN_CHARS
        mgr = ContextManager(window_size=1, max_tokens=235)
        mgr.add_turn(Turn(role="user", content="short", token_count=100, step_name="input"))
        mgr.add_turn(Turn(role="user", content="next", token_count=100))
        # Total=200, 200/235=85.1% => stage 3
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_3
        # "short" is < _FILE_POINTER_MIN_CHARS, so not replaced
        assert mgr.turns[0].masked is False

    def test_stage_reports_in_result(self) -> None:
        mgr = ContextManager(window_size=1, max_tokens=250)
        mgr.add_turn(Turn(role="tool", content="d", token_count=200, step_name="s-0"))
        # 200/250 = 80% => stage 2
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_2

    def test_progressive_activation(self) -> None:
        """As tokens grow, the stage escalates progressively."""
        mgr = ContextManager(window_size=1, max_tokens=1000)

        # Add turns incrementally and check stage
        mgr.add_turn(Turn(role="tool", content="a", token_count=500, step_name="s-0"))
        assert mgr.active_stage == MaskingStage.NONE  # 50%

        mgr.add_turn(Turn(role="tool", content="b", token_count=260, step_name="s-1"))
        assert mgr.active_stage == MaskingStage.STAGE_1  # 76%

        # After compaction, stage 1 masks the oldest tool turn
        result = mgr.compact()
        assert result.stage_applied == MaskingStage.STAGE_1


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
        mgr.add_turn(Turn(role="user", content="big", token_count=100))
        mgr.add_turn(Turn(role="user", content="small", token_count=10))
        mgr.add_turn(Turn(role="user", content="small", token_count=10))
        assert all(not t.masked for t in mgr.turns)

    def test_cooldown_expires_after_threshold(self) -> None:
        mgr = ContextManager(
            window_size=1,
            max_tokens=50,
            compaction_cooldown_turns=2,
        )
        mgr.add_turn(Turn(role="user", content="big", token_count=100))
        mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name="s1"))
        mgr.add_turn(Turn(role="tool", content="data2", token_count=100, step_name="s2"))
        masked = [t for t in mgr.turns if t.masked]
        assert len(masked) >= 1

    def test_manual_compact_resets_cooldown(self) -> None:
        # 5*100=500; max_tokens=625 => 80% => stage 1
        mgr = ContextManager(window_size=2, max_tokens=625)
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
        mgr = ContextManager(window_size=1, max_tokens=120)
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


class TestFormatForApi:
    """format_for_api assembles cache-stable API payloads."""

    def test_returns_all_keys(self) -> None:
        mgr = ContextManager()
        result = mgr.format_for_api("You are a researcher.")
        assert "system" in result
        assert "tools" in result
        assert "messages" in result

    def test_system_block_structure(self) -> None:
        mgr = ContextManager()
        result = mgr.format_for_api("System prompt text.")
        system = result["system"]
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "System prompt text."

    def test_includes_tool_definitions(self) -> None:
        mgr = ContextManager()
        tools = [{"name": "search", "parameters": {"query": "string"}}]
        result = mgr.format_for_api("test", tool_definitions=tools)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search"

    def test_empty_tools_by_default(self) -> None:
        mgr = ContextManager()
        result = mgr.format_for_api("test")
        assert result["tools"] == []

    def test_includes_conversation(self) -> None:
        mgr = ContextManager()
        mgr.add_turn(Turn(role="user", content="Hello"))
        mgr.add_turn(Turn(role="assistant", content="Hi"))
        result = mgr.format_for_api("test")
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "Hello"

    def test_messages_include_masked_content(self) -> None:
        mgr = ContextManager(window_size=1, max_tokens=120)
        mgr.add_turn(Turn(role="tool", content="data", token_count=100, step_name="search"))
        mgr.add_turn(Turn(role="user", content="next", token_count=10))
        mgr.compact()
        result = mgr.format_for_api("test")
        assert "[masked tool output" in result["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Window report
# ---------------------------------------------------------------------------


class TestWindowReport:
    """window_report provides diagnostic context state."""

    def test_empty_report(self) -> None:
        mgr = ContextManager(max_tokens=1000)
        report = mgr.window_report()
        assert report["turn_count"] == 0
        assert report["total_tokens"] == 0
        assert report["max_tokens"] == 1000
        assert report["utilization_percent"] == 0.0
        assert report["active_stage"] == "NONE"
        assert report["masked_count"] == 0
        assert report["unmasked_count"] == 0

    def test_populated_report(self) -> None:
        mgr = ContextManager(window_size=2, max_tokens=625)
        for i in range(5):
            mgr.add_turn(Turn(role="tool", content="d", token_count=100, step_name=f"s-{i}"))
        mgr.compact()
        report = mgr.window_report()
        assert report["turn_count"] == 5
        assert report["masked_count"] == 3
        assert report["unmasked_count"] == 2
        assert report["window_size"] == 2

    def test_report_keys(self) -> None:
        mgr = ContextManager()
        report = mgr.window_report()
        expected_keys = {
            "turn_count", "total_tokens", "max_tokens",
            "utilization_percent", "active_stage",
            "masked_count", "unmasked_count", "window_size",
        }
        assert set(report.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    """clear() removes all turns and resets state."""

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

    def test_clear_resets_compaction_state(self) -> None:
        mgr = ContextManager(max_tokens=50)
        mgr.add_turn(Turn(role="user", content="big", token_count=100))
        # Should have triggered compaction attempt
        mgr.clear()
        assert mgr._compaction_pending is False
        assert mgr._turns_since_compaction == 0


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
        # Total: 50+20+100+2000+50+1500+200+10 = 3930 tokens
        # max_tokens=4500 => 87.3% => stage 3
        mgr = ContextManager(window_size=4, max_tokens=4500)
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
