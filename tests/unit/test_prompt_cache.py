"""Unit tests for research_agent.prompt_cache - caching optimization."""

from __future__ import annotations

import json

from research_agent.prompt_cache import (
    _CACHE_CONTROL_EPHEMERAL,
    _CACHE_READ_COST_MULTIPLIER,
    CacheTracker,
    deterministic_json,
    order_messages_for_cache,
)

# ---------------------------------------------------------------------------
# TestDeterministicJson
# ---------------------------------------------------------------------------


class TestDeterministicJson:
    """deterministic_json produces stable, sorted JSON."""

    def test_sorted_keys(self) -> None:
        result = deterministic_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_no_whitespace(self) -> None:
        result = deterministic_json({"key": "value"})
        assert " " not in result

    def test_nested_objects_sorted(self) -> None:
        obj = {"z": {"b": 2, "a": 1}, "a": 1}
        result = deterministic_json(obj)
        parsed = json.loads(result)
        keys = list(parsed.keys())
        assert keys == ["a", "z"]

    def test_same_dict_different_order(self) -> None:
        d1 = {"c": 3, "a": 1, "b": 2}
        d2 = {"a": 1, "b": 2, "c": 3}
        assert deterministic_json(d1) == deterministic_json(d2)

    def test_lists_preserved(self) -> None:
        result = deterministic_json({"items": [3, 1, 2]})
        assert json.loads(result)["items"] == [3, 1, 2]

    def test_empty_dict(self) -> None:
        assert deterministic_json({}) == "{}"


# ---------------------------------------------------------------------------
# TestOrderMessagesForCache
# ---------------------------------------------------------------------------


class TestOrderMessagesForCache:
    """order_messages_for_cache builds cache-stable payloads."""

    def test_system_block_has_cache_control(self) -> None:
        result = order_messages_for_cache(system_prompt="You are a researcher.")
        system = result["system"]
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are a researcher."
        assert system[0]["cache_control"] == _CACHE_CONTROL_EPHEMERAL

    def test_tools_have_cache_control(self) -> None:
        tools = [{"name": "search", "parameters": {"query": "string"}}]
        result = order_messages_for_cache(system_prompt="test", tool_definitions=tools)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["cache_control"] == _CACHE_CONTROL_EPHEMERAL

    def test_tools_serialized_deterministically(self) -> None:
        tools = [{"b_param": "value", "a_param": "value"}]
        result = order_messages_for_cache(system_prompt="test", tool_definitions=tools)
        tool = result["tools"][0]
        # Keys should be sorted after deterministic serialization
        keys = [k for k in tool if k != "cache_control"]
        assert keys == sorted(keys)

    def test_conversation_appended(self) -> None:
        conv = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
        ]
        result = order_messages_for_cache(system_prompt="test", conversation=conv)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "first"

    def test_latest_message_appended(self) -> None:
        conv = [{"role": "user", "content": "first"}]
        latest = {"role": "user", "content": "second"}
        result = order_messages_for_cache(
            system_prompt="test", conversation=conv, latest_message=latest
        )
        assert len(result["messages"]) == 2
        assert result["messages"][-1]["content"] == "second"

    def test_empty_tools_and_conversation(self) -> None:
        result = order_messages_for_cache(system_prompt="test")
        assert result["tools"] == []
        assert result["messages"] == []

    def test_message_order_stability(self) -> None:
        conv = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        result1 = order_messages_for_cache(system_prompt="test", conversation=conv)
        result2 = order_messages_for_cache(system_prompt="test", conversation=conv)
        assert result1["messages"] == result2["messages"]

    def test_returns_all_keys(self) -> None:
        result = order_messages_for_cache(system_prompt="test")
        assert "system" in result
        assert "tools" in result
        assert "messages" in result


# ---------------------------------------------------------------------------
# TestCacheTrackerInit
# ---------------------------------------------------------------------------


class TestCacheTrackerInit:
    """CacheTracker initializes with zero counts."""

    def test_initial_state(self) -> None:
        tracker = CacheTracker()
        assert tracker.total_calls == 0
        assert tracker.cache_hits == 0
        assert tracker.cache_misses == 0

    def test_initial_hit_rate(self) -> None:
        tracker = CacheTracker()
        assert tracker.hit_rate == 0.0


# ---------------------------------------------------------------------------
# TestCacheTrackerRecordCall
# ---------------------------------------------------------------------------


class TestCacheTrackerRecordCall:
    """record_call tracks hits and misses."""

    def test_cache_hit(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=1000, cached_tokens=800)
        assert tracker.cache_hits == 1
        assert tracker.cache_misses == 0
        assert tracker.total_calls == 1

    def test_cache_miss(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=1000, cached_tokens=0)
        assert tracker.cache_hits == 0
        assert tracker.cache_misses == 1

    def test_multiple_calls(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=500, cached_tokens=400)
        tracker.record_call(input_tokens=500, cached_tokens=0)
        tracker.record_call(input_tokens=500, cached_tokens=300)
        assert tracker.total_calls == 3
        assert tracker.cache_hits == 2
        assert tracker.cache_misses == 1

    def test_accumulates_tokens(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=100, cached_tokens=50)
        tracker.record_call(input_tokens=200, cached_tokens=100)
        assert tracker._total_input_tokens == 300
        assert tracker._cached_input_tokens == 150


# ---------------------------------------------------------------------------
# TestCacheTrackerHitRate
# ---------------------------------------------------------------------------


class TestCacheTrackerHitRate:
    """hit_rate computes correct ratio."""

    def test_all_hits(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=100, cached_tokens=80)
        tracker.record_call(input_tokens=100, cached_tokens=90)
        assert tracker.hit_rate == 1.0

    def test_no_hits(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=100)
        tracker.record_call(input_tokens=100)
        assert tracker.hit_rate == 0.0

    def test_mixed_hits(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=100, cached_tokens=80)
        tracker.record_call(input_tokens=100)
        assert tracker.hit_rate == 0.5

    def test_zero_calls(self) -> None:
        tracker = CacheTracker()
        assert tracker.hit_rate == 0.0


# ---------------------------------------------------------------------------
# TestCacheTrackerEstimatedSavings
# ---------------------------------------------------------------------------


class TestCacheTrackerEstimatedSavings:
    """estimated_savings calculates cost reduction from cache hits."""

    def test_no_cached_tokens(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=1000)
        assert tracker.estimated_savings(3.0) == 0.0

    def test_savings_calculation(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=1_000_000, cached_tokens=1_000_000)
        # Cost per million = $3.00
        # Uncached cost = 1M * $3/1M = $3.00
        # Cached cost = 1M * $3/1M * 0.10 = $0.30
        # Savings = $3.00 - $0.30 = $2.70
        savings = tracker.estimated_savings(3.0)
        assert abs(savings - 2.70) < 0.01

    def test_partial_cache(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=1000, cached_tokens=500)
        # Only 500 tokens cached; savings on those 500
        savings = tracker.estimated_savings(3.0)
        expected = (500 * 3.0 / 1_000_000) * (1.0 - _CACHE_READ_COST_MULTIPLIER)
        assert abs(savings - expected) < 1e-10

    def test_zero_calls(self) -> None:
        tracker = CacheTracker()
        assert tracker.estimated_savings(3.0) == 0.0


# ---------------------------------------------------------------------------
# TestCacheTrackerSummary
# ---------------------------------------------------------------------------


class TestCacheTrackerSummary:
    """summary() returns a dict of cache statistics."""

    def test_empty_summary(self) -> None:
        tracker = CacheTracker()
        summary = tracker.summary()
        assert summary["total_calls"] == 0
        assert summary["hit_rate"] == 0.0

    def test_populated_summary(self) -> None:
        tracker = CacheTracker()
        tracker.record_call(input_tokens=500, cached_tokens=400)
        tracker.record_call(input_tokens=300, cached_tokens=0)
        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["cache_hits"] == 1
        assert summary["cache_misses"] == 1
        assert summary["hit_rate"] == 0.5
        assert summary["total_input_tokens"] == 800
        assert summary["cached_input_tokens"] == 400

    def test_summary_returns_dict(self) -> None:
        tracker = CacheTracker()
        assert isinstance(tracker.summary(), dict)

    def test_all_keys_present(self) -> None:
        tracker = CacheTracker()
        summary = tracker.summary()
        expected_keys = {
            "total_calls",
            "cache_hits",
            "cache_misses",
            "hit_rate",
            "total_input_tokens",
            "cached_input_tokens",
        }
        assert set(summary.keys()) == expected_keys
