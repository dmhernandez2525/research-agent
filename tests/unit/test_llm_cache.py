"""Unit tests for research_agent.llm_cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from research_agent.llm_cache import LLMCache, _build_cache_key

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# TestBuildCacheKey
# ---------------------------------------------------------------------------


class TestBuildCacheKey:
    """_build_cache_key produces deterministic SHA-256 keys."""

    def test_same_inputs_produce_same_key(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        key1 = _build_cache_key("model-a", 0.0, messages)
        key2 = _build_cache_key("model-a", 0.0, messages)
        assert key1 == key2

    def test_different_model_produces_different_key(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        key1 = _build_cache_key("model-a", 0.0, messages)
        key2 = _build_cache_key("model-b", 0.0, messages)
        assert key1 != key2

    def test_different_temperature_produces_different_key(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        key1 = _build_cache_key("model-a", 0.0, messages)
        key2 = _build_cache_key("model-a", 0.5, messages)
        assert key1 != key2

    def test_different_messages_produce_different_key(self) -> None:
        key1 = _build_cache_key("m", 0.0, [{"role": "user", "content": "a"}])
        key2 = _build_cache_key("m", 0.0, [{"role": "user", "content": "b"}])
        assert key1 != key2

    def test_extra_changes_key(self) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        key1 = _build_cache_key("m", 0.0, messages, extra="")
        key2 = _build_cache_key("m", 0.0, messages, extra="v2-hash")
        assert key1 != key2

    def test_returns_hex_string(self) -> None:
        key = _build_cache_key("m", 0.0, [])
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        int(key, 16)  # Valid hex

    def test_message_order_matters(self) -> None:
        m1: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        m2: list[dict[str, Any]] = [
            {"role": "user", "content": "usr"},
            {"role": "system", "content": "sys"},
        ]
        key1 = _build_cache_key("m", 0.0, m1)
        key2 = _build_cache_key("m", 0.0, m2)
        assert key1 != key2


# ---------------------------------------------------------------------------
# TestLLMCacheInit
# ---------------------------------------------------------------------------


class TestLLMCacheInit:
    """LLMCache initialization and configuration."""

    def test_default_values(self) -> None:
        cache = LLMCache()
        assert cache.ttl_seconds == 86400
        assert cache.max_temperature == 0.0

    def test_custom_values(self, tmp_path: Path) -> None:
        cache = LLMCache(
            cache_dir=tmp_path / "custom",
            ttl_seconds=3600,
            max_temperature=0.5,
        )
        assert cache.ttl_seconds == 3600
        assert cache.max_temperature == 0.5


# ---------------------------------------------------------------------------
# TestLLMCacheGetSet
# ---------------------------------------------------------------------------


class TestLLMCacheGetSet:
    """LLMCache get/set operations."""

    def test_get_returns_none_on_miss(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        result = cache.get("model", 0.0, [{"role": "user", "content": "hi"}])
        assert result is None

    def test_set_and_get_round_trip(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        response = {"choices": [{"message": {"content": "world"}}]}

        cache.set("model-a", 0.0, messages, response)
        result = cache.get("model-a", 0.0, messages)
        assert result == response

    def test_set_returns_true_on_success(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        ok = cache.set("m", 0.0, [], {"data": "value"})
        assert ok is True

    def test_skips_cache_for_high_temperature(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache", max_temperature=0.0)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        response = {"choices": []}

        # set should return False for temp > max_temperature
        ok = cache.set("m", 0.5, messages, response)
        assert ok is False

        # get should return None for temp > max_temperature
        result = cache.get("m", 0.5, messages)
        assert result is None

    def test_caches_at_max_temperature(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache", max_temperature=0.5)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        response = {"data": "cached"}

        ok = cache.set("m", 0.5, messages, response)
        assert ok is True

        result = cache.get("m", 0.5, messages)
        assert result == response

    def test_different_extra_separate_entries(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        resp_v1 = {"version": 1}
        resp_v2 = {"version": 2}

        cache.set("m", 0.0, messages, resp_v1, extra="hash-v1")
        cache.set("m", 0.0, messages, resp_v2, extra="hash-v2")

        assert cache.get("m", 0.0, messages, extra="hash-v1") == resp_v1
        assert cache.get("m", 0.0, messages, extra="hash-v2") == resp_v2

    def test_get_returns_none_when_diskcache_unavailable(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        with patch.dict("sys.modules", {"diskcache": None}):
            cache._cache = None  # Reset lazy init
            result = cache.get("m", 0.0, [])
        assert result is None

    def test_set_returns_false_when_diskcache_unavailable(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        with patch.dict("sys.modules", {"diskcache": None}):
            cache._cache = None
            ok = cache.set("m", 0.0, [], {"data": "x"})
        assert ok is False


# ---------------------------------------------------------------------------
# TestLLMCacheClear
# ---------------------------------------------------------------------------


class TestLLMCacheClear:
    """LLMCache clear operation."""

    def test_clear_empties_cache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        cache.set("m", 0.0, [{"role": "user", "content": "a"}], {"a": 1})
        cache.set("m", 0.0, [{"role": "user", "content": "b"}], {"b": 2})

        removed = cache.clear()
        assert removed == 2
        assert cache.size == 0

    def test_clear_on_empty_cache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        removed = cache.clear()
        assert removed == 0

    def test_clear_returns_zero_without_diskcache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        with patch.dict("sys.modules", {"diskcache": None}):
            cache._cache = None
            removed = cache.clear()
        assert removed == 0


# ---------------------------------------------------------------------------
# TestLLMCacheSize
# ---------------------------------------------------------------------------


class TestLLMCacheSize:
    """LLMCache size property."""

    def test_size_starts_at_zero(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        assert cache.size == 0

    def test_size_increments_on_set(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        cache.set("m", 0.0, [{"role": "user", "content": "a"}], {"a": 1})
        assert cache.size == 1
        cache.set("m", 0.0, [{"role": "user", "content": "b"}], {"b": 2})
        assert cache.size == 2

    def test_size_returns_zero_without_diskcache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        with patch.dict("sys.modules", {"diskcache": None}):
            cache._cache = None
            assert cache.size == 0


# ---------------------------------------------------------------------------
# TestLLMCacheClose
# ---------------------------------------------------------------------------


class TestLLMCacheClose:
    """LLMCache close operation."""

    def test_close_resets_internal_cache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        # Initialize the cache by doing a set
        cache.set("m", 0.0, [], {"data": "x"})
        assert cache._cache is not None

        cache.close()
        assert cache._cache is None

    def test_close_on_uninitialized_cache(self, tmp_path: Path) -> None:
        cache = LLMCache(cache_dir=tmp_path / "cache")
        # Should not raise
        cache.close()
        assert cache._cache is None

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        cache = LLMCache(cache_dir=nested)
        cache.set("m", 0.0, [], {"x": 1})
        assert nested.exists()
        cache.close()
