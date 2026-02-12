"""Unit tests for research_agent.key_rotation."""

from __future__ import annotations

import time
from unittest.mock import patch

from research_agent.key_rotation import KeyRotator

# ---------------------------------------------------------------------------
# TestKeyRotatorInit
# ---------------------------------------------------------------------------


class TestKeyRotatorInit:
    """KeyRotator initialization and configuration."""

    def test_default_cooldown(self) -> None:
        rotator = KeyRotator()
        assert rotator.cooldown_seconds == 60

    def test_custom_cooldown(self) -> None:
        rotator = KeyRotator(cooldown_seconds=120)
        assert rotator.cooldown_seconds == 120

    def test_empty_state(self) -> None:
        rotator = KeyRotator()
        assert rotator._keys == {}
        assert rotator._index == {}
        assert rotator._cooldowns == {}


# ---------------------------------------------------------------------------
# TestLoadKeys
# ---------------------------------------------------------------------------


class TestLoadKeys:
    """Key loading from environment variables."""

    def test_loads_multi_key_env_var(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "key1,key2,key3"}):
            keys = rotator._load_keys("anthropic")
        assert keys == ["key1", "key2", "key3"]

    def test_loads_single_key_fallback(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "single_key"}, clear=True):
            keys = rotator._load_keys("anthropic")
        assert keys == ["single_key"]

    def test_multi_key_takes_priority(self) -> None:
        rotator = KeyRotator()
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEYS": "multi1,multi2", "ANTHROPIC_API_KEY": "single"},
        ):
            keys = rotator._load_keys("anthropic")
        assert keys == ["multi1", "multi2"]

    def test_returns_empty_when_no_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {}, clear=True):
            keys = rotator._load_keys("anthropic")
        assert keys == []

    def test_strips_whitespace(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"OPENAI_API_KEYS": " key1 , key2 , key3 "}):
            keys = rotator._load_keys("openai")
        assert keys == ["key1", "key2", "key3"]

    def test_skips_empty_entries(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "key1,,key2,"}):
            keys = rotator._load_keys("anthropic")
        assert keys == ["key1", "key2"]

    def test_caches_loaded_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "key1"}):
            rotator._load_keys("anthropic")

        # Even without env var, cached keys are returned
        with patch.dict("os.environ", {}, clear=True):
            keys = rotator._load_keys("anthropic")
        assert keys == ["key1"]

    def test_unknown_provider_returns_empty(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {}, clear=True):
            keys = rotator._load_keys("unknown_provider")
        assert keys == []

    def test_loads_google_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"GOOGLE_API_KEYS": "gk1,gk2"}):
            keys = rotator._load_keys("google")
        assert keys == ["gk1", "gk2"]


# ---------------------------------------------------------------------------
# TestGetKey
# ---------------------------------------------------------------------------


class TestGetKey:
    """Round-robin key selection with cooldown."""

    def test_returns_none_when_no_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {}, clear=True):
            assert rotator.get_key("anthropic") is None

    def test_round_robin_rotation(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2,k3"}):
            first = rotator.get_key("anthropic")
            second = rotator.get_key("anthropic")
            third = rotator.get_key("anthropic")
            fourth = rotator.get_key("anthropic")

        assert first == "k1"
        assert second == "k2"
        assert third == "k3"
        assert fourth == "k1"  # wraps around

    def test_single_key_always_returns_same(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "only_key"}, clear=True):
            for _ in range(5):
                assert rotator.get_key("anthropic") == "only_key"

    def test_skips_cooled_down_key(self) -> None:
        rotator = KeyRotator(cooldown_seconds=300)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2"}):
            rotator._load_keys("anthropic")
            rotator.mark_rate_limited("anthropic", "k1")

            # Should skip k1 and return k2
            key = rotator.get_key("anthropic")
        assert key == "k2"

    def test_returns_none_when_all_in_cooldown(self) -> None:
        rotator = KeyRotator(cooldown_seconds=300)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2"}):
            rotator._load_keys("anthropic")
            rotator.mark_rate_limited("anthropic", "k1")
            rotator.mark_rate_limited("anthropic", "k2")

            key = rotator.get_key("anthropic")
        assert key is None

    def test_key_available_after_cooldown_expires(self) -> None:
        rotator = KeyRotator(cooldown_seconds=0.01)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1"}):
            rotator._load_keys("anthropic")
            rotator.mark_rate_limited("anthropic", "k1")

            time.sleep(0.02)
            key = rotator.get_key("anthropic")
        assert key == "k1"


# ---------------------------------------------------------------------------
# TestMarkRateLimited
# ---------------------------------------------------------------------------


class TestMarkRateLimited:
    """Rate limit marking and cooldown."""

    def test_marks_key_cooldown(self) -> None:
        rotator = KeyRotator(cooldown_seconds=60)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2"}):
            rotator._load_keys("anthropic")
            rotator.mark_rate_limited("anthropic", "k1")

        assert "anthropic:0" in rotator._cooldowns

    def test_ignores_unknown_key(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1"}):
            rotator._load_keys("anthropic")
            # Should not raise
            rotator.mark_rate_limited("anthropic", "unknown_key")

        assert len(rotator._cooldowns) == 0

    def test_cooldown_uses_correct_duration(self) -> None:
        rotator = KeyRotator(cooldown_seconds=120)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1"}):
            rotator._load_keys("anthropic")
            before = time.monotonic()
            rotator.mark_rate_limited("anthropic", "k1")

        cooldown_until = rotator._cooldowns["anthropic:0"]
        assert cooldown_until >= before + 119
        assert cooldown_until <= before + 121


# ---------------------------------------------------------------------------
# TestGetLitellmKwargs
# ---------------------------------------------------------------------------


class TestGetLitellmKwargs:
    """get_litellm_kwargs returns api_key for litellm calls."""

    def test_returns_api_key(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2"}):
            kwargs = rotator.get_litellm_kwargs("anthropic")
        assert kwargs == {"api_key": "k1"}

    def test_returns_empty_when_no_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {}, clear=True):
            kwargs = rotator.get_litellm_kwargs("anthropic")
        assert kwargs == {}

    def test_rotates_on_successive_calls(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"OPENAI_API_KEYS": "ok1,ok2"}):
            first = rotator.get_litellm_kwargs("openai")
            second = rotator.get_litellm_kwargs("openai")
        assert first == {"api_key": "ok1"}
        assert second == {"api_key": "ok2"}


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------


class TestStats:
    """Key pool statistics."""

    def test_stats_empty(self) -> None:
        rotator = KeyRotator()
        assert rotator.stats == {}

    def test_stats_with_loaded_keys(self) -> None:
        rotator = KeyRotator()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2,k3"}):
            rotator._load_keys("anthropic")
            stats = rotator.stats
        assert stats == {"anthropic": {"total": 3, "available": 3}}

    def test_stats_with_cooldowns(self) -> None:
        rotator = KeyRotator(cooldown_seconds=300)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEYS": "k1,k2,k3"}):
            rotator._load_keys("anthropic")
            rotator.mark_rate_limited("anthropic", "k1")
            stats = rotator.stats
        assert stats == {"anthropic": {"total": 3, "available": 2}}

    def test_stats_multiple_providers(self) -> None:
        rotator = KeyRotator()
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEYS": "ak1", "OPENAI_API_KEYS": "ok1,ok2"},
        ):
            rotator._load_keys("anthropic")
            rotator._load_keys("openai")
            stats = rotator.stats
        assert stats == {
            "anthropic": {"total": 1, "available": 1},
            "openai": {"total": 2, "available": 2},
        }
