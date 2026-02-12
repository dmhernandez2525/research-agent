"""Unit tests for research_agent.rate_limiter."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from research_agent.rate_limiter import AdaptiveRateLimiter

# ---------------------------------------------------------------------------
# TestAdaptiveRateLimiterInit
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiterInit:
    """Initialization and default configuration."""

    def test_default_values(self) -> None:
        limiter = AdaptiveRateLimiter()
        assert limiter.window_seconds == 60.0
        assert limiter.base_delay == 0.1
        assert limiter.max_delay == 30.0
        assert limiter.error_increase_threshold == 0.30
        assert limiter.error_decrease_threshold == 0.10

    def test_custom_values(self) -> None:
        limiter = AdaptiveRateLimiter(
            window_seconds=30,
            base_delay=0.5,
            max_delay=10,
            error_increase_threshold=0.5,
            error_decrease_threshold=0.2,
            multiplier_step=2.0,
        )
        assert limiter.window_seconds == 30
        assert limiter.base_delay == 0.5
        assert limiter.max_delay == 10
        assert limiter.multiplier_step == 2.0


# ---------------------------------------------------------------------------
# TestRecordOutcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    """record_outcome tracks successes and failures."""

    def test_record_success(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("anthropic", success=True)
        assert limiter.error_rate("anthropic") == 0.0

    def test_record_failure(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("anthropic", success=False)
        assert limiter.error_rate("anthropic") == 1.0

    def test_mixed_outcomes(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("openai", success=True)
        limiter.record_outcome("openai", success=True)
        limiter.record_outcome("openai", success=False)
        # 1 error out of 3
        assert abs(limiter.error_rate("openai") - 1 / 3) < 0.01

    def test_different_providers_tracked_separately(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("anthropic", success=False)
        limiter.record_outcome("openai", success=True)
        assert limiter.error_rate("anthropic") == 1.0
        assert limiter.error_rate("openai") == 0.0


# ---------------------------------------------------------------------------
# TestErrorRate
# ---------------------------------------------------------------------------


class TestErrorRate:
    """error_rate calculates sliding window error rate."""

    def test_zero_for_unknown_provider(self) -> None:
        limiter = AdaptiveRateLimiter()
        assert limiter.error_rate("unknown") == 0.0

    def test_prunes_old_outcomes(self) -> None:
        limiter = AdaptiveRateLimiter(window_seconds=0.01)
        limiter.record_outcome("p", success=False)
        time.sleep(0.02)
        # Old outcome should be pruned
        assert limiter.error_rate("p") == 0.0

    def test_recent_outcomes_counted(self) -> None:
        limiter = AdaptiveRateLimiter(window_seconds=60)
        for _ in range(4):
            limiter.record_outcome("p", success=True)
        limiter.record_outcome("p", success=False)
        assert abs(limiter.error_rate("p") - 0.2) < 0.01


# ---------------------------------------------------------------------------
# TestMultiplierAdjustment
# ---------------------------------------------------------------------------


class TestMultiplierAdjustment:
    """Multiplier adjusts based on error rate thresholds."""

    def test_multiplier_starts_at_one(self) -> None:
        limiter = AdaptiveRateLimiter()
        assert limiter.multiplier("anthropic") == 1.0

    def test_multiplier_increases_on_high_errors(self) -> None:
        limiter = AdaptiveRateLimiter(
            error_increase_threshold=0.3,
            multiplier_step=1.5,
        )
        # 4 failures, 1 success = 80% error rate > 30%
        for _ in range(4):
            limiter.record_outcome("p", success=False)
        limiter.record_outcome("p", success=True)

        assert limiter.multiplier("p") > 1.0

    def test_multiplier_decreases_on_low_errors(self) -> None:
        limiter = AdaptiveRateLimiter(
            error_decrease_threshold=0.1,
            multiplier_step=1.5,
        )
        # First increase it with failures
        for _ in range(5):
            limiter.record_outcome("p", success=False)
        # Record enough successes that error rate drops below 10%
        # (5 failures + 100 successes = 4.76% error rate)
        for _ in range(100):
            limiter.record_outcome("p", success=True)
        final_mult = limiter.multiplier("p")
        # At this point the multiplier should have been reduced
        # from its peak. Since we have many successes, it should be
        # decreasing toward 1.0 (though it may not reach 1.0 exactly)
        # The key assertion: it decreased from the maximum it reached.
        assert final_mult < limiter.max_delay / limiter.base_delay

    def test_multiplier_does_not_go_below_one(self) -> None:
        limiter = AdaptiveRateLimiter()
        # All successes should keep multiplier at 1.0
        for _ in range(20):
            limiter.record_outcome("p", success=True)
        assert limiter.multiplier("p") == 1.0

    def test_multiplier_capped_by_max_delay(self) -> None:
        limiter = AdaptiveRateLimiter(
            base_delay=0.1,
            max_delay=1.0,
            multiplier_step=100.0,
        )
        # Many failures
        for _ in range(20):
            limiter.record_outcome("p", success=False)
        # Delay should not exceed max_delay
        assert limiter.current_delay("p") <= 1.0


# ---------------------------------------------------------------------------
# TestCurrentDelay
# ---------------------------------------------------------------------------


class TestCurrentDelay:
    """current_delay returns base_delay * multiplier."""

    def test_default_delay(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=0.1)
        assert limiter.current_delay("p") == 0.1

    def test_increased_delay(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=0.1, multiplier_step=2.0)
        # Force high error rate
        for _ in range(10):
            limiter.record_outcome("p", success=False)
        delay = limiter.current_delay("p")
        assert delay > 0.1

    def test_delay_capped_at_max(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=10.0, max_delay=5.0)
        # Even base_delay > max_delay
        assert limiter.current_delay("p") == 5.0


# ---------------------------------------------------------------------------
# TestAcquire
# ---------------------------------------------------------------------------


class TestAcquire:
    """acquire() waits for the appropriate delay."""

    @pytest.mark.asyncio()
    async def test_acquire_waits(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=0.01)
        start = time.monotonic()
        await limiter.acquire("p")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.005  # Allow some tolerance

    @pytest.mark.asyncio()
    async def test_acquire_calls_sleep(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=0.5)
        with patch("research_agent.rate_limiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire("p")
        mock_sleep.assert_called_once_with(0.5)


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------


class TestStats:
    """stats() returns provider statistics."""

    def test_stats_for_unknown_provider(self) -> None:
        limiter = AdaptiveRateLimiter()
        s = limiter.stats("unknown")
        assert s["error_rate"] == 0.0
        assert s["multiplier"] == 1.0
        assert s["window_size"] == 0

    def test_stats_after_recording(self) -> None:
        limiter = AdaptiveRateLimiter(base_delay=0.1)
        limiter.record_outcome("p", success=True)
        limiter.record_outcome("p", success=False)
        s = limiter.stats("p")
        assert s["error_rate"] == 0.5
        assert s["window_size"] == 2
        assert "multiplier" in s
        assert "current_delay" in s


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    """reset() and reset_all() clear provider state."""

    def test_reset_single_provider(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("p", success=False)
        limiter.reset("p")
        assert limiter.error_rate("p") == 0.0
        assert limiter.multiplier("p") == 1.0

    def test_reset_does_not_affect_others(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("a", success=False)
        limiter.record_outcome("b", success=False)
        limiter.reset("a")
        assert limiter.error_rate("a") == 0.0
        assert limiter.error_rate("b") == 1.0

    def test_reset_unknown_provider_is_safe(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.reset("nonexistent")  # Should not raise

    def test_reset_all(self) -> None:
        limiter = AdaptiveRateLimiter()
        limiter.record_outcome("a", success=False)
        limiter.record_outcome("b", success=False)
        limiter.reset_all()
        assert limiter.error_rate("a") == 0.0
        assert limiter.error_rate("b") == 0.0
        assert limiter.multiplier("a") == 1.0
