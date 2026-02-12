"""Adaptive rate limiting with sliding window error tracking.

Dynamically adjusts backoff delays based on recent error rates per
provider. When a provider's error rate exceeds a threshold, the backoff
multiplier increases; when errors drop below a threshold, it decreases.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_WINDOW_SECONDS = 60.0
_DEFAULT_BASE_DELAY = 0.1  # 100ms base delay between requests
_DEFAULT_MAX_DELAY = 30.0  # 30s max backoff delay
_DEFAULT_ERROR_INCREASE_THRESHOLD = 0.30  # 30% error rate triggers increase
_DEFAULT_ERROR_DECREASE_THRESHOLD = 0.10  # 10% error rate allows decrease
_DEFAULT_MULTIPLIER_STEP = 1.5  # Factor to increase/decrease multiplier


class _RequestOutcome:
    """Record of a single request outcome."""

    __slots__ = ("success", "timestamp")

    def __init__(self, timestamp: float, success: bool) -> None:
        self.timestamp = timestamp
        self.success = success


class AdaptiveRateLimiter:
    """Rate limiter that adapts backoff based on recent error rates.

    Tracks a sliding window of request outcomes per provider. When the
    error rate exceeds ``error_increase_threshold``, the backoff multiplier
    increases. When it drops below ``error_decrease_threshold``, the
    multiplier decreases back toward 1.0.

    Attributes:
        window_seconds: Size of the sliding window in seconds.
        base_delay: Minimum delay between requests in seconds.
        max_delay: Maximum backoff delay in seconds.
        error_increase_threshold: Error rate that triggers multiplier increase.
        error_decrease_threshold: Error rate that allows multiplier decrease.
    """

    def __init__(
        self,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
        base_delay: float = _DEFAULT_BASE_DELAY,
        max_delay: float = _DEFAULT_MAX_DELAY,
        error_increase_threshold: float = _DEFAULT_ERROR_INCREASE_THRESHOLD,
        error_decrease_threshold: float = _DEFAULT_ERROR_DECREASE_THRESHOLD,
        multiplier_step: float = _DEFAULT_MULTIPLIER_STEP,
    ) -> None:
        self.window_seconds = window_seconds
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.error_increase_threshold = error_increase_threshold
        self.error_decrease_threshold = error_decrease_threshold
        self.multiplier_step = multiplier_step

        self._outcomes: dict[str, deque[_RequestOutcome]] = {}
        self._multipliers: dict[str, float] = {}

    def _get_outcomes(self, provider: str) -> deque[_RequestOutcome]:
        """Get or create the outcome deque for a provider."""
        if provider not in self._outcomes:
            self._outcomes[provider] = deque()
        return self._outcomes[provider]

    def _prune_window(self, provider: str) -> None:
        """Remove outcomes older than the sliding window."""
        outcomes = self._get_outcomes(provider)
        cutoff = time.monotonic() - self.window_seconds
        while outcomes and outcomes[0].timestamp < cutoff:
            outcomes.popleft()

    def record_outcome(self, provider: str, success: bool) -> None:
        """Record the outcome of a request and adjust the multiplier.

        Args:
            provider: The provider name (e.g. "anthropic", "openai").
            success: True if the request succeeded, False on error/429.
        """
        now = time.monotonic()
        outcomes = self._get_outcomes(provider)
        outcomes.append(_RequestOutcome(timestamp=now, success=success))

        self._prune_window(provider)
        self._adjust_multiplier(provider)

    def _adjust_multiplier(self, provider: str) -> None:
        """Adjust the backoff multiplier based on current error rate."""
        error_rate = self.error_rate(provider)
        current = self._multipliers.get(provider, 1.0)

        if error_rate > self.error_increase_threshold:
            new_multiplier = min(
                current * self.multiplier_step,
                self.max_delay / max(self.base_delay, 0.001),
            )
            if new_multiplier != current:
                self._multipliers[provider] = new_multiplier
                logger.info(
                    "rate_limit_multiplier_increased",
                    provider=provider,
                    error_rate=round(error_rate, 3),
                    multiplier=round(new_multiplier, 2),
                )
        elif error_rate < self.error_decrease_threshold and current > 1.0:
            new_multiplier = max(current / self.multiplier_step, 1.0)
            if new_multiplier != current:
                self._multipliers[provider] = new_multiplier
                logger.debug(
                    "rate_limit_multiplier_decreased",
                    provider=provider,
                    error_rate=round(error_rate, 3),
                    multiplier=round(new_multiplier, 2),
                )

    def error_rate(self, provider: str) -> float:
        """Calculate the current error rate for a provider.

        Args:
            provider: The provider name.

        Returns:
            Error rate between 0.0 and 1.0 (0.0 if no outcomes recorded).
        """
        self._prune_window(provider)
        outcomes = self._get_outcomes(provider)
        if not outcomes:
            return 0.0
        errors = sum(1 for o in outcomes if not o.success)
        return errors / len(outcomes)

    def current_delay(self, provider: str) -> float:
        """Calculate the current delay for a provider.

        Args:
            provider: The provider name.

        Returns:
            Delay in seconds (base_delay * multiplier, capped at max_delay).
        """
        multiplier = self._multipliers.get(provider, 1.0)
        return min(self.base_delay * multiplier, self.max_delay)

    def multiplier(self, provider: str) -> float:
        """Get the current backoff multiplier for a provider.

        Args:
            provider: The provider name.

        Returns:
            The current multiplier (1.0 = no backoff increase).
        """
        return self._multipliers.get(provider, 1.0)

    async def acquire(self, provider: str) -> None:
        """Wait for the current backoff delay before proceeding.

        Call this before making an API request to the given provider.
        The delay adapts based on recent error rates.

        Args:
            provider: The provider name.
        """
        delay = self.current_delay(provider)
        if delay > 0:
            await asyncio.sleep(delay)

    def stats(self, provider: str) -> dict[str, Any]:
        """Return statistics for a provider.

        Args:
            provider: The provider name.

        Returns:
            Dict with error_rate, multiplier, current_delay, and
            window_size (number of outcomes in the window).
        """
        self._prune_window(provider)
        outcomes = self._get_outcomes(provider)
        return {
            "error_rate": round(self.error_rate(provider), 3),
            "multiplier": round(self.multiplier(provider), 2),
            "current_delay": round(self.current_delay(provider), 4),
            "window_size": len(outcomes),
        }

    def reset(self, provider: str) -> None:
        """Reset a provider's outcomes and multiplier.

        Args:
            provider: The provider name.
        """
        self._outcomes.pop(provider, None)
        self._multipliers.pop(provider, None)
        logger.debug("rate_limiter_reset", provider=provider)

    def reset_all(self) -> None:
        """Reset all providers' outcomes and multipliers."""
        self._outcomes.clear()
        self._multipliers.clear()
        logger.debug("rate_limiter_reset_all")
