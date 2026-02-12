"""Prompt caching optimization for Anthropic API calls.

Provides cache-stable message ordering, deterministic serialization,
cache_control markers, and hit rate tracking to maximize Anthropic's
prompt caching benefits.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic cache_control marker for cacheable content
_CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# Pricing: cached reads cost 90% less than uncached
_CACHE_WRITE_COST_MULTIPLIER = 1.25  # 25% surcharge on first write
_CACHE_READ_COST_MULTIPLIER = 0.10  # 90% discount on cache hits


# ---------------------------------------------------------------------------
# Deterministic serialization
# ---------------------------------------------------------------------------


def deterministic_json(obj: Any) -> str:
    """Serialize an object to JSON with sorted keys and no extra whitespace.

    Ensures identical Python dicts always produce the same JSON string,
    which is critical for cache key stability.

    Args:
        obj: The object to serialize.

    Returns:
        A deterministic JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Message ordering
# ---------------------------------------------------------------------------


def order_messages_for_cache(
    system_prompt: str,
    tool_definitions: list[dict[str, Any]] | None = None,
    conversation: list[dict[str, Any]] | None = None,
    latest_message: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a cache-stable message payload for Anthropic API calls.

    Orders content as: tools -> system -> conversation -> latest.
    This ensures that stable content (tools, system prompt) is at the
    front, maximizing cache prefix hits even as conversation grows.

    Args:
        system_prompt: The system prompt text.
        tool_definitions: Optional tool schemas (sorted deterministically).
        conversation: Prior conversation messages (append-only).
        latest_message: The newest user message to append.

    Returns:
        A dict with ``system``, ``tools``, and ``messages`` keys
        ready for the Anthropic API.
    """
    # Build system block with cache_control marker
    system_block = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": _CACHE_CONTROL_EPHEMERAL,
        }
    ]

    # Sort tool definitions deterministically
    tools: list[dict[str, Any]] = []
    if tool_definitions:
        for tool in tool_definitions:
            serialized = deterministic_json(tool)
            stable_tool = json.loads(serialized)
            stable_tool["cache_control"] = _CACHE_CONTROL_EPHEMERAL
            tools.append(stable_tool)

    # Build messages list: conversation history + latest
    messages: list[dict[str, Any]] = []
    if conversation:
        messages.extend(conversation)
    if latest_message:
        messages.append(latest_message)

    return {
        "system": system_block,
        "tools": tools,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Cache hit tracking
# ---------------------------------------------------------------------------


class CacheTracker:
    """Tracks prompt cache hit/miss statistics for a session.

    Records each API call's cache status and computes aggregate
    hit rate and estimated savings.

    Attributes:
        total_calls: Total number of tracked calls.
        cache_hits: Number of calls that hit the cache.
        cache_misses: Number of calls that missed the cache.
    """

    def __init__(self) -> None:
        self.total_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._total_input_tokens: int = 0
        self._cached_input_tokens: int = 0

    def record_call(
        self,
        input_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        """Record a single API call's cache statistics.

        Args:
            input_tokens: Total input tokens for the call.
            cached_tokens: Number of input tokens served from cache.
        """
        self.total_calls += 1
        self._total_input_tokens += input_tokens
        self._cached_input_tokens += cached_tokens

        if cached_tokens > 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        logger.debug(
            "cache_call_recorded",
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            hit=cached_tokens > 0,
        )

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate as a fraction.

        Returns:
            Hit rate between 0.0 and 1.0 (0.0 if no calls recorded).
        """
        if self.total_calls == 0:
            return 0.0
        return self.cache_hits / self.total_calls

    def estimated_savings(self, input_cost_per_million: float) -> float:
        """Estimate cost savings from cache hits.

        Savings = cached_tokens * cost_per_token * (1 - read_multiplier)

        Args:
            input_cost_per_million: Cost per million input tokens
                for the model being used.

        Returns:
            Estimated savings in USD.
        """
        if self._cached_input_tokens == 0:
            return 0.0

        cost_per_token = input_cost_per_million / 1_000_000
        uncached_cost = self._cached_input_tokens * cost_per_token
        cached_cost = (
            self._cached_input_tokens * cost_per_token * _CACHE_READ_COST_MULTIPLIER
        )
        return uncached_cost - cached_cost

    def summary(self) -> dict[str, Any]:
        """Return a summary of cache statistics.

        Returns:
            Dict with hit_rate, total_calls, cache_hits, cache_misses,
            total_input_tokens, and cached_input_tokens.
        """
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 3),
            "total_input_tokens": self._total_input_tokens,
            "cached_input_tokens": self._cached_input_tokens,
        }
