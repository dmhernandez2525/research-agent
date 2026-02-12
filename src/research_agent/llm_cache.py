"""Disk-based LLM response cache for deduplicating API calls.

Caches LLM responses keyed by model, temperature, and message content to
avoid redundant API calls for identical prompts. Uses ``diskcache`` for
persistent storage with configurable TTL.

Only deterministic calls (temperature == 0.0) are cached by default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_CACHE_DIR = Path("./data/llm_cache")
_DEFAULT_TTL_SECONDS = 86400  # 24 hours
_CACHE_VERSION = "v1"


def _build_cache_key(
    model: str,
    temperature: float,
    messages: list[dict[str, Any]],
    extra: str = "",
) -> str:
    """Build a deterministic cache key from call parameters.

    The key is a SHA-256 hash of the model identifier, temperature,
    sorted message content, and an optional extra string (for prompt
    version hashes).

    Args:
        model: The litellm model identifier.
        temperature: The temperature parameter.
        messages: Chat messages list.
        extra: Optional extra string to include in the key (e.g. prompt hash).

    Returns:
        A hex SHA-256 digest string.
    """
    key_parts = {
        "version": _CACHE_VERSION,
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "extra": extra,
    }
    serialized = json.dumps(key_parts, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


class LLMCache:
    """Disk-backed cache for LLM API responses.

    Stores serialized responses in a diskcache.Cache directory with
    configurable TTL. Only caches deterministic calls (temperature == 0.0)
    by default.

    Attributes:
        cache_dir: Directory path for the cache store.
        ttl_seconds: Time-to-live for cache entries in seconds.
        max_temperature: Maximum temperature that allows caching.
    """

    def __init__(
        self,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_temperature: float = 0.0,
    ) -> None:
        """Initialize the LLM cache.

        Args:
            cache_dir: Directory for the diskcache store.
            ttl_seconds: TTL for cached entries in seconds.
            max_temperature: Calls with temperature above this are not cached.
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.max_temperature = max_temperature
        self._cache: Any = None

    def _get_cache(self) -> Any:
        """Lazy-initialize the diskcache.Cache instance.

        Returns:
            A diskcache.Cache instance.

        Raises:
            ImportError: If diskcache is not installed.
        """
        if self._cache is not None:
            return self._cache

        try:
            import diskcache
        except ImportError:
            logger.warning(
                "diskcache_not_installed",
                hint="Install with: pip install diskcache",
            )
            raise

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.cache_dir))
        return self._cache

    def get(
        self,
        model: str,
        temperature: float,
        messages: list[dict[str, Any]],
        extra: str = "",
    ) -> dict[str, Any] | None:
        """Look up a cached LLM response.

        Returns None if:
        - The temperature exceeds max_temperature (non-deterministic)
        - No matching cache entry exists
        - diskcache is not installed

        Args:
            model: The litellm model identifier.
            temperature: The temperature parameter.
            messages: Chat messages list.
            extra: Optional extra key component (e.g. prompt hash).

        Returns:
            Cached response dict, or None on miss.
        """
        if temperature > self.max_temperature:
            return None

        try:
            cache = self._get_cache()
        except ImportError:
            return None

        key = _build_cache_key(model, temperature, messages, extra)
        result = cache.get(key)

        if result is not None:
            logger.debug(
                "llm_cache_hit",
                model=model,
                key_prefix=key[:12],
            )
            return result  # type: ignore[no-any-return]

        logger.debug(
            "llm_cache_miss",
            model=model,
            key_prefix=key[:12],
        )
        return None

    def set(
        self,
        model: str,
        temperature: float,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        extra: str = "",
    ) -> bool:
        """Store an LLM response in the cache.

        Skips caching if temperature exceeds max_temperature or if
        diskcache is not installed.

        Args:
            model: The litellm model identifier.
            temperature: The temperature parameter.
            messages: Chat messages list.
            response: The LLM response dict to cache.
            extra: Optional extra key component (e.g. prompt hash).

        Returns:
            True if the response was cached, False otherwise.
        """
        if temperature > self.max_temperature:
            return False

        try:
            cache = self._get_cache()
        except ImportError:
            return False

        key = _build_cache_key(model, temperature, messages, extra)
        cache.set(key, response, expire=self.ttl_seconds)

        logger.debug(
            "llm_cache_set",
            model=model,
            key_prefix=key[:12],
            ttl_seconds=self.ttl_seconds,
        )
        return True

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries removed, or 0 if cache is unavailable.
        """
        try:
            cache = self._get_cache()
        except ImportError:
            return 0

        count = len(cache)
        cache.clear()
        logger.info("llm_cache_cleared", entries_removed=count)
        return count

    @property
    def size(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Entry count, or 0 if cache is unavailable.
        """
        try:
            cache = self._get_cache()
        except ImportError:
            return 0
        return len(cache)

    def close(self) -> None:
        """Close the underlying diskcache store."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None
