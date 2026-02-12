"""Round-robin API key rotation with cooldown on rate limits.

Provides the ``KeyRotator`` class for distributing LLM API calls across
multiple keys per provider, with automatic cooldown when a key hits a
rate limit (HTTP 429).

Configure keys via comma-separated environment variables:
``ANTHROPIC_API_KEYS=key1,key2,key3``
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_COOLDOWN_SECONDS = 60

# Environment variable names for multi-key configuration
_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEYS",
    "openai": "OPENAI_API_KEYS",
    "google": "GOOGLE_API_KEYS",
}

# Fallback single-key env vars
_SINGLE_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class KeyRotator:
    """Distributes API calls across multiple keys per provider.

    Uses round-robin selection with automatic cooldown when a key
    encounters a rate limit. Keys in cooldown are skipped until the
    cooldown period expires.

    Attributes:
        cooldown_seconds: Duration to skip a key after rate limiting.
    """

    def __init__(self, cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS) -> None:
        """Initialize the key rotator.

        Args:
            cooldown_seconds: How long to skip a rate-limited key.
        """
        self.cooldown_seconds = cooldown_seconds
        self._keys: dict[str, list[str]] = {}
        self._index: dict[str, int] = {}
        self._cooldowns: dict[str, float] = {}  # key -> cooldown_until timestamp

    def _load_keys(self, provider: str) -> list[str]:
        """Load API keys for a provider from environment variables.

        Checks for multi-key env var first (e.g., ANTHROPIC_API_KEYS),
        then falls back to the single-key env var (e.g., ANTHROPIC_API_KEY).

        Args:
            provider: The provider name (anthropic, openai, google).

        Returns:
            List of API keys for the provider.
        """
        if provider in self._keys:
            return self._keys[provider]

        # Try multi-key env var first
        multi_var = _KEY_ENV_VARS.get(provider, "")
        if multi_var:
            raw = os.environ.get(multi_var, "")
            if raw.strip():
                keys = [k.strip() for k in raw.split(",") if k.strip()]
                if keys:
                    self._keys[provider] = keys
                    self._index[provider] = 0
                    logger.info(
                        "keys_loaded",
                        provider=provider,
                        count=len(keys),
                        source=multi_var,
                    )
                    return keys

        # Fall back to single-key env var
        single_var = _SINGLE_KEY_ENV_VARS.get(provider, "")
        if single_var:
            key = os.environ.get(single_var, "").strip()
            if key:
                self._keys[provider] = [key]
                self._index[provider] = 0
                logger.debug(
                    "single_key_loaded",
                    provider=provider,
                    source=single_var,
                )
                return [key]

        self._keys[provider] = []
        return []

    def get_key(self, provider: str) -> str | None:
        """Get the next available API key for a provider.

        Rotates through keys in round-robin order, skipping any key
        currently in cooldown. Returns None if no keys are available.

        Args:
            provider: The provider name.

        Returns:
            An API key string, or None if no keys are configured/available.
        """
        keys = self._load_keys(provider)
        if not keys:
            return None

        now = time.monotonic()
        attempts = len(keys)

        for _ in range(attempts):
            idx = self._index[provider] % len(keys)
            key = keys[idx]
            self._index[provider] = idx + 1

            cooldown_key = f"{provider}:{idx}"
            cooldown_until = self._cooldowns.get(cooldown_key, 0)
            if now >= cooldown_until:
                return key

            logger.debug(
                "key_in_cooldown",
                provider=provider,
                key_index=idx,
                remaining=round(cooldown_until - now, 1),
            )

        logger.warning(
            "all_keys_in_cooldown",
            provider=provider,
            count=len(keys),
        )
        return None

    def mark_rate_limited(self, provider: str, key: str) -> None:
        """Mark a key as rate-limited, placing it in cooldown.

        Args:
            provider: The provider name.
            key: The API key that was rate-limited.
        """
        keys = self._load_keys(provider)
        try:
            idx = keys.index(key)
        except ValueError:
            return

        cooldown_key = f"{provider}:{idx}"
        self._cooldowns[cooldown_key] = time.monotonic() + self.cooldown_seconds
        logger.info(
            "key_rate_limited",
            provider=provider,
            key_index=idx,
            cooldown_seconds=self.cooldown_seconds,
        )

    def get_litellm_kwargs(self, provider: str) -> dict[str, Any]:
        """Get litellm keyword arguments with the rotated API key.

        Returns an empty dict if no keys are available (litellm will
        fall back to its own env var resolution).

        Args:
            provider: The provider name.

        Returns:
            Dict with ``api_key`` if a key is available, empty dict otherwise.
        """
        key = self.get_key(provider)
        if key is None:
            return {}
        return {"api_key": key}

    @property
    def stats(self) -> dict[str, dict[str, int]]:
        """Return key pool statistics per provider.

        Returns:
            Dict of provider -> {"total": N, "available": M}.
        """
        now = time.monotonic()
        result: dict[str, dict[str, int]] = {}
        for provider, keys in self._keys.items():
            available = 0
            for idx in range(len(keys)):
                cooldown_key = f"{provider}:{idx}"
                if now >= self._cooldowns.get(cooldown_key, 0):
                    available += 1
            result[provider] = {"total": len(keys), "available": available}
        return result
