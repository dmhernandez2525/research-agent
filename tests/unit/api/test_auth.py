"""Tests for API auth and rate limiting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from research_agent.api.auth import (
    APIKeyAuthError,
    APIKeyStore,
    RateLimiter,
    require_valid_key,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_api_key_store_create_verify_and_revoke(tmp_path: Path) -> None:
    store = APIKeyStore(tmp_path / "keys.json")

    created = store.create_key("test", admin=True)
    assert created.admin is True

    verified = store.verify(created.key)
    assert verified is not None
    assert verified.id == created.id

    assert store.revoke_key(created.id) is True
    assert store.verify(created.key) is None


def test_rate_limiter_enforces_limit() -> None:
    limiter = RateLimiter()

    allowed_1, remaining_1, _reset_1 = limiter.check("k1", limit_per_minute=1)
    allowed_2, remaining_2, _reset_2 = limiter.check("k1", limit_per_minute=1)

    assert allowed_1 is True
    assert remaining_1 == 0
    assert allowed_2 is False
    assert remaining_2 == 0


def test_require_valid_key_errors(tmp_path: Path) -> None:
    store = APIKeyStore(tmp_path / "keys.json")

    with pytest.raises(APIKeyAuthError):
        require_valid_key(None, store)

    with pytest.raises(APIKeyAuthError):
        require_valid_key("invalid", store)
