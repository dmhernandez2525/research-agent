"""API key auth and rate limiting for FastAPI endpoints."""

from __future__ import annotations

import json
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import TYPE_CHECKING, cast

from research_agent.api.models import APIKeyRecord

if TYPE_CHECKING:
    from pathlib import Path


class APIKeyAuthError(Exception):
    """Authentication error with HTTP-compatible metadata."""

    def __init__(self, detail: str, status_code: int = 401) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class APIKeyStore:
    """File-backed API key storage with generate/revoke/list operations."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._save([])

    def _load(self) -> list[APIKeyRecord]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        payload = cast("list[dict[str, object]]", raw)
        return [APIKeyRecord.model_validate(item) for item in payload]

    def _save(self, records: list[APIKeyRecord]) -> None:
        self._path.write_text(
            json.dumps([record.model_dump() for record in records], indent=2),
            encoding="utf-8",
        )

    def list_keys(self) -> list[APIKeyRecord]:
        return self._load()

    def create_key(self, name: str, admin: bool = False) -> APIKeyRecord:
        records = self._load()
        key_value = f"ra_{secrets.token_urlsafe(24)}"
        record = APIKeyRecord(
            id=f"key-{uuid.uuid4().hex[:10]}",
            key=key_value,
            name=name,
            admin=admin,
        )
        records.append(record)
        self._save(records)
        return record

    def revoke_key(self, key_id: str) -> bool:
        records = self._load()
        changed = False
        updated: list[APIKeyRecord] = []
        for record in records:
            if record.id == key_id:
                record.revoked = True
                changed = True
            updated.append(record)
        if changed:
            self._save(updated)
        return changed

    def verify(self, api_key: str) -> APIKeyRecord | None:
        for record in self._load():
            if record.key == api_key and not record.revoked:
                return record
        return None

    def update_usage(
        self,
        key_id: str,
        *,
        requests: int = 0,
        sessions_started: int = 0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        records = self._load()
        updated: list[APIKeyRecord] = []
        for record in records:
            if record.id == key_id:
                record.requests += requests
                record.sessions_started += sessions_started
                record.tokens_used += tokens_used
                record.cost_usd += cost_usd
            updated.append(record)
        self._save(updated)


class RateLimiter:
    """Simple in-memory per-key request rate limiter."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key_id: str, limit_per_minute: int) -> tuple[bool, int, int]:
        now = time.time()
        window_start = now - 60.0
        bucket = self._events[key_id]

        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= limit_per_minute:
            reset = int(bucket[0] + 60)
            return False, 0, reset

        bucket.append(now)
        remaining = max(limit_per_minute - len(bucket), 0)
        reset = int(now + 60)
        return True, remaining, reset


def require_valid_key(api_key: str | None, store: APIKeyStore) -> APIKeyRecord:
    """Validate API key header value and return key metadata."""
    if not api_key:
        raise APIKeyAuthError("Missing X-API-Key header", status_code=401)

    record = store.verify(api_key)
    if record is None:
        raise APIKeyAuthError("Invalid API key", status_code=401)

    return record
