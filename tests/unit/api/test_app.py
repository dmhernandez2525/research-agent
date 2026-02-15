"""Tests for FastAPI app endpoints, auth, and streaming."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from research_agent.api.app import create_app
from research_agent.config import Settings

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path, *, rate_limit: int = 60) -> Settings:
    settings = Settings()
    settings.api.api_key_file = tmp_path / "keys.json"
    settings.api.frontend_dist_dir = tmp_path / "frontend-dist"
    settings.api.rate_limit_per_minute = rate_limit
    settings.api.max_concurrent_sessions = 2
    settings.api.queue_limit = 10
    settings.checkpoints.directory = tmp_path / "checkpoints"
    settings.report.output_dir = tmp_path / "reports"
    settings.api.cors_origins = ["http://localhost:3000"]
    return settings


def test_openapi_and_cors_headers(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200

        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code == 200
        assert (
            resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
        )


def test_auth_required_for_api_routes(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/sessions")
        assert resp.status_code == 401


def test_session_crud_lifecycle(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    admin_key = app.state.api_key_store.list_keys()[0].key

    with TestClient(app) as client:
        created = client.post(
            "/api/sessions",
            json={"query": "demo"},
            headers={"X-API-Key": admin_key},
        )
        assert created.status_code == 200
        session_id = created.json()["id"]

        listed = client.get("/api/sessions", headers={"X-API-Key": admin_key})
        assert listed.status_code == 200
        assert any(item["id"] == session_id for item in listed.json()["sessions"])

        for _ in range(40):
            current = client.get(
                f"/api/sessions/{session_id}",
                headers={"X-API-Key": admin_key},
            )
            assert current.status_code == 200
            if current.json()["status"] == "COMPLETED":
                break
            time.sleep(0.01)

        report = client.get(
            f"/api/sessions/{session_id}/report",
            headers={"X-API-Key": admin_key},
        )
        assert report.status_code == 200
        assert "API Research Report" in report.text

        deleted = client.delete(
            f"/api/sessions/{session_id}",
            headers={"X-API-Key": admin_key},
        )
        assert deleted.status_code == 200


def test_rate_limit_for_non_admin_keys(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, rate_limit=1))
    admin = app.state.api_key_store.list_keys()[0]
    user = app.state.api_key_store.create_key("user", admin=False)

    with TestClient(app) as client:
        first = client.get("/api/sessions", headers={"X-API-Key": user.key})
        assert first.status_code == 200
        assert first.headers.get("X-RateLimit-Limit") == "1"
        assert first.headers.get("X-RateLimit-Remaining") == "0"

        second = client.get("/api/sessions", headers={"X-API-Key": user.key})
        assert second.status_code == 429

        admin_req = client.get("/api/sessions", headers={"X-API-Key": admin.key})
        assert admin_req.status_code == 200


def test_sse_last_event_id_and_websocket_streaming(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    admin_key = app.state.api_key_store.list_keys()[0].key
    bus = app.state.event_bus

    first = bus.publish("sess-stream", "plan_ready", {"step": "plan"})
    second = bus.publish("sess-stream", "search_complete", {"step": "search"})

    with TestClient(app) as client:
        with client.stream(
            "GET",
            "/api/sessions/sess-stream/events",
            headers={"X-API-Key": admin_key, "Last-Event-ID": str(first.id)},
        ) as response:
            assert response.status_code == 200
            found = False
            for line in response.iter_lines():
                if f"id: {second.id}" in line:
                    found = True
                    break
            assert found is True

        with client.websocket_connect(
            "/ws/sessions/sess-stream",
            headers={"x-api-key": admin_key},
        ) as websocket:
            bus.publish("sess-stream", "summary_ready", {"step": "summarize"})
            for _ in range(5):
                payload = websocket.receive_json()
                if payload["event_type"] == "summary_ready":
                    break
            assert payload["event_type"] == "summary_ready"

        with client.websocket_connect(
            f"/ws/sessions/sess-stream?api_key={admin_key}",
        ) as websocket:
            bus.publish("sess-stream", "summary_ready", {"step": "summarize"})
            for _ in range(5):
                payload = websocket.receive_json()
                if payload["event_type"] == "summary_ready":
                    break
            assert payload["event_type"] == "summary_ready"


def test_frontend_static_serving_and_spa_fallback(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    dist = settings.api.frontend_dist_dir
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<html><body>frontend</body></html>", encoding="utf-8")
    (dist / "app.js").write_text("console.log('ok')", encoding="utf-8")

    app = create_app(settings)
    admin_key = app.state.api_key_store.list_keys()[0].key

    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "frontend" in root.text

        asset = client.get("/app.js")
        assert asset.status_code == 200
        assert "console.log('ok')" in asset.text

        spa = client.get("/history")
        assert spa.status_code == 200
        assert "frontend" in spa.text

        api_route = client.get("/api/sessions", headers={"X-API-Key": admin_key})
        assert api_route.status_code == 200
