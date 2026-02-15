"""FastAPI application for research-agent sessions and streaming."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from research_agent.api.auth import (
    APIKeyAuthError,
    APIKeyStore,
    RateLimiter,
    require_valid_key,
)
from research_agent.api.events import EventBus
from research_agent.api.manager import QueueOverflowError, SessionManager
from research_agent.api.models import (
    APIKeyCreateResponse,
    APIKeyRecord,
    SessionCreateRequest,
    SessionListResponse,
    SessionRecord,
)
from research_agent.api.notifications import NotificationDispatcher
from research_agent.api.static import mount_frontend
from research_agent.config import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI server app."""
    app_settings = settings or Settings.load()

    app = FastAPI(title="research-agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key_store = APIKeyStore(Path(app_settings.api.api_key_file))
    if not api_key_store.list_keys():
        api_key_store.create_key("default-admin", admin=True)

    event_bus = EventBus(app_settings.checkpoints.directory / "api-events")
    notifications = NotificationDispatcher(app_settings.api)
    session_manager = SessionManager(
        app_settings, event_bus, api_key_store, notifications
    )
    rate_limiter = RateLimiter()

    app.state.settings = app_settings
    app.state.api_key_store = api_key_store
    app.state.rate_limiter = rate_limiter
    app.state.event_bus = event_bus
    app.state.session_manager = session_manager

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await session_manager.shutdown()

    async def auth_key(
        request: Request,
        response: Response,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> APIKeyRecord:
        try:
            key_record = require_valid_key(x_api_key, api_key_store)
        except APIKeyAuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        allowed = True
        remaining = app_settings.api.rate_limit_per_minute
        reset = 0
        if not key_record.admin:
            allowed, remaining, reset = rate_limiter.check(
                key_record.id,
                app_settings.api.rate_limit_per_minute,
            )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(app_settings.api.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                },
            )

        request.state.rate_headers = {
            "X-RateLimit-Limit": str(app_settings.api.rate_limit_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset),
        }
        for header_name, value in request.state.rate_headers.items():
            response.headers[header_name] = value
        api_key_store.update_usage(key_record.id, requests=1)
        return key_record

    def require_admin(key_record: APIKeyRecord) -> None:
        if not key_record.admin:
            raise HTTPException(status_code=403, detail="Admin API key required")

    auth_dep = Depends(auth_key)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/sessions", response_model=SessionRecord)
    async def create_session(
        payload: SessionCreateRequest,
        request: Request,
        key_record: APIKeyRecord = auth_dep,
    ) -> SessionRecord:
        try:
            session = await session_manager.start_session(
                payload, api_key_id=key_record.id
            )
            return session
        except QueueOverflowError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/api/sessions", response_model=SessionListResponse)
    async def list_sessions(
        request: Request,
        key_record: APIKeyRecord = auth_dep,
    ) -> SessionListResponse:
        _ = key_record
        return SessionListResponse(sessions=session_manager.list_sessions())

    @app.get("/api/sessions/{session_id}", response_model=SessionRecord)
    async def get_session(
        session_id: str,
        request: Request,
        key_record: APIKeyRecord = auth_dep,
    ) -> SessionRecord:
        _ = key_record
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.get("/api/sessions/{session_id}/report")
    async def get_report(
        session_id: str,
        format: str = "md",
        request: Request | None = None,
        key_record: APIKeyRecord = auth_dep,
    ) -> FileResponse | PlainTextResponse:
        _ = key_record
        session = session_manager.get_session(session_id)
        if session is None or session.report_path is None:
            raise HTTPException(status_code=404, detail="Report not found")
        path = Path(session.report_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report file missing")

        if format == "pdf":
            pdf_path = path.with_suffix(".pdf")
            if not pdf_path.exists():
                raise HTTPException(status_code=404, detail="PDF report not available")
            return FileResponse(pdf_path)

        return PlainTextResponse(
            path.read_text(encoding="utf-8"), media_type="text/markdown"
        )

    @app.delete("/api/sessions/{session_id}")
    async def cancel_or_delete_session(
        session_id: str,
        request: Request,
        key_record: APIKeyRecord = auth_dep,
    ) -> dict[str, str]:
        _ = key_record
        cancelled = await session_manager.cancel_session(session_id)
        if cancelled:
            return {"status": "cancelled"}

        deleted = await session_manager.delete_session(session_id)
        if deleted:
            return {"status": "deleted"}

        raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/api/sessions/{session_id}/events")
    async def sse_events(
        session_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        key_record: APIKeyRecord = auth_dep,
    ) -> StreamingResponse:
        _ = key_record

        parsed_last = (
            int(last_event_id) if last_event_id and last_event_id.isdigit() else None
        )

        async def event_gen() -> AsyncIterator[str]:
            queue = event_bus.subscribe(session_id, parsed_last)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    event = await queue.get()
                    yield (
                        f"id: {event.id}\n"
                        f"event: {event.event_type}\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )
            finally:
                event_bus.unsubscribe(session_id, queue)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.websocket("/ws/sessions/{session_id}")
    async def ws_events(websocket: WebSocket, session_id: str) -> None:
        key_value = websocket.headers.get("x-api-key") or websocket.query_params.get(
            "api_key"
        )
        try:
            require_valid_key(key_value, api_key_store)
        except APIKeyAuthError:
            await websocket.close(code=4401)
            return

        await websocket.accept()

        queue = event_bus.subscribe(session_id)

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(15)
                await websocket.send_json({"event_type": "ping"})

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.model_dump())
        except Exception:
            pass
        finally:
            heartbeat_task.cancel()
            with suppress(Exception):
                await heartbeat_task
            event_bus.unsubscribe(session_id, queue)

    @app.post("/api/keys", response_model=APIKeyCreateResponse)
    async def create_api_key(
        name: str,
        admin: bool = False,
        key_record: APIKeyRecord = auth_dep,
    ) -> APIKeyCreateResponse:
        require_admin(key_record)
        created = api_key_store.create_key(name=name, admin=admin)
        return APIKeyCreateResponse(id=created.id, key=created.key, admin=created.admin)

    @app.get("/api/keys", response_model=list[APIKeyRecord])
    async def list_api_keys(
        key_record: APIKeyRecord = auth_dep,
    ) -> list[APIKeyRecord]:
        require_admin(key_record)
        return api_key_store.list_keys()

    @app.delete("/api/keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        key_record: APIKeyRecord = auth_dep,
    ) -> dict[str, str]:
        require_admin(key_record)
        if not api_key_store.revoke_key(key_id):
            raise HTTPException(status_code=404, detail="Key not found")
        return {"status": "revoked"}

    mount_frontend(app, Path(app_settings.api.frontend_dist_dir))

    return app
