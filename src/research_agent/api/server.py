"""Uvicorn server runner for the research-agent API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.config import Settings


def run_server(settings: Settings) -> None:
    """Run uvicorn with settings-backed host/port values."""
    try:
        import uvicorn

        from research_agent.api.app import create_app
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI server dependencies are not installed. "
            "Install extras that include `fastapi` and `uvicorn`."
        ) from exc

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
    )
