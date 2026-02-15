"""Frontend static build serving helpers for FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

if TYPE_CHECKING:
    from pathlib import Path
from fastapi.staticfiles import StaticFiles

_RESERVED_PREFIXES = ("api", "docs", "openapi.json", "ws", "health")


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Mount built frontend files and SPA fallback routes when available."""
    index_path = dist_dir / "index.html"
    if not index_path.exists():
        return

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    async def frontend_index() -> FileResponse:
        return FileResponse(index_path)

    async def frontend_routes(path: str) -> FileResponse:
        if path in _RESERVED_PREFIXES or path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = dist_dir / path
        if candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(index_path)

    app.add_api_route("/", frontend_index, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/{path:path}",
        frontend_routes,
        methods=["GET"],
        include_in_schema=False,
    )
