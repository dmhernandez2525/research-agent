"""Project registry for AgentPrompts integrations."""

from __future__ import annotations

import json
from pathlib import Path


class ProjectRegistry:
    """Persist and resolve project-name to path mappings."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, project_path: Path) -> None:
        payload = self._load()
        payload[self._normalize(name)] = str(project_path.resolve())
        self._save(payload)

    def resolve(self, name: str) -> Path | None:
        payload = self._load()
        raw = payload.get(self._normalize(name))
        if raw is None:
            return None
        return Path(raw)

    def list_projects(self) -> list[tuple[str, Path]]:
        payload = self._load()
        items = sorted(payload.items(), key=lambda item: item[0])
        return [(name, Path(path)) for name, path in items]

    def discover_and_register(self, projects_dir: Path) -> list[tuple[str, Path]]:
        """Auto-discover projects that contain RESEARCH_PROMPT.md."""
        discovered: list[tuple[str, Path]] = []
        if not projects_dir.exists():
            return discovered

        for item in sorted(projects_dir.iterdir()):
            if not item.is_dir():
                continue
            prompt_path = item / "RESEARCH_PROMPT.md"
            if not prompt_path.exists():
                continue
            name = item.name
            self.register(name, item)
            discovered.append((name, item))
        return discovered

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            parsed = {str(key): str(value) for key, value in payload.items()}
            return parsed
        return {}

    def _save(self, payload: dict[str, str]) -> None:
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _normalize(self, name: str) -> str:
        return name.strip().lower()
