"""Persistence for enhancement incremental knowledge."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from research_agent.enhance_models import KnowledgeEntry, OpportunityCategory

if TYPE_CHECKING:
    from pathlib import Path


class KnowledgeStore:
    """JSON-backed enhancement knowledge store."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, dict[str, object]]]:
        if not self._store_path.exists():
            return {"projects": {}}
        payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        return cast("dict[str, dict[str, dict[str, object]]]", payload)

    def _save(self, payload: dict[str, dict[str, dict[str, object]]]) -> None:
        self._store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_project_entries(self, project_id: str) -> dict[str, KnowledgeEntry]:
        raw = self._load().get("projects", {}).get(project_id, {})
        if not isinstance(raw, dict):
            return {}
        result: dict[str, KnowledgeEntry] = {}
        for topic, value in raw.items():
            if isinstance(value, dict):
                result[topic] = KnowledgeEntry.model_validate(value)
        return result

    def upsert_entries(self, project_id: str, entries: list[KnowledgeEntry]) -> None:
        payload = self._load()
        projects = payload.setdefault("projects", {})
        project_entries = projects.setdefault(project_id, {})
        for entry in entries:
            project_entries[entry.topic] = entry.model_dump()
        self._save(payload)

    def cross_project_matches(
        self,
        project_id: str,
        category: OpportunityCategory,
    ) -> list[KnowledgeEntry]:
        payload = self._load().get("projects", {})
        if not isinstance(payload, dict):
            return []

        matches: list[KnowledgeEntry] = []
        for other_id, topic_map in payload.items():
            if other_id == project_id or not isinstance(topic_map, dict):
                continue
            for value in topic_map.values():
                if not isinstance(value, dict):
                    continue
                entry = KnowledgeEntry.model_validate(value)
                if entry.category == category:
                    matches.append(entry)
        return matches
