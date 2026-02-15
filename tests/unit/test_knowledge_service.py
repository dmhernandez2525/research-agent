"""Unit tests for Phase 22 knowledge service orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from research_agent.knowledge.models import KnowledgeExportPayload, KnowledgeFinding
from research_agent.knowledge.service import KnowledgeService
from research_agent.knowledge.store import KnowledgeStore

if TYPE_CHECKING:
    from pathlib import Path


def _store_with_seed(path: Path) -> KnowledgeStore:
    store = KnowledgeStore(path)
    payload = KnowledgeExportPayload(
        findings=[
            KnowledgeFinding(
                id="k1",
                topic="AI news",
                statement="OpenAI is recommended for CUDA workflows.",
                sources=["https://example.com/a"],
                confidence=0.35,
                cluster="ai",
                updated_at=(datetime.now(tz=UTC) - timedelta(days=90)).isoformat(),
            ),
            KnowledgeFinding(
                id="k2",
                topic="AI news",
                statement="OpenAI not recommended for CUDA workflows.",
                sources=["https://example.com/b", "https://example.com/c"],
                confidence=0.75,
                cluster="ai",
                updated_at=datetime.now(tz=UTC).isoformat(),
            ),
        ]
    )
    store.save(payload)
    return store


def test_summarize_conflicts_and_due_refresh(tmp_path: Path) -> None:
    store = _store_with_seed(tmp_path / "knowledge.json")
    service = KnowledgeService(store)

    service.rebuild_relationships("AI news")
    summary = service.summarize(topic="AI news", threshold=0.7, refresh_days=7)

    assert summary.findings
    assert summary.conflicts
    assert summary.due_for_refresh_ids
    assert "ai" in summary.cluster_summaries


def test_refresh_topic_updates_history_and_json_graph(tmp_path: Path) -> None:
    store = _store_with_seed(tmp_path / "knowledge.json")
    service = KnowledgeService(store)
    service.rebuild_relationships("AI news")

    refreshed = service.refresh_topic(
        topic="AI news",
        threshold=0.8,
        refresh_days=1,
        new_statement="Updated statement after refresh cycle.",
    )
    assert refreshed >= 1

    payload = store.load()
    assert payload.refresh_history
    assert any("Updated statement" in item.statement for item in payload.findings)

    graph = service.to_json_graph(topic="AI news")
    assert "nodes" in graph
    assert "edges" in graph
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
