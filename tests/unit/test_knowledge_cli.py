"""Tests for Phase 22 knowledge CLI workflows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Settings
from research_agent.knowledge.models import KnowledgeExportPayload, KnowledgeFinding
from research_agent.knowledge.store import KnowledgeStore

runner = CliRunner()


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.vector_store.persist_directory = tmp_path / "vector"
    return settings


def _seed_knowledge(path: Path) -> None:
    store = KnowledgeStore(path)
    payload = KnowledgeExportPayload(
        findings=[
            KnowledgeFinding(
                id="finding-1",
                topic="AI news",
                statement="OpenAI depends on CUDA tooling for some workloads.",
                sources=["https://example.com/a"],
                confidence=0.25,
                cluster="ai",
                updated_at=(datetime.now(tz=UTC) - timedelta(days=120)).isoformat(),
            ),
            KnowledgeFinding(
                id="finding-2",
                topic="AI news",
                statement="OpenAI contradicts prior guidance for model deployment.",
                sources=["https://example.com/b", "https://example.com/c"],
                confidence=0.70,
                cluster="ai",
                updated_at=datetime.now(tz=UTC).isoformat(),
            ),
        ]
    )
    store.save(payload)


def test_knowledge_summarize_and_mermaid_output(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    settings = _settings(tmp_path)
    store_path = Path(settings.vector_store.persist_directory) / "knowledge.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_knowledge(store_path)
    monkeypatch.setattr(
        "research_agent.cli._load_settings",
        lambda *_args, **_kwargs: settings,
    )

    graph_path = tmp_path / "graph.mmd"
    result = runner.invoke(
        app,
        [
            "knowledge",
            "summarize",
            "--topic",
            "AI news",
            "--mermaid-out",
            str(graph_path),
        ],
    )

    assert result.exit_code == 0
    assert "Knowledge Summary" in result.output
    assert graph_path.exists()
    assert "graph TD" in graph_path.read_text(encoding="utf-8")


def test_knowledge_refresh_updates_finding_and_history(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    settings = _settings(tmp_path)
    store_path = Path(settings.vector_store.persist_directory) / "knowledge.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_knowledge(store_path)
    monkeypatch.setattr(
        "research_agent.cli._load_settings",
        lambda *_args, **_kwargs: settings,
    )

    result = runner.invoke(
        app,
        [
            "knowledge",
            "refresh",
            "--topic",
            "AI news",
            "--threshold",
            "0.8",
            "--refresh-days",
            "1",
            "--statement",
            "Updated refresh statement for AI news.",
        ],
    )

    assert result.exit_code == 0
    assert "Refreshed findings:" in result.output

    payload = KnowledgeStore(store_path).load()
    refreshed = [item for item in payload.findings if item.topic == "AI news"]
    assert refreshed
    assert any("Updated refresh statement" in item.statement for item in refreshed)
    assert payload.refresh_history


def test_knowledge_export_then_import_roundtrip(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    settings_a = _settings(tmp_path / "a")
    settings_b = _settings(tmp_path / "b")
    store_path_a = Path(settings_a.vector_store.persist_directory) / "knowledge.json"
    store_path_a.parent.mkdir(parents=True, exist_ok=True)
    _seed_knowledge(store_path_a)

    active = {"settings": settings_a}
    monkeypatch.setattr(
        "research_agent.cli._load_settings",
        lambda *_args, **_kwargs: active["settings"],
    )

    export_json = tmp_path / "knowledge-export.json"
    export_md = tmp_path / "knowledge-export.md"

    exported = runner.invoke(
        app,
        [
            "knowledge",
            "export",
            str(export_json),
            "--format",
            "json",
            "--min-confidence",
            "0.2",
        ],
    )
    assert exported.exit_code == 0
    assert export_json.exists()

    exported_md = runner.invoke(
        app,
        [
            "knowledge",
            "export",
            str(export_md),
            "--format",
            "md",
        ],
    )
    assert exported_md.exit_code == 0
    assert export_md.exists()
    assert "Knowledge Base Export" in export_md.read_text(encoding="utf-8")

    active["settings"] = settings_b
    imported = runner.invoke(
        app,
        [
            "knowledge",
            "import",
            str(export_json),
        ],
    )
    assert imported.exit_code == 0
    assert "Knowledge Import" in imported.output

    store_b = KnowledgeStore(
        Path(settings_b.vector_store.persist_directory) / "knowledge.json"
    )
    payload_b = store_b.load()
    assert payload_b.findings
