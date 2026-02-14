"""Unit tests for enhancement-mode functionality."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Settings
from research_agent.enhance_context import build_project_context
from research_agent.enhance_engine import (
    generate_enhancement_report,
    identify_opportunities,
    plan_incremental_research,
)
from research_agent.enhance_models import (
    DeltaSummary,
    EnhancementOpportunity,
    KnowledgeEntry,
    OpportunityCategory,
    ProjectContext,
    ProjectDependency,
)
from research_agent.enhance_store import KnowledgeStore

runner = CliRunner()


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "sample-project"
    (project / "src").mkdir(parents=True)
    (project / "frontend").mkdir(parents=True)
    (project / "README.md").write_text("# Sample\n\nA demo project.", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[project]\nname='sample'\nversion='0.1.0'\ndependencies=['fastapi>=0.110']\n",
        encoding="utf-8",
    )
    (project / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (project / "frontend" / "app.ts").write_text("export const x = 1", encoding="utf-8")
    (project / "ignored.py").write_text("x = 1", encoding="utf-8")
    (project / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (project / "src" / "large.py").write_text("a" * 1200, encoding="utf-8")
    return project


def test_context_loader_multilanguage_gitignore_and_size_limits(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    context = build_project_context(project, max_files=20, max_chars_per_file=120)

    assert "python" in context.languages
    assert "typescript" in context.languages
    key_paths = {Path(item.path).name for item in context.key_files}
    assert "ignored.py" not in key_paths

    large_items = [item for item in context.key_files if item.path.endswith("large.py")]
    assert large_items
    assert large_items[0].truncated is True


def test_opportunity_identifier_ranking_and_focus_filter() -> None:
    context = ProjectContext(
        project_path="/tmp/x",
        project_name="demo",
        description="",
        languages=["python"],
        framework="fastapi",
        dependencies=[
            ProjectDependency(name="fastapi", version=">=0.110", source="pyproject")
        ],
        key_files=[],
    )

    opportunities = identify_opportunities(context)
    assert opportunities
    assert opportunities[0].impact_score >= opportunities[-1].impact_score

    focused = identify_opportunities(
        context,
        focus_areas={OpportunityCategory.SECURITY, OpportunityCategory.PERFORMANCE},
    )
    assert all(
        opp.category in {OpportunityCategory.SECURITY, OpportunityCategory.PERFORMANCE}
        for opp in focused
    )


def test_incremental_staleness_and_force_refresh(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.json")

    fresh_entry = KnowledgeEntry(
        topic="Security controls and secrets handling review",
        category=OpportunityCategory.SECURITY,
        updated_at=datetime.now(tz=UTC).isoformat(),
        finding="fresh",
        query="q1",
    )
    old_entry = KnowledgeEntry(
        topic="Performance profiling and hotspot optimization",
        category=OpportunityCategory.PERFORMANCE,
        updated_at=(datetime.now(tz=UTC) - timedelta(days=90)).isoformat(),
        finding="old",
        query="q2",
    )
    shared_entry = KnowledgeEntry(
        topic="Performance profiling and hotspot optimization",
        category=OpportunityCategory.PERFORMANCE,
        finding="shared",
        query="q3",
    )

    store.upsert_entries("proj-a", [fresh_entry, old_entry])
    store.upsert_entries("proj-b", [shared_entry])

    opportunities = [
        EnhancementOpportunity(
            title="Security controls and secrets handling review",
            category=OpportunityCategory.SECURITY,
            impact_score=5,
            effort_score=3,
            rationale="r",
            suggested_query="q",
        ),
        EnhancementOpportunity(
            title="Performance profiling and hotspot optimization",
            category=OpportunityCategory.PERFORMANCE,
            impact_score=4,
            effort_score=2,
            rationale="r",
            suggested_query="q",
        ),
    ]

    refresh_targets, delta, shared = plan_incremental_research(
        project_id="proj-a",
        opportunities=opportunities,
        store=store,
        stale_days=30,
        force_refresh=False,
    )

    assert (
        "Security controls and secrets handling review" in delta.skipped_recent_topics
    )
    assert any(
        op.title == "Performance profiling and hotspot optimization"
        for op in refresh_targets
    )
    assert shared

    force_targets, _delta_force, _shared_force = plan_incremental_research(
        project_id="proj-a",
        opportunities=opportunities,
        store=store,
        stale_days=30,
        force_refresh=True,
    )
    assert len(force_targets) == len(opportunities)


def test_report_generation_and_apply_to(tmp_path: Path, monkeypatch: object) -> None:
    project = _make_project(tmp_path)
    output = tmp_path / "COMPILED_RESEARCH.md"
    apply_to = tmp_path / "docs" / "ENHANCEMENTS.md"

    settings = Settings()
    settings.vector_store.persist_directory = tmp_path / "vector"
    monkeypatch.setattr(
        "research_agent.cli._load_settings", lambda *_args, **_kwargs: settings
    )

    result = runner.invoke(
        app,
        [
            "enhance",
            "--project",
            str(project),
            "--focus",
            "security,performance",
            "--output",
            str(output),
            "--apply-to",
            str(apply_to),
            "--force-refresh",
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert apply_to.exists()
    content = output.read_text(encoding="utf-8")
    assert "## Current State" in content
    assert "## Recommendations" in content
    assert "## Priority Matrix" in content


def test_generate_enhancement_report_structure() -> None:
    context = ProjectContext(
        project_path="/tmp/demo",
        project_name="demo",
        description="desc",
        languages=["python"],
        framework="fastapi",
        dependencies=[ProjectDependency(name="fastapi", version=">=0.1", source="x")],
        key_files=[],
    )
    opportunities = [
        EnhancementOpportunity(
            title="Security controls and secrets handling review",
            category=OpportunityCategory.SECURITY,
            impact_score=5,
            effort_score=3,
            rationale="r",
            suggested_query="q",
        )
    ]
    delta = DeltaSummary(new_topics=[opportunities[0].title])
    shared = [
        KnowledgeEntry(
            topic="Shared",
            category=OpportunityCategory.SECURITY,
            finding="f",
            query="q",
        )
    ]

    report = generate_enhancement_report(context, opportunities, delta, shared)
    assert "# COMPILED_RESEARCH" in report
    assert "## Implementation Steps" in report
    assert "## Dependency Upgrade Analysis" in report
