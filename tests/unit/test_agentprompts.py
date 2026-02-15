"""Tests for Phase 23 AgentPrompts ecosystem commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from research_agent.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _make_project(projects_dir: Path, name: str) -> Path:
    project_dir = projects_dir / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "RESEARCH_PROMPT.md").write_text(
        "# Topic\nResearch CI strategy.\n\n## Constraints\n- Include caching\n",
        encoding="utf-8",
    )
    (project_dir / "BUILD_PROMPT.md").write_text(
        "# Build Tasks\n- Add CI workflow\n- Add unit tests\n",
        encoding="utf-8",
    )
    return project_dir


def test_for_project_generates_compiled_research_and_status(tmp_path: Path) -> None:
    projects_dir = tmp_path / "agent-prompts"
    project_dir = _make_project(projects_dir, "alpha")
    registry_path = tmp_path / "registry.json"

    result = runner.invoke(
        app,
        [
            "for-project",
            "alpha",
            "--projects-dir",
            str(projects_dir),
            "--registry-path",
            str(registry_path),
        ],
    )

    assert result.exit_code == 0
    output_path = project_dir / "COMPILED_RESEARCH.md"
    status_path = project_dir / ".research-status.json"
    assert output_path.exists()
    assert status_path.exists()

    compiled = output_path.read_text(encoding="utf-8")
    assert "## Build Phase Alignment" in compiled

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"


def test_projects_list_discovers_registered_projects(tmp_path: Path) -> None:
    projects_dir = tmp_path / "agent-prompts"
    _make_project(projects_dir, "alpha")
    _make_project(projects_dir, "beta")
    registry_path = tmp_path / "registry.json"

    result = runner.invoke(
        app,
        [
            "projects",
            "list",
            "--projects-dir",
            str(projects_dir),
            "--registry-path",
            str(registry_path),
        ],
    )

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output


def test_template_list_and_use(tmp_path: Path) -> None:
    listed = runner.invoke(app, ["template", "list"])
    assert listed.exit_code == 0
    assert "technology-evaluation" in listed.output

    output_path = tmp_path / "RESEARCH_PROMPT.md"
    rendered = runner.invoke(
        app,
        [
            "template",
            "use",
            "technology-evaluation",
            "--project-name",
            "demo-service",
            "--language",
            "python",
            "--focus-area",
            "security",
            "--output",
            str(output_path),
        ],
    )
    assert rendered.exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "demo-service" in content
    assert "security" in content
    assert "{{PROJECT_NAME}}" not in content


def test_watch_once_triggers_generation(tmp_path: Path) -> None:
    projects_dir = tmp_path / "agent-prompts"
    project_dir = _make_project(projects_dir, "alpha")
    registry_path = tmp_path / "registry.json"

    result = runner.invoke(
        app,
        [
            "watch",
            "--projects-dir",
            str(projects_dir),
            "--registry-path",
            str(registry_path),
            "--once",
            "--no-notify",
        ],
    )

    assert result.exit_code == 0
    assert (project_dir / "COMPILED_RESEARCH.md").exists()
