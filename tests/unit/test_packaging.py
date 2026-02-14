"""Packaging and version metadata tests."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

from research_agent import __version__


def _load_pyproject() -> dict[str, object]:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        return tomllib.load(f)


def test_console_entrypoint_is_configured() -> None:
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert scripts["research-agent"] == "research_agent.cli:main"


def test_module_version_matches_pyproject() -> None:
    data = _load_pyproject()
    expected = data["project"]["version"]
    assert __version__ == expected


def test_importlib_metadata_version_matches_when_installed() -> None:
    data = _load_pyproject()
    expected = data["project"]["version"]
    try:
        installed_version = importlib.metadata.version("research-agent")
    except importlib.metadata.PackageNotFoundError:
        installed_version = expected
    assert installed_version == expected
