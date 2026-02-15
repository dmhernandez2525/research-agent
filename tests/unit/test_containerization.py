"""Containerization configuration tests."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_dockerfile_uses_python_311_and_non_root_user() -> None:
    dockerfile = (_repo_root() / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "USER agent" in dockerfile


def test_dockerfile_has_healthcheck_and_volumes() -> None:
    dockerfile = (_repo_root() / "Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in dockerfile
    assert 'research-agent", "doctor", "--quiet", "--no-api-probes"' in dockerfile
    assert 'VOLUME ["/app/data", "/app/reports"]' in dockerfile


def test_docker_compose_persists_data_and_reports() -> None:
    compose = (_repo_root() / "docker-compose.yml").read_text(encoding="utf-8")
    assert "research_agent_data:/app/data" in compose
    assert "research_agent_reports:/app/reports" in compose
    assert "read_only: true" in compose
