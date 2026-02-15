"""Health checks and self-diagnostics for research-agent."""

from __future__ import annotations

import importlib.util
import os
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from research_agent.config import Settings


class CheckStatus(StrEnum):
    """Status for a doctor check item."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class CheckResult(BaseModel):
    """A single doctor check result."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class DoctorReport(BaseModel):
    """Aggregate report for all diagnostics."""

    checks: list[CheckResult]

    @property
    def healthy(self) -> bool:
        return all(check.status != CheckStatus.FAIL for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.healthy else 1


def _check_config_schema(config_path: Path | None) -> CheckResult:
    try:
        Settings.load(config_path=config_path)
        return CheckResult(
            name="config-schema",
            status=CheckStatus.OK,
            message="Configuration schema is valid.",
        )
    except Exception as exc:
        return CheckResult(
            name="config-schema",
            status=CheckStatus.FAIL,
            message="Configuration schema validation failed.",
            details={"error": str(exc)},
        )


def _check_chromadb_directory(path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return CheckResult(
            name="chromadb-directory",
            status=CheckStatus.OK,
            message="ChromaDB persistence directory is writable.",
            details={"path": str(path)},
        )
    except OSError as exc:
        return CheckResult(
            name="chromadb-directory",
            status=CheckStatus.FAIL,
            message="ChromaDB persistence directory is not writable.",
            details={"path": str(path), "error": str(exc)},
        )


def _probe_openai(api_key: str, timeout: float) -> CheckResult:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            params={"limit": 1},
            timeout=timeout,
        )
        if response.status_code == 200:
            return CheckResult(
                name="openai-api-key",
                status=CheckStatus.OK,
                message="OpenAI API key is valid.",
            )
        return CheckResult(
            name="openai-api-key",
            status=CheckStatus.FAIL,
            message="OpenAI API key probe failed.",
            details={"status": str(response.status_code)},
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            name="openai-api-key",
            status=CheckStatus.FAIL,
            message="OpenAI API probe request failed.",
            details={"error": str(exc)},
        )


def _probe_anthropic(api_key: str, timeout: float) -> CheckResult:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            return CheckResult(
                name="anthropic-api-key",
                status=CheckStatus.OK,
                message="Anthropic API key is valid.",
            )
        return CheckResult(
            name="anthropic-api-key",
            status=CheckStatus.FAIL,
            message="Anthropic API key probe failed.",
            details={"status": str(response.status_code)},
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            name="anthropic-api-key",
            status=CheckStatus.FAIL,
            message="Anthropic API probe request failed.",
            details={"error": str(exc)},
        )


def _probe_tavily(api_key: str, timeout: float) -> CheckResult:
    payload = {
        "api_key": api_key,
        "query": "health check",
        "max_results": 1,
        "search_depth": "basic",
    }
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 200:
            return CheckResult(
                name="tavily-api-key",
                status=CheckStatus.OK,
                message="Tavily API key is valid.",
            )
        return CheckResult(
            name="tavily-api-key",
            status=CheckStatus.FAIL,
            message="Tavily API key probe failed.",
            details={"status": str(response.status_code)},
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            name="tavily-api-key",
            status=CheckStatus.FAIL,
            message="Tavily API probe request failed.",
            details={"error": str(exc)},
        )


def _check_api_keys(timeout: float = 5.0) -> list[CheckResult]:
    checks: list[CheckResult] = []

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    key_specs = [
        ("anthropic-api-key", anthropic_key, _probe_anthropic),
        ("openai-api-key", openai_key, _probe_openai),
        ("tavily-api-key", tavily_key, _probe_tavily),
    ]
    for check_name, key_value, checker in key_specs:
        if not key_value:
            checks.append(
                CheckResult(
                    name=check_name,
                    status=CheckStatus.FAIL,
                    message="API key is not set.",
                )
            )
            continue
        checks.append(checker(key_value, timeout))

    return checks


def _check_optional_dependencies() -> CheckResult:
    packages = {
        "pymupdf": "pymupdf",
        "crawl4ai": "crawl4ai",
        "sentence-transformers": "sentence_transformers",
    }
    installed: dict[str, str] = {}
    missing: list[str] = []

    for label, module_name in packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(label)
        else:
            installed[label] = "installed"

    if missing:
        return CheckResult(
            name="optional-dependencies",
            status=CheckStatus.WARN,
            message="Some optional dependencies are missing.",
            details={
                "installed": ", ".join(sorted(installed.keys())) or "none",
                "missing": ", ".join(sorted(missing)),
            },
        )

    return CheckResult(
        name="optional-dependencies",
        status=CheckStatus.OK,
        message="All optional dependencies are installed.",
        details={"installed": ", ".join(sorted(installed.keys()))},
    )


def run_doctor(
    settings: Settings,
    config_path: Path | None = None,
    check_api_probes: bool = True,
) -> DoctorReport:
    """Run all health checks and return a structured report."""
    checks = [
        _check_config_schema(config_path),
        _check_chromadb_directory(Path(settings.vector_store.persist_directory)),
    ]

    if check_api_probes:
        checks.extend(_check_api_keys())
    else:
        checks.append(
            CheckResult(
                name="api-probes",
                status=CheckStatus.WARN,
                message="API key probe checks were skipped.",
            )
        )

    checks.append(_check_optional_dependencies())
    return DoctorReport(checks=checks)
