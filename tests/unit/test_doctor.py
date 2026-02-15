"""Unit tests for doctor diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import respx
from httpx import Response
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.config import Settings
from research_agent.doctor import (
    CheckResult,
    CheckStatus,
    DoctorReport,
    _check_api_keys,
    _check_chromadb_directory,
    _check_config_schema,
    _check_optional_dependencies,
    run_doctor,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class TestCheckConfigSchema:
    """Config schema diagnostics."""

    def test_valid_schema_returns_ok(self) -> None:
        result = _check_config_schema(config_path=None)
        assert result.status == CheckStatus.OK


class TestCheckChromaDirectory:
    """ChromaDB directory diagnostics."""

    def test_directory_writable(self, tmp_path: Path) -> None:
        result = _check_chromadb_directory(tmp_path / "chromadb")
        assert result.status == CheckStatus.OK


class TestOptionalDependencies:
    """Optional dependency diagnostics."""

    def test_reports_warn_when_optional_missing(self, monkeypatch: object) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        result = _check_optional_dependencies()
        assert result.status == CheckStatus.WARN
        assert "missing" in result.details


class TestApiChecks:
    """API probe diagnostics."""

    @respx.mock
    def test_api_key_probes_pass_on_200(self, monkeypatch: object) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
        monkeypatch.setenv("OPENAI_API_KEY", "open")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly")

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json={"id": "msg"})
        )
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=Response(200, json={"data": []})
        )
        respx.post("https://api.tavily.com/search").mock(
            return_value=Response(200, json={"results": []})
        )

        checks = _check_api_keys(timeout=0.5)
        assert len(checks) == 3
        assert all(check.status == CheckStatus.OK for check in checks)

    def test_api_key_missing_is_fail(self, monkeypatch: object) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        checks = _check_api_keys(timeout=0.01)
        assert all(check.status == CheckStatus.FAIL for check in checks)


class TestRunDoctor:
    """Aggregate doctor workflow."""

    def test_run_doctor_includes_core_checks(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.vector_store.persist_directory = tmp_path / "db"
        report = run_doctor(settings, check_api_probes=False)
        names = {check.name for check in report.checks}
        assert "config-schema" in names
        assert "chromadb-directory" in names
        assert "api-probes" in names
        assert "optional-dependencies" in names


class TestDoctorCli:
    """CLI formatting and exit behavior."""

    def test_doctor_command_renders_table(
        self, monkeypatch: object
    ) -> None:
        fake_report = DoctorReport(
            checks=[
                CheckResult(name="a", status=CheckStatus.OK, message="ok"),
                CheckResult(name="b", status=CheckStatus.FAIL, message="bad"),
            ]
        )

        monkeypatch.setattr(
            "research_agent.cli._load_settings", lambda *_args, **_kwargs: Settings()
        )
        monkeypatch.setattr(
            "research_agent.cli.run_doctor", lambda **_kwargs: fake_report
        )

        result = runner.invoke(app, ["doctor", "--no-api-probes"])
        assert result.exit_code == 1
        assert "Research Agent Doctor" in result.output
        assert "FAIL" in result.output

    def test_doctor_command_quiet_healthy_zero_exit(
        self, monkeypatch: object
    ) -> None:
        fake_report = DoctorReport(
            checks=[CheckResult(name="a", status=CheckStatus.OK, message="ok")]
        )
        monkeypatch.setattr(
            "research_agent.cli._load_settings", lambda *_args, **_kwargs: Settings()
        )
        monkeypatch.setattr(
            "research_agent.cli.run_doctor", lambda **_kwargs: fake_report
        )

        result = runner.invoke(app, ["doctor", "--quiet"])
        assert result.exit_code == 0
        assert "Research Agent Doctor" not in result.output
