"""Unit tests for research_agent.config - Settings loading and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from research_agent.config import (
    APISettings,
    CheckpointSettings,
    CostSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    RecoverySettings,
    ReportSettings,
    ScrapingSettings,
    SearchSettings,
    Settings,
    VectorStoreSettings,
    format_validation_error,
)

# ---- Sub-model defaults ------------------------------------------------------


class TestLLMSettings:
    """LLMSettings should have sensible defaults and validation."""

    def test_default_values(self) -> None:
        s = LLMSettings()
        assert s.provider == "anthropic"
        assert s.temperature == 0.1
        assert s.max_tokens == 4096
        assert s.retries == 3

    def test_invalid_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMSettings(temperature=3.0)

    def test_negative_max_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMSettings(max_tokens=-1)

    def test_zero_max_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMSettings(max_tokens=0)


class TestSearchSettings:
    """SearchSettings defaults and constraints."""

    def test_default_values(self) -> None:
        s = SearchSettings()
        assert s.provider == "tavily"
        assert s.max_results == 10
        assert s.search_depth == "advanced"

    def test_invalid_search_depth_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchSettings(search_depth="ultra")  # type: ignore[arg-type]


class TestScrapingSettings:
    """ScrapingSettings defaults."""

    def test_default_values(self) -> None:
        s = ScrapingSettings()
        assert s.engine == "trafilatura"
        assert s.timeout == 30
        assert s.max_concurrent == 5
        assert s.max_content_length == 500_000


class TestCostSettings:
    """CostSettings defaults and validation."""

    def test_default_values(self) -> None:
        s = CostSettings()
        assert s.max_cost_per_run == 2.00
        assert s.max_llm_calls_per_run == 50
        assert s.warn_at_percentage == 80

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CostSettings(max_cost_per_run=-1.0)


class TestCheckpointSettings:
    """CheckpointSettings defaults."""

    def test_default_enabled(self) -> None:
        s = CheckpointSettings()
        assert s.enabled is True
        assert s.save_interval == 5
        assert s.max_checkpoints == 5


class TestAPISettings:
    """APISettings defaults."""

    def test_default_values(self) -> None:
        s = APISettings()
        assert s.port == 8000
        assert s.frontend_dist_dir.name == "dist"
        assert s.max_concurrent_sessions == 3
        assert s.rate_limit_per_minute == 60


class TestReportSettings:
    """ReportSettings defaults."""

    def test_default_format(self) -> None:
        s = ReportSettings()
        assert s.format == "markdown"
        assert s.max_length == 10_000

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReportSettings(format="docx")  # type: ignore[arg-type]


class TestLoggingSettings:
    """LoggingSettings defaults."""

    def test_defaults(self) -> None:
        s = LoggingSettings()
        assert s.level == "INFO"
        assert s.format == "console"
        assert s.file is None


class TestEmbeddingSettings:
    """EmbeddingSettings defaults."""

    def test_default_values(self) -> None:
        s = EmbeddingSettings()
        assert s.provider == "sentence_transformers"
        assert s.dimensions == 768


class TestVectorStoreSettings:
    """VectorStoreSettings defaults."""

    def test_default_collection_name(self) -> None:
        s = VectorStoreSettings()
        assert s.collection_name == "research_docs"


# ---- Top-level Settings ------------------------------------------------------


class TestSettings:
    """Top-level Settings should compose all sub-models."""

    def test_default_construction(self) -> None:
        s = Settings()
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.search, SearchSettings)
        assert isinstance(s.scraping, ScrapingSettings)
        assert isinstance(s.costs, CostSettings)
        assert isinstance(s.checkpoints, CheckpointSettings)
        assert isinstance(s.recovery, RecoverySettings)
        assert isinstance(s.api, APISettings)
        assert isinstance(s.report, ReportSettings)
        assert isinstance(s.logging, LoggingSettings)
        assert isinstance(s.embedding, EmbeddingSettings)
        assert isinstance(s.vector_store, VectorStoreSettings)

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_AGENT_LLM__TEMPERATURE", "0.5")
        s = Settings()
        assert s.llm.temperature == 0.5

    def test_init_override(self) -> None:
        s = Settings(llm=LLMSettings(provider="openai", model="gpt-4o"))
        assert s.llm.provider == "openai"
        assert s.llm.model == "gpt-4o"

    def test_load_with_defaults(self) -> None:
        s = Settings.load()
        assert s.llm.provider == "anthropic"

    def test_load_with_overrides(self) -> None:
        s = Settings.load(llm=LLMSettings(temperature=0.9))
        assert s.llm.temperature == 0.9

    def test_nested_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_AGENT_SEARCH__MAX_RESULTS", "25")
        s = Settings()
        assert s.search.max_results == 25

    def test_extra_fields_ignored(self) -> None:
        s = Settings(unknown_field="should_be_ignored")  # type: ignore[call-arg]
        assert isinstance(s.llm, LLMSettings)


# ---- YAML loading -----------------------------------------------------------


class TestYamlLoading:
    """Settings should load values from a YAML config file."""

    def test_load_from_custom_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(
            "llm:\n  temperature: 0.7\n  model: gpt-4o\nsearch:\n  max_results: 25\n"
        )
        s = Settings.load(config_path=yaml_file)
        assert s.llm.temperature == 0.7
        assert s.llm.model == "gpt-4o"
        assert s.search.max_results == 25

    def test_yaml_values_override_defaults(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("costs:\n  max_cost_per_run: 5.00\n")
        s = Settings.load(config_path=yaml_file)
        assert s.costs.max_cost_per_run == 5.00

    def test_missing_yaml_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        s = Settings.load(config_path=missing)
        assert s.llm.provider == "anthropic"
        assert s.llm.temperature == 0.1

    def test_partial_yaml_fills_remaining_with_defaults(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("llm:\n  temperature: 0.5\n")
        s = Settings.load(config_path=yaml_file)
        assert s.llm.temperature == 0.5
        assert s.llm.provider == "anthropic"
        assert s.search.max_results == 10


# ---- Dotenv loading ----------------------------------------------------------


class TestDotenvLoading:
    """Settings should load RESEARCH_AGENT_-prefixed vars from .env files."""

    def test_dotenv_file_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCH_AGENT_LLM__TEMPERATURE=0.7\n")
        monkeypatch.chdir(tmp_path)
        s = Settings()
        assert s.llm.temperature == 0.7

    def test_env_var_overrides_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCH_AGENT_LLM__TEMPERATURE=0.3\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RESEARCH_AGENT_LLM__TEMPERATURE", "0.9")
        s = Settings()
        assert s.llm.temperature == 0.9

    def test_missing_dotenv_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        s = Settings()
        assert s.llm.temperature == 0.1


# ---- Layer resolution order --------------------------------------------------


class TestLayerResolution:
    """Verify 4-layer resolution: defaults < YAML < .env < env vars < CLI."""

    def test_yaml_overrides_defaults(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  temperature: 0.3\n")
        s = Settings.load(config_path=yaml_file)
        assert s.llm.temperature == 0.3

    def test_dotenv_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  temperature: 0.3\n")
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCH_AGENT_LLM__TEMPERATURE=0.5\n")
        monkeypatch.chdir(tmp_path)
        s = Settings.load(config_path=yaml_file)
        assert s.llm.temperature == 0.5

    def test_env_var_overrides_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCH_AGENT_LLM__TEMPERATURE=0.5\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RESEARCH_AGENT_LLM__TEMPERATURE", "0.8")
        s = Settings()
        assert s.llm.temperature == 0.8

    def test_cli_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_AGENT_LLM__TEMPERATURE", "0.8")
        s = Settings(llm=LLMSettings(temperature=1.5))
        assert s.llm.temperature == 1.5

    def test_full_chain(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n  temperature: 0.3\n"
            "search:\n  max_results: 20\n"
            "costs:\n  max_cost_per_run: 5.00\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text("RESEARCH_AGENT_LLM__TEMPERATURE=0.5\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RESEARCH_AGENT_SEARCH__MAX_RESULTS", "30")
        s = Settings.load(
            config_path=yaml_file,
            costs=CostSettings(max_cost_per_run=10.00),
        )
        # YAML set temperature to 0.3, .env overrides to 0.5
        assert s.llm.temperature == 0.5
        # YAML set max_results to 20, env var overrides to 30
        assert s.search.max_results == 30
        # YAML set max_cost to 5.00, CLI overrides to 10.00
        assert s.costs.max_cost_per_run == 10.00


# ---- Validation error formatting ---------------------------------------------


class TestFormatValidationError:
    """format_validation_error should produce user-friendly messages."""

    def test_includes_field_path(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LLMSettings(temperature=3.0)
        msg = format_validation_error(exc_info.value)
        assert "temperature" in msg

    def test_includes_input_value(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LLMSettings(temperature=3.0)
        msg = format_validation_error(exc_info.value)
        assert "3.0" in msg

    def test_multiple_errors_all_shown(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LLMSettings(temperature=5.0, max_tokens=-1)
        msg = format_validation_error(exc_info.value)
        assert "temperature" in msg
        assert "max_tokens" in msg
        assert msg.startswith("Configuration error:")
