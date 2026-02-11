"""Unit tests for research_agent.config - Settings loading and validation."""

from __future__ import annotations

from typing import Any

import pytest

# TODO: Uncomment once the config module is implemented.
# from research_agent.config import Settings, load_config


class TestConfigLoading:
    """Verify that configuration files are loaded and parsed correctly."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_load_default_config_from_yaml(self, tmp_path: Any) -> None:
        """Loading config.yaml should produce a valid Settings object."""
        # TODO: Write a minimal config.yaml to tmp_path, call load_config,
        #       and assert the returned Settings object has the expected fields.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_missing_config_file_raises(self, tmp_path: Any) -> None:
        """Attempting to load a non-existent config file should raise FileNotFoundError."""
        # TODO: Call load_config with a path that does not exist and assert
        #       the appropriate exception.


class TestEnvVarOverride:
    """Environment variables should override YAML values."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_env_var_overrides_llm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting RESEARCH_AGENT_LLM__MODEL should override the YAML value."""
        # TODO: Use monkeypatch.setenv to set the env var, load config,
        #       and assert the model field matches the env value.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_env_var_overrides_cost_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting RESEARCH_AGENT_COSTS__MAX_COST_PER_RUN should override YAML."""
        # TODO: Set the env var to a custom budget, load config, and verify.


class TestConfigValidation:
    """Pydantic validation should reject invalid configurations."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_negative_max_tokens_rejected(self) -> None:
        """max_tokens must be a positive integer."""
        # TODO: Instantiate Settings with max_tokens=-1 and assert
        #       ValidationError is raised.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_unsupported_provider_rejected(self) -> None:
        """An unknown LLM provider should fail validation."""
        # TODO: Pass provider="unknown_provider" and assert ValidationError.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.config exists")
    def test_default_values_applied(self, sample_config: dict[str, Any]) -> None:
        """Omitted optional fields should receive sensible defaults."""
        # TODO: Create a Settings object with minimal input and verify
        #       defaults like temperature=0.1, retries=3 are populated.
