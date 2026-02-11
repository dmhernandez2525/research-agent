"""Configuration with 4-layer resolution: defaults -> YAML -> env -> CLI.

Uses pydantic-settings with YamlConfigSettingsSource for layered configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

try:
    from pydantic_settings import YamlConfigSettingsSource
except ImportError:  # pragma: no cover
    YamlConfigSettingsSource = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=120, gt=0, description="Request timeout in seconds.")
    retries: int = Field(default=3, ge=0)


class SearchSettings(BaseModel):
    """Web search provider configuration."""

    provider: Literal["tavily", "serper", "searxng"] = "tavily"
    max_results: int = Field(default=10, gt=0)
    search_depth: Literal["basic", "advanced"] = "advanced"


class ScrapingSettings(BaseModel):
    """Web scraping configuration."""

    engine: Literal["trafilatura", "httpx"] = "trafilatura"
    timeout: int = Field(
        default=30, gt=0, description="Per-request timeout in seconds."
    )
    max_concurrent: int = Field(default=5, gt=0)
    max_content_length: int = Field(
        default=500_000, gt=0, description="Max characters per scraped page."
    )


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    provider: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = Field(default=768, gt=0)


class VectorStoreSettings(BaseModel):
    """Vector store (ChromaDB) configuration."""

    persist_directory: Path = Path("./data/chromadb")
    collection_name: str = "research_docs"


class CostSettings(BaseModel):
    """Budget and cost guardrail configuration."""

    max_cost_per_run: float = Field(
        default=2.00, gt=0.0, description="Maximum cost in USD per research run."
    )
    max_llm_calls_per_run: int = Field(default=50, gt=0)
    warn_at_percentage: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Warn when budget usage exceeds this percent.",
    )


class CheckpointSettings(BaseModel):
    """Checkpoint / crash-recovery configuration."""

    enabled: bool = True
    directory: Path = Path("./data/checkpoints")
    save_interval: int = Field(
        default=5, gt=0, description="Save checkpoint every N steps."
    )
    max_checkpoints: int = Field(
        default=5, gt=0, description="Maximum number of retained checkpoints."
    )


class ReportSettings(BaseModel):
    """Report output configuration."""

    output_dir: Path = Path("./reports")
    format: Literal["markdown", "html", "pdf"] = "markdown"
    max_length: int = Field(
        default=10_000,
        gt=0,
        description="Soft max length in tokens for the final report.",
    )


class LoggingSettings(BaseModel):
    """Logging / observability configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    file: Path | None = None


# ---------------------------------------------------------------------------
# Main Settings (4-layer resolution)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path("config.yaml")


class Settings(BaseSettings):
    """Top-level application settings.

    Resolution order (last wins):
        1. Field defaults (defined above)
        2. YAML config file (``config.yaml`` or ``--config`` path)
        3. Environment variables (prefixed ``RESEARCH_AGENT_``)
        4. CLI overrides (applied programmatically after loading)
    """

    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_AGENT_",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    checkpoints: CheckpointSettings = Field(default_factory=CheckpointSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings source priority.

        Resolution order (last wins):
            init_settings (CLI) > env > yaml > defaults

        Args:
            settings_cls: The settings class.
            init_settings: Init / programmatic overrides.
            env_settings: Environment variable source.
            dotenv_settings: Dotenv file source.
            file_secret_settings: Secret file source.

        Returns:
            Ordered tuple of settings sources (first = highest priority).
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]

        if YamlConfigSettingsSource is not None:
            sources.append(YamlConfigSettingsSource(settings_cls))

        return tuple(sources)

    @classmethod
    def load(cls, config_path: Path | None = None, **overrides: Any) -> Settings:
        """Load settings with optional config path and CLI overrides.

        Args:
            config_path: Optional path to a YAML config file.
            **overrides: Key-value CLI overrides applied at highest priority.

        Returns:
            Fully-resolved Settings instance.
        """
        if config_path is not None:
            overrides.setdefault("_yaml_file", str(config_path))

        return cls(**overrides)
