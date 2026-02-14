"""Configuration with 4-layer resolution: defaults -> YAML -> env -> CLI.

Uses pydantic-settings with YamlConfigSettingsSource for layered configuration.
Supports ``.env`` file loading, ``RESEARCH_AGENT_`` prefixed env vars, and
nested delimiter ``__`` for overriding sub-model fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import structlog
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

try:
    from pydantic_settings import YamlConfigSettingsSource
except ImportError:  # pragma: no cover
    YamlConfigSettingsSource = None  # type: ignore[assignment, misc]

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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
    js_fallback: bool = Field(
        default=False,
        description="Use Crawl4AI as fallback for JS-heavy sites with low quality scores.",
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


class RecoveryNodePolicySettings(BaseModel):
    """Per-node retry/backoff policy overrides."""

    attempts: int = Field(default=3, ge=1, le=10)
    backoff_initial_seconds: float = Field(default=0.5, gt=0.0)
    backoff_max_seconds: float = Field(default=8.0, gt=0.0)


class RecoverySettings(BaseModel):
    """Error recovery orchestration settings."""

    enabled: bool = True
    default_policy: RecoveryNodePolicySettings = Field(
        default_factory=RecoveryNodePolicySettings
    )
    node_policies: dict[str, RecoveryNodePolicySettings] = Field(default_factory=dict)
    circuit_breaker_threshold: int = Field(default=3, ge=1, le=20)
    circuit_breaker_cooldown_seconds: int = Field(default=120, ge=1, le=3600)
    dead_letter_max_entries: int = Field(default=200, ge=1, le=5000)


class APISettings(BaseModel):
    """FastAPI server settings."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_sessions: int = Field(default=3, ge=1, le=100)
    queue_limit: int = Field(default=50, ge=0, le=10_000)
    api_key_file: Path = Path("./data/api_keys.json")
    frontend_dist_dir: Path = Path("./frontend/dist")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    webhook_url: str | None = None
    slack_webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: str | None = None
    notify_on: list[Literal["completion", "error", "budget_warning"]] = Field(
        default_factory=lambda: cast(
            "list[Literal['completion', 'error', 'budget_warning']]",
            ["completion", "error"],
        )
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
    serial_synthesis_threshold: int = Field(
        default=3,
        ge=1,
        description="Use serial section-by-section synthesis when subtopic count exceeds this.",
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
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    _config_path_override: ClassVar[Path | None] = None

    llm: LLMSettings = Field(default_factory=LLMSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    checkpoints: CheckpointSettings = Field(default_factory=CheckpointSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    api: APISettings = Field(default_factory=APISettings)
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

        Resolution order (first = highest priority):
            init_settings (CLI) > env_settings > dotenv (.env) > yaml > defaults

        Args:
            settings_cls: The settings class.
            init_settings: Init / programmatic overrides.
            env_settings: Environment variable source.
            dotenv_settings: Dotenv file source (.env).
            file_secret_settings: Secret file source (unused).

        Returns:
            Ordered tuple of settings sources (first = highest priority).
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]

        if YamlConfigSettingsSource is not None:
            yaml_file = cls._config_path_override or settings_cls.model_config.get(
                "yaml_file", "config.yaml"
            )
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))

        return tuple(sources)

    @classmethod
    def load(cls, config_path: Path | None = None, **overrides: Any) -> Settings:
        """Load settings with optional config path and CLI overrides.

        Args:
            config_path: Optional path to a YAML config file.
            **overrides: Key-value CLI overrides applied at highest priority.

        Returns:
            Fully-resolved Settings instance.

        Raises:
            ValidationError: If any setting value fails validation.
        """
        cls._config_path_override = config_path
        try:
            return cls(**overrides)
        finally:
            cls._config_path_override = None


def format_validation_error(exc: ValidationError) -> str:
    """Format a Pydantic ValidationError into a user-friendly message.

    Args:
        exc: The validation error to format.

    Returns:
        A multi-line string with each error on its own line.
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = " -> ".join(str(part) for part in error["loc"])
        msg = error["msg"]
        raw_input = error.get("input")
        if raw_input is not None:
            lines.append(f"  {loc}: {msg} (got {raw_input!r})")
        else:
            lines.append(f"  {loc}: {msg}")
    return "Configuration error:\n" + "\n".join(lines)
