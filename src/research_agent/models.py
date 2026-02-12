"""Three-tier model router with fallback chains and tenacity retry.

Routes LLM calls to FAST, SMART, or STRATEGIC tiers depending on the
task requirements, with automatic fallback on failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.exceptions import ModelRoutingError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_MIN_SECONDS = 1
_BACKOFF_MAX_SECONDS = 10


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelTier(StrEnum):
    """LLM routing tier based on task complexity."""

    FAST = "FAST"
    SMART = "SMART"
    STRATEGIC = "STRATEGIC"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Specification for a single model in a fallback chain."""

    provider: str = Field(description="Provider name: anthropic, openai, google.")
    model_id: str = Field(description="Model identifier string.")
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


# Default fallback chains per tier
DEFAULT_CHAINS: dict[ModelTier, list[ModelSpec]] = {
    ModelTier.FAST: [
        ModelSpec(provider="anthropic", model_id="claude-haiku-3-5-20241022"),
        ModelSpec(provider="openai", model_id="gpt-4o-mini"),
    ],
    ModelTier.SMART: [
        ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5-20250929"),
        ModelSpec(provider="openai", model_id="gpt-4o"),
    ],
    ModelTier.STRATEGIC: [
        ModelSpec(
            provider="anthropic", model_id="claude-sonnet-4-5-20250929", max_tokens=8192
        ),
        ModelSpec(provider="openai", model_id="gpt-4o", max_tokens=8192),
    ],
}

# Mapping of graph node names to their recommended tier
NODE_TIER_MAP: dict[str, ModelTier] = {
    "plan": ModelTier.SMART,
    "search": ModelTier.FAST,
    "scrape": ModelTier.FAST,
    "summarize": ModelTier.SMART,
    "synthesize": ModelTier.STRATEGIC,
}


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


def _create_chat_model(spec: ModelSpec) -> BaseChatModel:
    """Instantiate a LangChain chat model from a ModelSpec.

    Uses lazy imports so provider SDKs are only loaded when needed.

    Args:
        spec: The model specification.

    Returns:
        A configured LangChain chat model instance.

    Raises:
        ModelRoutingError: If the provider is not supported.
    """
    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=spec.model_id,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )

    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=spec.model_id,
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
        )

    raise ModelRoutingError(f"Unsupported provider: {spec.provider!r}")


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """Routes LLM calls to the appropriate tier with automatic fallback.

    Attributes:
        chains: Mapping of tiers to fallback model chains.
    """

    def __init__(
        self,
        chains: dict[ModelTier, list[ModelSpec]] | None = None,
    ) -> None:
        """Initialize the model router.

        Args:
            chains: Optional custom fallback chains. Defaults to
                ``DEFAULT_CHAINS``.
        """
        self.chains = chains or dict(DEFAULT_CHAINS)
        self._model_cache: dict[str, BaseChatModel] = {}

    def get_tier_for_node(self, node_name: str) -> ModelTier:
        """Return the recommended model tier for a graph node.

        Args:
            node_name: The graph node name.

        Returns:
            The model tier for the node (defaults to SMART).
        """
        return NODE_TIER_MAP.get(node_name, ModelTier.SMART)

    @staticmethod
    async def _invoke_with_retry(
        model: BaseChatModel,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Invoke a model with tenacity retry (3 attempts, exponential backoff).

        Args:
            model: The LangChain chat model to invoke.
            messages: Chat messages to send.
            **kwargs: Additional keyword arguments.

        Returns:
            The model's response.

        Raises:
            RetryError: If all retry attempts fail.
        """

        @retry(
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(min=_BACKOFF_MIN_SECONDS, max=_BACKOFF_MAX_SECONDS),
            reraise=False,
        )
        async def _do_invoke() -> Any:
            return await model.ainvoke(messages, **kwargs)

        return await _do_invoke()

    def get_model(self, tier: ModelTier) -> BaseChatModel:
        """Get the primary model for a tier (with caching).

        Args:
            tier: The model tier.

        Returns:
            A LangChain chat model instance.

        Raises:
            ModelRoutingError: If no models are available for the tier.
        """
        chain = self.chains.get(tier, [])
        if not chain:
            raise ModelRoutingError(f"No models configured for tier {tier.value}")

        primary = chain[0]
        cache_key = f"{primary.provider}:{primary.model_id}"

        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = _create_chat_model(primary)

        return self._model_cache[cache_key]

    async def invoke_with_fallback(
        self,
        tier: ModelTier,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Invoke a model with tier-based fallback and retry.

        Tries each model in the tier's fallback chain. Each model is
        retried individually (3 attempts with exponential backoff) before
        falling to the next model in the chain.

        Args:
            tier: The model tier to use.
            messages: Chat messages to send.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            The model's response.

        Raises:
            ModelRoutingError: If all models in the chain fail.
        """
        chain = self.chains.get(tier, [])
        if not chain:
            raise ModelRoutingError(f"No models configured for tier {tier.value}")

        errors: list[tuple[str, Exception]] = []

        for spec in chain:
            cache_key = f"{spec.provider}:{spec.model_id}"

            if cache_key not in self._model_cache:
                try:
                    self._model_cache[cache_key] = _create_chat_model(spec)
                except ModelRoutingError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "model_instantiation_failed",
                        provider=spec.provider,
                        model_id=spec.model_id,
                        error=str(exc),
                    )
                    errors.append((cache_key, exc))
                    continue

            model = self._model_cache[cache_key]

            try:
                result = await self._invoke_with_retry(model, messages, **kwargs)
                logger.info(
                    "model_invoke_success",
                    provider=spec.provider,
                    model_id=spec.model_id,
                    tier=tier.value,
                )
                return result
            except RetryError as exc:
                last_err = exc.last_attempt.exception() if exc.last_attempt else exc
                logger.warning(
                    "model_retries_exhausted",
                    provider=spec.provider,
                    model_id=spec.model_id,
                    tier=tier.value,
                    error=str(last_err),
                )
                errors.append((cache_key, exc))

        failed_models = ", ".join(key for key, _ in errors)
        raise ModelRoutingError(
            f"All models in {tier.value} chain failed: [{failed_models}]"
        )

    async def invoke_for_node(
        self,
        node_name: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Invoke the appropriate model for a graph node.

        Convenience wrapper that looks up the tier for the node and
        delegates to ``invoke_with_fallback``.

        Args:
            node_name: The graph node name.
            messages: Chat messages to send.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            The model's response.

        Raises:
            ModelRoutingError: If all models in the resolved tier's chain fail.
        """
        tier = self.get_tier_for_node(node_name)
        return await self.invoke_with_fallback(tier, messages, **kwargs)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


