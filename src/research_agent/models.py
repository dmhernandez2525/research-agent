"""Three-tier model router with fallback chains and tenacity retry.

Routes LLM calls to FAST, SMART, or STRATEGIC tiers depending on the
task requirements, with automatic fallback on failure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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

    Args:
        spec: The model specification.

    Returns:
        A configured LangChain chat model instance.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_create_chat_model is not yet implemented")


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
            NotImplementedError: Stub -- full implementation pending.
        """
        raise NotImplementedError("invoke_with_fallback is not yet implemented")

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
            NotImplementedError: Stub -- full implementation pending.
        """
        tier = self.get_tier_for_node(node_name)
        return await self.invoke_with_fallback(tier, messages, **kwargs)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelRoutingError(Exception):
    """Raised when no model is available or all fallbacks fail."""
