"""Three-tier model router with fallback chains and tenacity retry.

Routes LLM calls to FAST, SMART, or STRATEGIC tiers depending on the
task requirements, with automatic fallback on failure. Uses litellm
for provider-agnostic LLM access.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.exceptions import ModelRoutingError

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_MIN_SECONDS = 1
_BACKOFF_MAX_SECONDS = 10

_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai", "google"})

_PROVIDER_PREFIX: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
}


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
# litellm helpers
# ---------------------------------------------------------------------------


def _resolve_litellm_model(spec: ModelSpec) -> str:
    """Build a litellm model identifier from a ModelSpec.

    litellm uses provider-prefixed identifiers (e.g.
    ``anthropic/claude-haiku-3-5-20241022``).

    Args:
        spec: The model specification.

    Returns:
        A litellm-compatible model identifier string.

    Raises:
        ModelRoutingError: If the provider is not supported.
    """
    if spec.provider not in _SUPPORTED_PROVIDERS:
        raise ModelRoutingError(f"Unsupported provider: {spec.provider!r}")
    prefix = _PROVIDER_PREFIX[spec.provider]
    return f"{prefix}/{spec.model_id}"


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response text.

    Handles cases where JSON is wrapped in markdown code fences or
    surrounded by explanation text.

    Args:
        text: Raw LLM response content.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = text.strip()

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Extract from code fences
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Extract first { ... } block
    brace_match = _JSON_BRACE_RE.search(text)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


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

    def get_tier_for_node(self, node_name: str) -> ModelTier:
        """Return the recommended model tier for a graph node.

        Args:
            node_name: The graph node name.

        Returns:
            The model tier for the node (defaults to SMART).
        """
        return NODE_TIER_MAP.get(node_name, ModelTier.SMART)

    def get_model(self, tier: ModelTier) -> str:
        """Get the litellm model identifier for the primary model in a tier.

        Args:
            tier: The model tier.

        Returns:
            A litellm model identifier string.

        Raises:
            ModelRoutingError: If no models are available for the tier.
        """
        chain = self.chains.get(tier, [])
        if not chain:
            raise ModelRoutingError(f"No models configured for tier {tier.value}")
        return _resolve_litellm_model(chain[0])

    @staticmethod
    async def _call_with_retry(
        model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Call litellm.acompletion with tenacity retry.

        Args:
            model_id: The litellm model identifier.
            messages: Chat messages to send.
            **kwargs: Additional keyword arguments for litellm.

        Returns:
            The litellm ModelResponse.

        Raises:
            RetryError: If all retry attempts fail.
        """
        import litellm

        @retry(
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(min=_BACKOFF_MIN_SECONDS, max=_BACKOFF_MAX_SECONDS),
            reraise=False,
        )
        async def _do_call() -> Any:
            return await litellm.acompletion(
                model=model_id,
                messages=messages,
                **kwargs,
            )

        return await _do_call()

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
            **kwargs: Additional keyword arguments for litellm.

        Returns:
            The litellm ModelResponse.

        Raises:
            ModelRoutingError: If all models in the chain fail.
        """
        chain = self.chains.get(tier, [])
        if not chain:
            raise ModelRoutingError(f"No models configured for tier {tier.value}")

        errors: list[tuple[str, Exception]] = []

        for spec in chain:
            cache_key = f"{spec.provider}:{spec.model_id}"

            try:
                model_id = _resolve_litellm_model(spec)
            except ModelRoutingError:
                raise
            except Exception as exc:
                logger.warning(
                    "model_resolution_failed",
                    provider=spec.provider,
                    model_id=spec.model_id,
                    error=str(exc),
                )
                errors.append((cache_key, exc))
                continue

            call_kwargs = {
                "max_tokens": spec.max_tokens,
                "temperature": spec.temperature,
                **kwargs,
            }

            try:
                result = await self._call_with_retry(model_id, messages, **call_kwargs)
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
            **kwargs: Additional keyword arguments for litellm.

        Returns:
            The litellm ModelResponse.

        Raises:
            ModelRoutingError: If all models in the resolved tier's chain fail.
        """
        tier = self.get_tier_for_node(node_name)
        return await self.invoke_with_fallback(tier, messages, **kwargs)
