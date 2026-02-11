"""Unit tests for research_agent.models - routing, fallback chains, tier selection."""

from __future__ import annotations

from typing import Any

import pytest

# TODO: Uncomment once the models module is implemented.
# from research_agent.models import ModelRouter, get_fallback_chain, select_tier


class TestModelRouting:
    """ModelRouter should pick the correct LLM based on task and config."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_default_provider_is_anthropic(self, sample_config: dict[str, Any]) -> None:
        """With default config, the router should select an Anthropic model."""
        # TODO: router = ModelRouter(config=sample_config["llm"])
        #       assert router.provider == "anthropic"

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_openai_provider_selected(self, sample_config: dict[str, Any]) -> None:
        """Setting provider='openai' should route to an OpenAI model."""
        # TODO: Override provider to "openai", construct router, and
        #       verify router.provider == "openai".

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_unknown_provider_raises(self, sample_config: dict[str, Any]) -> None:
        """An unsupported provider should raise ValueError."""
        # TODO: Set provider="unsupported", construct router, assert ValueError.


class TestFallbackChain:
    """Fallback logic should try cheaper/faster models on failure."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_fallback_chain_has_multiple_models(self) -> None:
        """get_fallback_chain should return at least 2 models."""
        # TODO: chain = get_fallback_chain(provider="anthropic")
        #       assert len(chain) >= 2

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_fallback_chain_ordered_by_capability(self) -> None:
        """The first model in the chain should be the most capable."""
        # TODO: Verify chain[0] is the premium model and chain[-1] is
        #       the cheapest/fastest.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_fallback_chain_respects_provider(self) -> None:
        """All models in the chain should belong to the requested provider."""
        # TODO: For each model in get_fallback_chain("anthropic"), verify
        #       it is an Anthropic model identifier.


class TestTierSelection:
    """select_tier should choose a model tier appropriate to the task."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_synthesis_uses_high_tier(self) -> None:
        """The synthesis task should use the highest-capability tier."""
        # TODO: tier = select_tier(task="synthesis")
        #       assert tier == "high"

    @pytest.mark.skip(reason="TODO: Implement once research_agent.models exists")
    def test_search_query_uses_low_tier(self) -> None:
        """Simple search query generation should use a low-cost tier."""
        # TODO: tier = select_tier(task="search_query")
        #       assert tier == "low"
