"""Unit tests for research_agent.models - three-tier model router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from research_agent.exceptions import ModelRoutingError
from research_agent.models import (
    DEFAULT_CHAINS,
    NODE_TIER_MAP,
    ModelRouter,
    ModelSpec,
    ModelTier,
    _create_chat_model,
)

# ---------------------------------------------------------------------------
# TestModelTier
# ---------------------------------------------------------------------------


class TestModelTier:
    """ModelTier enum defines three routing tiers."""

    def test_has_three_tiers(self) -> None:
        assert len(ModelTier) == 3

    def test_fast_tier(self) -> None:
        assert ModelTier.FAST == "FAST"

    def test_smart_tier(self) -> None:
        assert ModelTier.SMART == "SMART"

    def test_strategic_tier(self) -> None:
        assert ModelTier.STRATEGIC == "STRATEGIC"


# ---------------------------------------------------------------------------
# TestModelSpec
# ---------------------------------------------------------------------------


class TestModelSpec:
    """ModelSpec validates model configuration."""

    def test_basic_spec(self) -> None:
        spec = ModelSpec(provider="anthropic", model_id="claude-haiku-3-5-20241022")
        assert spec.provider == "anthropic"
        assert spec.max_tokens == 4096
        assert spec.temperature == 0.1

    def test_custom_spec(self) -> None:
        spec = ModelSpec(
            provider="openai",
            model_id="gpt-4o",
            max_tokens=8192,
            temperature=0.7,
        )
        assert spec.max_tokens == 8192
        assert spec.temperature == 0.7

    def test_rejects_invalid_temperature(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to 2"):
            ModelSpec(provider="anthropic", model_id="test", temperature=3.0)

    def test_rejects_zero_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            ModelSpec(provider="anthropic", model_id="test", max_tokens=0)


# ---------------------------------------------------------------------------
# TestDefaultChains
# ---------------------------------------------------------------------------


class TestDefaultChains:
    """DEFAULT_CHAINS has entries for all three tiers."""

    def test_all_tiers_present(self) -> None:
        for tier in ModelTier:
            assert tier in DEFAULT_CHAINS

    def test_each_tier_has_fallback(self) -> None:
        for tier in ModelTier:
            assert len(DEFAULT_CHAINS[tier]) >= 2

    def test_fast_tier_primary_is_haiku(self) -> None:
        primary = DEFAULT_CHAINS[ModelTier.FAST][0]
        assert "haiku" in primary.model_id

    def test_smart_tier_primary_is_sonnet(self) -> None:
        primary = DEFAULT_CHAINS[ModelTier.SMART][0]
        assert "sonnet" in primary.model_id

    def test_strategic_tier_has_high_max_tokens(self) -> None:
        primary = DEFAULT_CHAINS[ModelTier.STRATEGIC][0]
        assert primary.max_tokens == 8192


# ---------------------------------------------------------------------------
# TestNodeTierMap
# ---------------------------------------------------------------------------


class TestNodeTierMap:
    """NODE_TIER_MAP routes graph nodes to tiers."""

    def test_plan_uses_smart(self) -> None:
        assert NODE_TIER_MAP["plan"] == ModelTier.SMART

    def test_search_uses_fast(self) -> None:
        assert NODE_TIER_MAP["search"] == ModelTier.FAST

    def test_scrape_uses_fast(self) -> None:
        assert NODE_TIER_MAP["scrape"] == ModelTier.FAST

    def test_summarize_uses_smart(self) -> None:
        assert NODE_TIER_MAP["summarize"] == ModelTier.SMART

    def test_synthesize_uses_strategic(self) -> None:
        assert NODE_TIER_MAP["synthesize"] == ModelTier.STRATEGIC


# ---------------------------------------------------------------------------
# TestCreateChatModel
# ---------------------------------------------------------------------------


class TestCreateChatModel:
    """_create_chat_model instantiates provider-specific models."""

    @patch("langchain_anthropic.ChatAnthropic")
    def test_creates_anthropic_model(self, mock_cls: MagicMock) -> None:
        spec = ModelSpec(provider="anthropic", model_id="claude-haiku-3-5-20241022")
        _create_chat_model(spec)
        mock_cls.assert_called_once_with(
            model="claude-haiku-3-5-20241022",
            max_tokens=4096,
            temperature=0.1,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_creates_openai_model(self, mock_cls: MagicMock) -> None:
        spec = ModelSpec(provider="openai", model_id="gpt-4o-mini")
        _create_chat_model(spec)
        mock_cls.assert_called_once_with(
            model="gpt-4o-mini",
            max_tokens=4096,
            temperature=0.1,
        )

    @patch("langchain_anthropic.ChatAnthropic")
    def test_passes_custom_params(self, mock_cls: MagicMock) -> None:
        spec = ModelSpec(
            provider="anthropic",
            model_id="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            temperature=0.5,
        )
        _create_chat_model(spec)
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            temperature=0.5,
        )

    def test_unsupported_provider_raises(self) -> None:
        spec = ModelSpec(provider="unsupported", model_id="test-model")
        with pytest.raises(ModelRoutingError, match="Unsupported provider"):
            _create_chat_model(spec)


# ---------------------------------------------------------------------------
# TestModelRouterInit
# ---------------------------------------------------------------------------


class TestModelRouterInit:
    """ModelRouter initializes with default or custom chains."""

    def test_default_chains(self) -> None:
        router = ModelRouter()
        assert ModelTier.FAST in router.chains
        assert ModelTier.SMART in router.chains
        assert ModelTier.STRATEGIC in router.chains

    def test_custom_chains(self) -> None:
        custom = {
            ModelTier.FAST: [
                ModelSpec(provider="openai", model_id="gpt-4o-mini"),
            ]
        }
        router = ModelRouter(chains=custom)
        assert router.chains == custom

    def test_empty_model_cache(self) -> None:
        router = ModelRouter()
        assert len(router._model_cache) == 0


# ---------------------------------------------------------------------------
# TestGetTierForNode
# ---------------------------------------------------------------------------


class TestGetTierForNode:
    """get_tier_for_node resolves node names to tiers."""

    def test_known_node(self) -> None:
        router = ModelRouter()
        assert router.get_tier_for_node("synthesize") == ModelTier.STRATEGIC

    def test_unknown_node_defaults_to_smart(self) -> None:
        router = ModelRouter()
        assert router.get_tier_for_node("unknown_node") == ModelTier.SMART


# ---------------------------------------------------------------------------
# TestGetModel
# ---------------------------------------------------------------------------


class TestGetModel:
    """get_model returns cached model instances."""

    @patch("langchain_anthropic.ChatAnthropic")
    def test_returns_model(self, mock_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        router = ModelRouter()
        model = router.get_model(ModelTier.FAST)
        assert model is mock_instance

    @patch("langchain_anthropic.ChatAnthropic")
    def test_caches_model(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        router = ModelRouter()
        model1 = router.get_model(ModelTier.FAST)
        model2 = router.get_model(ModelTier.FAST)
        assert model1 is model2
        mock_cls.assert_called_once()

    def test_empty_chain_raises(self) -> None:
        router = ModelRouter(chains={ModelTier.FAST: []})
        with pytest.raises(ModelRoutingError, match="No models configured"):
            router.get_model(ModelTier.FAST)

    def test_missing_tier_raises(self) -> None:
        router = ModelRouter(
            chains={ModelTier.SMART: [ModelSpec(provider="anthropic", model_id="x")]}
        )
        with pytest.raises(ModelRoutingError, match="No models configured"):
            router.get_model(ModelTier.FAST)


# ---------------------------------------------------------------------------
# TestInvokeWithFallback
# ---------------------------------------------------------------------------


class TestInvokeWithFallback:
    """invoke_with_fallback tries models with retry and fallback."""

    @pytest.mark.asyncio
    async def test_success_on_first_model(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value="response")

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="test-model"),
                ]
            }
        )
        router._model_cache["anthropic:test-model"] = mock_model

        result = await router.invoke_with_fallback(
            ModelTier.FAST, [{"role": "user", "content": "hello"}]
        )
        assert result == "response"
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_second_model(self) -> None:
        failing_model = AsyncMock()
        failing_model.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        success_model = AsyncMock()
        success_model.ainvoke = AsyncMock(return_value="fallback response")

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="primary"),
                    ModelSpec(provider="openai", model_id="fallback"),
                ]
            }
        )
        router._model_cache["anthropic:primary"] = failing_model
        router._model_cache["openai:fallback"] = success_model

        result = await router.invoke_with_fallback(
            ModelTier.FAST, [{"role": "user", "content": "hello"}]
        )
        assert result == "fallback response"

    @pytest.mark.asyncio
    async def test_all_models_fail_raises(self) -> None:
        failing_model = AsyncMock()
        failing_model.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="model-a"),
                ]
            }
        )
        router._model_cache["anthropic:model-a"] = failing_model

        with pytest.raises(ModelRoutingError, match="All models in FAST chain failed"):
            await router.invoke_with_fallback(
                ModelTier.FAST, [{"role": "user", "content": "hello"}]
            )

    @pytest.mark.asyncio
    async def test_empty_chain_raises(self) -> None:
        router = ModelRouter(chains={ModelTier.FAST: []})
        with pytest.raises(ModelRoutingError, match="No models configured"):
            await router.invoke_with_fallback(
                ModelTier.FAST, [{"role": "user", "content": "hello"}]
            )

    @pytest.mark.asyncio
    async def test_passes_kwargs_to_model(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value="ok")

        router = ModelRouter(
            chains={
                ModelTier.SMART: [
                    ModelSpec(provider="anthropic", model_id="test"),
                ]
            }
        )
        router._model_cache["anthropic:test"] = mock_model

        await router.invoke_with_fallback(
            ModelTier.SMART,
            [{"role": "user", "content": "hello"}],
            stop=["END"],
        )
        mock_model.ainvoke.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            stop=["END"],
        )

    @pytest.mark.asyncio
    async def test_instantiates_model_on_cache_miss(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value="ok")

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="haiku"),
                ]
            }
        )

        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_model):
            result = await router.invoke_with_fallback(
                ModelTier.FAST, [{"role": "user", "content": "hello"}]
            )

        assert result == "ok"
        assert "anthropic:haiku" in router._model_cache


# ---------------------------------------------------------------------------
# TestInvokeWithRetry
# ---------------------------------------------------------------------------


class TestInvokeWithRetry:
    """_invoke_with_retry retries on transient failures."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value="ok")

        result = await ModelRouter._invoke_with_retry(
            mock_model, [{"role": "user", "content": "test"}]
        )
        assert result == "ok"
        assert mock_model.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(
            side_effect=[RuntimeError("transient"), "recovered"]
        )

        result = await ModelRouter._invoke_with_retry(
            mock_model, [{"role": "user", "content": "test"}]
        )
        assert result == "recovered"
        assert mock_model.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("persistent failure"))

        with pytest.raises(RetryError):
            await ModelRouter._invoke_with_retry(
                mock_model, [{"role": "user", "content": "test"}]
            )
        assert mock_model.ainvoke.call_count == 3


# ---------------------------------------------------------------------------
# TestInvokeForNode
# ---------------------------------------------------------------------------


class TestInvokeForNode:
    """invoke_for_node resolves tier and delegates to invoke_with_fallback."""

    @pytest.mark.asyncio
    async def test_delegates_to_invoke_with_fallback(self) -> None:
        router = ModelRouter()
        router.invoke_with_fallback = AsyncMock(return_value="result")

        result = await router.invoke_for_node(
            "synthesize", [{"role": "user", "content": "hello"}]
        )
        assert result == "result"
        router.invoke_with_fallback.assert_called_once_with(
            ModelTier.STRATEGIC,
            [{"role": "user", "content": "hello"}],
        )

    @pytest.mark.asyncio
    async def test_unknown_node_uses_smart(self) -> None:
        router = ModelRouter()
        router.invoke_with_fallback = AsyncMock(return_value="result")

        await router.invoke_for_node(
            "custom_node", [{"role": "user", "content": "test"}]
        )
        router.invoke_with_fallback.assert_called_once_with(
            ModelTier.SMART,
            [{"role": "user", "content": "test"}],
        )

    @pytest.mark.asyncio
    async def test_passes_kwargs(self) -> None:
        router = ModelRouter()
        router.invoke_with_fallback = AsyncMock(return_value="ok")

        await router.invoke_for_node(
            "plan",
            [{"role": "user", "content": "test"}],
            stop=["END"],
        )
        router.invoke_with_fallback.assert_called_once_with(
            ModelTier.SMART,
            [{"role": "user", "content": "test"}],
            stop=["END"],
        )
