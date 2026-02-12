"""Unit tests for research_agent.models - three-tier model router with litellm."""

from __future__ import annotations

import json
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
    _extract_json,
    _resolve_litellm_model,
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
# TestResolveLitellmModel
# ---------------------------------------------------------------------------


class TestResolveLitellmModel:
    """_resolve_litellm_model builds provider-prefixed identifiers."""

    def test_anthropic_model(self) -> None:
        spec = ModelSpec(provider="anthropic", model_id="claude-haiku-3-5-20241022")
        assert _resolve_litellm_model(spec) == "anthropic/claude-haiku-3-5-20241022"

    def test_openai_model(self) -> None:
        spec = ModelSpec(provider="openai", model_id="gpt-4o-mini")
        assert _resolve_litellm_model(spec) == "openai/gpt-4o-mini"

    def test_google_model(self) -> None:
        spec = ModelSpec(provider="google", model_id="gemini-pro")
        assert _resolve_litellm_model(spec) == "gemini/gemini-pro"

    def test_unsupported_provider_raises(self) -> None:
        spec = ModelSpec(provider="unsupported", model_id="test-model")
        with pytest.raises(ModelRoutingError, match="Unsupported provider"):
            _resolve_litellm_model(spec)


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


class TestExtractJson:
    """_extract_json parses JSON from various LLM response formats."""

    def test_direct_json(self) -> None:
        data = _extract_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_with_whitespace(self) -> None:
        data = _extract_json('  \n  {"key": "value"}  \n  ')
        assert data == {"key": "value"}

    def test_json_in_code_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        data = _extract_json(text)
        assert data == {"key": "value"}

    def test_json_in_plain_code_fence(self) -> None:
        text = '```\n{"key": "value"}\n```'
        data = _extract_json(text)
        assert data == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        data = _extract_json(text)
        assert data == {"key": "value"}

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": [1, 2, 3]}}'
        data = _extract_json(text)
        assert data["outer"]["inner"] == [1, 2, 3]

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("no json here at all")

    def test_raises_on_json_array(self) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("[1, 2, 3]")

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("")

    def test_complex_nested_json(self) -> None:
        obj = {"title": "Test", "items": [{"a": 1}, {"b": 2}], "count": 5}
        data = _extract_json(json.dumps(obj))
        assert data == obj


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
    """get_model returns litellm model identifier strings."""

    def test_returns_model_id(self) -> None:
        router = ModelRouter()
        model_id = router.get_model(ModelTier.FAST)
        assert model_id == "anthropic/claude-haiku-3-5-20241022"

    def test_returns_correct_tier_model(self) -> None:
        router = ModelRouter()
        model_id = router.get_model(ModelTier.SMART)
        assert "sonnet" in model_id

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
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="test-model"),
                ]
            }
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await router.invoke_with_fallback(
                ModelTier.FAST, [{"role": "user", "content": "hello"}]
            )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_fallback_to_second_model(self) -> None:
        mock_response = MagicMock()

        call_count = 0

        async def mock_acompletion(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if "anthropic" in kwargs.get("model", ""):
                raise RuntimeError("API down")
            return mock_response

        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="primary"),
                    ModelSpec(provider="openai", model_id="fallback"),
                ]
            }
        )

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await router.invoke_with_fallback(
                ModelTier.FAST, [{"role": "user", "content": "hello"}]
            )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_all_models_fail_raises(self) -> None:
        router = ModelRouter(
            chains={
                ModelTier.FAST: [
                    ModelSpec(provider="anthropic", model_id="model-a"),
                ]
            }
        )

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fail"),
            ),
            pytest.raises(ModelRoutingError, match="All models in FAST chain failed"),
        ):
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
    async def test_passes_model_params(self) -> None:
        mock_response = MagicMock()

        router = ModelRouter(
            chains={
                ModelTier.SMART: [
                    ModelSpec(
                        provider="anthropic",
                        model_id="test",
                        max_tokens=2048,
                        temperature=0.5,
                    ),
                ]
            }
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            await router.invoke_with_fallback(
                ModelTier.SMART,
                [{"role": "user", "content": "hello"}],
            )
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["model"] == "anthropic/test"
            assert call_kwargs["max_tokens"] == 2048
            assert call_kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# TestCallWithRetry
# ---------------------------------------------------------------------------


class TestCallWithRetry:
    """_call_with_retry retries on transient failures."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        mock_response = MagicMock()

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await ModelRouter._call_with_retry(
                "anthropic/test", [{"role": "user", "content": "test"}]
            )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self) -> None:
        mock_response = MagicMock()

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("transient"), mock_response],
        ):
            result = await ModelRouter._call_with_retry(
                "anthropic/test", [{"role": "user", "content": "test"}]
            )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("persistent failure"),
            ),
            pytest.raises(RetryError),
        ):
            await ModelRouter._call_with_retry(
                "anthropic/test", [{"role": "user", "content": "test"}]
            )


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
