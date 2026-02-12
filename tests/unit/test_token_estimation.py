"""Unit tests for research_agent.token_estimation - tiktoken integration."""

from __future__ import annotations

from research_agent.token_estimation import (
    _DEFAULT_ENCODING,
    TokenEstimationTracker,
    _encoding_for_model,
    count_message_tokens,
    count_tokens,
    estimate_call_tokens,
    get_tokenizer,
)

# ---------------------------------------------------------------------------
# Tokenizer selection
# ---------------------------------------------------------------------------


class TestEncodingForModel:
    """_encoding_for_model selects the right tokenizer."""

    def test_claude_uses_cl100k(self) -> None:
        assert _encoding_for_model("claude-sonnet-4-5-20250929") == "cl100k_base"

    def test_claude_haiku_uses_cl100k(self) -> None:
        assert _encoding_for_model("claude-haiku-3-5-20241022") == "cl100k_base"

    def test_gpt4o_uses_o200k(self) -> None:
        assert _encoding_for_model("gpt-4o") == "o200k_base"

    def test_gpt4o_mini_uses_o200k(self) -> None:
        assert _encoding_for_model("gpt-4o-mini") == "o200k_base"

    def test_gpt4_uses_cl100k(self) -> None:
        assert _encoding_for_model("gpt-4-turbo") == "cl100k_base"

    def test_unknown_model_uses_default(self) -> None:
        assert _encoding_for_model("some-unknown-model") == _DEFAULT_ENCODING


class TestGetTokenizer:
    """get_tokenizer returns cached tiktoken encodings."""

    def test_returns_encoding(self) -> None:
        enc = get_tokenizer("claude-sonnet-4-5-20250929")
        assert enc is not None

    def test_default_model(self) -> None:
        enc = get_tokenizer()
        assert enc is not None

    def test_cached_across_calls(self) -> None:
        enc1 = get_tokenizer("gpt-4o")
        enc2 = get_tokenizer("gpt-4o")
        assert enc1 is enc2

    def test_different_models_different_tokenizers(self) -> None:
        enc_cl100k = get_tokenizer("claude-sonnet-4-5-20250929")
        enc_o200k = get_tokenizer("gpt-4o")
        assert enc_cl100k.name != enc_o200k.name


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    """count_tokens returns accurate token counts."""

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_single_word(self) -> None:
        tokens = count_tokens("hello")
        assert tokens >= 1

    def test_longer_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        tokens = count_tokens(text)
        assert tokens > 5

    def test_consistent_results(self) -> None:
        text = "Test consistency"
        t1 = count_tokens(text)
        t2 = count_tokens(text)
        assert t1 == t2

    def test_model_specific_counting(self) -> None:
        text = "Hello world"
        claude_tokens = count_tokens(text, "claude-sonnet-4-5-20250929")
        assert claude_tokens >= 1

    def test_caching_works(self) -> None:
        # Calling with same text/model should hit cache
        text = "Cache test string for token counting"
        t1 = count_tokens(text, "claude-sonnet-4-5-20250929")
        t2 = count_tokens(text, "claude-sonnet-4-5-20250929")
        assert t1 == t2


# ---------------------------------------------------------------------------
# Message token counting
# ---------------------------------------------------------------------------


class TestCountMessageTokens:
    """count_message_tokens estimates chat message tokens."""

    def test_empty_messages(self) -> None:
        tokens = count_message_tokens([])
        # Just the reply priming overhead
        assert tokens == 2

    def test_single_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        tokens = count_message_tokens(messages)
        # Content tokens + overhead (4 per message + 2 reply priming)
        assert tokens > 4

    def test_multiple_messages(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = count_message_tokens(messages)
        single = count_message_tokens([messages[0]])
        assert tokens > single

    def test_model_specific(self) -> None:
        messages = [{"role": "user", "content": "Test"}]
        tokens = count_message_tokens(messages, "claude-sonnet-4-5-20250929")
        assert tokens > 0

    def test_missing_content_key(self) -> None:
        messages = [{"role": "user"}]
        tokens = count_message_tokens(messages)
        # Should handle gracefully (empty content)
        assert tokens >= 6  # 4 overhead + 0 content + 2 reply


# ---------------------------------------------------------------------------
# Call token estimation
# ---------------------------------------------------------------------------


class TestEstimateCallTokens:
    """estimate_call_tokens provides pre-call budget estimates."""

    def test_returns_all_keys(self) -> None:
        result = estimate_call_tokens(
            system_prompt="You are a researcher.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_tokens" in result

    def test_output_tokens_default(self) -> None:
        result = estimate_call_tokens(
            system_prompt="Test",
            messages=[],
        )
        assert result["output_tokens"] == 500

    def test_custom_output_tokens(self) -> None:
        result = estimate_call_tokens(
            system_prompt="Test",
            messages=[],
            estimated_output_tokens=1000,
        )
        assert result["output_tokens"] == 1000

    def test_total_is_sum(self) -> None:
        result = estimate_call_tokens(
            system_prompt="System prompt text.",
            messages=[{"role": "user", "content": "Question"}],
        )
        assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]

    def test_larger_prompt_more_tokens(self) -> None:
        small = estimate_call_tokens(
            system_prompt="Short.",
            messages=[],
        )
        large = estimate_call_tokens(
            system_prompt="This is a much longer system prompt with detailed instructions.",
            messages=[],
        )
        assert large["input_tokens"] > small["input_tokens"]

    def test_model_specific(self) -> None:
        result = estimate_call_tokens(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o",
        )
        assert result["input_tokens"] > 0


# ---------------------------------------------------------------------------
# TokenEstimationTracker
# ---------------------------------------------------------------------------


class TestTokenEstimationTrackerInit:
    """Tracker initialization."""

    def test_starts_empty(self) -> None:
        tracker = TokenEstimationTracker()
        assert tracker.total_estimates == 0

    def test_initial_mean_error(self) -> None:
        tracker = TokenEstimationTracker()
        assert tracker.mean_error_percent == 0.0

    def test_initial_absolute_error(self) -> None:
        tracker = TokenEstimationTracker()
        assert tracker.mean_absolute_error_percent == 0.0


class TestTokenEstimationTrackerRecord:
    """record() tracks estimate/actual pairs."""

    def test_increments_count(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=100, actual=95)
        assert tracker.total_estimates == 1

    def test_multiple_records(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=100, actual=95)
        tracker.record(estimated=200, actual=210)
        assert tracker.total_estimates == 2


class TestTokenEstimationTrackerAccuracy:
    """Accuracy metrics for estimation tracking."""

    def test_perfect_estimate(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=100, actual=100)
        assert tracker.mean_error_percent == 0.0
        assert tracker.mean_absolute_error_percent == 0.0

    def test_overestimate(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=110, actual=100)
        # (110-100)/100 = 10%
        assert abs(tracker.mean_error_percent - 10.0) < 0.01

    def test_underestimate(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=90, actual=100)
        # (90-100)/100 = -10%
        assert abs(tracker.mean_error_percent - (-10.0)) < 0.01

    def test_mean_error_with_mixed_estimates(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=110, actual=100)  # +10%
        tracker.record(estimated=90, actual=100)   # -10%
        # Mean: (10 + -10) / 2 = 0%
        assert abs(tracker.mean_error_percent) < 0.01

    def test_mean_absolute_error_with_mixed(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=110, actual=100)  # |10%|
        tracker.record(estimated=90, actual=100)   # |10%|
        # Mean absolute: (10 + 10) / 2 = 10%
        assert abs(tracker.mean_absolute_error_percent - 10.0) < 0.01

    def test_zero_actual_handled(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=100, actual=0)
        # Should not divide by zero
        assert tracker.mean_error_percent == 0.0


class TestTokenEstimationTrackerSummary:
    """summary() returns a dict of accuracy stats."""

    def test_empty_summary(self) -> None:
        tracker = TokenEstimationTracker()
        summary = tracker.summary()
        assert summary["total_estimates"] == 0
        assert summary["mean_error_percent"] == 0.0
        assert summary["mean_absolute_error_percent"] == 0.0

    def test_populated_summary(self) -> None:
        tracker = TokenEstimationTracker()
        tracker.record(estimated=105, actual=100)
        tracker.record(estimated=200, actual=190)
        summary = tracker.summary()
        assert summary["total_estimates"] == 2
        assert isinstance(summary["mean_error_percent"], float)
        assert isinstance(summary["mean_absolute_error_percent"], float)

    def test_summary_keys(self) -> None:
        tracker = TokenEstimationTracker()
        summary = tracker.summary()
        expected_keys = {"total_estimates", "mean_error_percent", "mean_absolute_error_percent"}
        assert set(summary.keys()) == expected_keys
