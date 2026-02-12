"""Token estimation using tiktoken for accurate pre-call budget checks.

Provides model-specific tokenizer selection, token counting with
caching for repeated content, and actual-vs-estimated tracking.
"""

from __future__ import annotations

import functools
from typing import Any

import structlog
import tiktoken

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer selection
# ---------------------------------------------------------------------------

# Mapping from model ID prefixes to tiktoken encoding names.
# cl100k_base covers GPT-4, GPT-3.5-turbo, and text-embedding models.
# o200k_base covers GPT-4o and newer OpenAI models.
# Claude models don't have a public tokenizer; cl100k_base provides
# a reasonable approximation (~5% margin on average).
_MODEL_ENCODING_MAP: dict[str, str] = {
    "claude": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
}

_DEFAULT_ENCODING = "cl100k_base"


def _encoding_for_model(model: str) -> str:
    """Determine the tiktoken encoding name for a model.

    Args:
        model: Model identifier string.

    Returns:
        Tiktoken encoding name.
    """
    for prefix, encoding in _MODEL_ENCODING_MAP.items():
        if model.startswith(prefix):
            return encoding
    return _DEFAULT_ENCODING


@functools.lru_cache(maxsize=8)
def get_tokenizer(model: str = "") -> tiktoken.Encoding:
    """Return the tiktoken tokenizer for a model, cached for reuse.

    Args:
        model: Model identifier. If empty, uses the default encoding.

    Returns:
        A tiktoken Encoding instance.
    """
    encoding_name = _encoding_for_model(model) if model else _DEFAULT_ENCODING
    return tiktoken.get_encoding(encoding_name)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in a text string using tiktoken.

    Results are cached by (text, model) for repeated content.

    Args:
        text: The text to tokenize.
        model: Model identifier for tokenizer selection.

    Returns:
        Number of tokens.
    """
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def count_message_tokens(
    messages: list[dict[str, Any]],
    model: str = "",
) -> int:
    """Estimate token count for a list of chat messages.

    Includes per-message overhead tokens for role markers and separators.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model identifier for tokenizer selection.

    Returns:
        Estimated total token count.
    """
    # Per-message overhead: role marker + separator tokens
    per_message_overhead = 4  # <|im_start|>role\ncontent<|im_end|>
    total = 0
    for msg in messages:
        total += per_message_overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
    # Final reply priming
    total += 2
    return total


def estimate_call_tokens(
    system_prompt: str,
    messages: list[dict[str, Any]],
    model: str = "",
    estimated_output_tokens: int = 500,
) -> dict[str, int]:
    """Estimate total tokens for a prospective LLM call.

    Useful for budget pre-checks before making the actual call.

    Args:
        system_prompt: The system prompt text.
        messages: Conversation messages.
        model: Model identifier.
        estimated_output_tokens: Expected output tokens (default 500).

    Returns:
        Dict with 'input_tokens', 'output_tokens', and 'total_tokens'.
    """
    system_tokens = count_tokens(system_prompt, model) + 4  # overhead
    message_tokens = count_message_tokens(messages, model)
    input_tokens = system_tokens + message_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": estimated_output_tokens,
        "total_tokens": input_tokens + estimated_output_tokens,
    }


# ---------------------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------------------


class TokenEstimationTracker:
    """Tracks accuracy of token estimates vs actual usage.

    Records each estimate/actual pair and computes aggregate accuracy
    metrics for monitoring estimation drift.

    Attributes:
        total_estimates: Number of tracked estimates.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, int]] = []

    @property
    def total_estimates(self) -> int:
        """Return the number of tracked estimate/actual pairs.

        Returns:
            Count of tracked records.
        """
        return len(self._records)

    def record(self, estimated: int, actual: int) -> None:
        """Record an estimate/actual token pair.

        Args:
            estimated: Pre-call token estimate.
            actual: Post-call actual token count.
        """
        self._records.append({"estimated": estimated, "actual": actual})
        logger.debug(
            "token_estimate_recorded",
            estimated=estimated,
            actual=actual,
            error_pct=self._error_pct(estimated, actual),
        )

    @staticmethod
    def _error_pct(estimated: int, actual: int) -> float:
        """Calculate percentage error.

        Args:
            estimated: Estimated value.
            actual: Actual value.

        Returns:
            Percentage error (positive = overestimate).
        """
        if actual == 0:
            return 0.0
        return ((estimated - actual) / actual) * 100.0

    @property
    def mean_error_percent(self) -> float:
        """Calculate the mean percentage error across all records.

        Positive means estimates tend to overcount; negative means undercount.

        Returns:
            Mean percentage error, or 0.0 if no records.
        """
        if not self._records:
            return 0.0
        total = sum(
            self._error_pct(r["estimated"], r["actual"])
            for r in self._records
        )
        return total / len(self._records)

    @property
    def mean_absolute_error_percent(self) -> float:
        """Calculate the mean absolute percentage error.

        Returns:
            Mean absolute percentage error, or 0.0 if no records.
        """
        if not self._records:
            return 0.0
        total = sum(
            abs(self._error_pct(r["estimated"], r["actual"]))
            for r in self._records
        )
        return total / len(self._records)

    def summary(self) -> dict[str, Any]:
        """Return a summary of estimation accuracy.

        Returns:
            Dict with total_estimates, mean_error_percent,
            and mean_absolute_error_percent.
        """
        return {
            "total_estimates": self.total_estimates,
            "mean_error_percent": round(self.mean_error_percent, 2),
            "mean_absolute_error_percent": round(self.mean_absolute_error_percent, 2),
        }
