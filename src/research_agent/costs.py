"""Budget tracking and degradation management.

Implements a 4-tier state machine (FULL -> REDUCED -> CACHED -> PARTIAL)
that progressively degrades capabilities as the cost budget is consumed,
along with model fallback chains.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DegradationTier(StrEnum):
    """Budget degradation tiers (ordered by capability)."""

    FULL = "FULL"
    REDUCED = "REDUCED"
    CACHED = "CACHED"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
# Cost tracking models
# ---------------------------------------------------------------------------


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call for cost tracking."""

    model: str = Field(description="Model identifier.")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    step_name: str = Field(default="", description="Graph node that made the call.")


class BudgetStatus(BaseModel):
    """Current budget consumption status."""

    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_llm_calls: int = Field(default=0, ge=0)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    budget_remaining_usd: float = Field(ge=0.0)
    budget_used_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    current_tier: DegradationTier = DegradationTier.FULL


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Approximate pricing per 1M tokens (input, output) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# Tier transition thresholds (percentage of max budget consumed)
_TIER_THRESHOLDS: dict[DegradationTier, float] = {
    DegradationTier.FULL: 0.0,
    DegradationTier.REDUCED: 60.0,
    DegradationTier.CACHED: 80.0,
    DegradationTier.PARTIAL: 95.0,
}


# ---------------------------------------------------------------------------
# Budget Tracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """Tracks cumulative cost of LLM calls within a research run.

    Attributes:
        max_cost_usd: Maximum allowed cost in USD.
        max_llm_calls: Maximum allowed LLM calls.
        warn_at_percent: Emit a warning when usage exceeds this percentage.
    """

    def __init__(
        self,
        max_cost_usd: float = 2.00,
        max_llm_calls: int = 50,
        warn_at_percent: int = 80,
    ) -> None:
        """Initialize the budget tracker.

        Args:
            max_cost_usd: Maximum cost budget in USD.
            max_llm_calls: Maximum number of LLM calls.
            warn_at_percent: Warning threshold as a percentage.
        """
        self.max_cost_usd = max_cost_usd
        self.max_llm_calls = max_llm_calls
        self.warn_at_percent = warn_at_percent
        self._records: list[LLMCallRecord] = []
        self._warned = False

    @property
    def total_cost(self) -> float:
        """Return total accumulated cost in USD.

        Returns:
            Total cost.
        """
        return sum(r.cost_usd for r in self._records)

    @property
    def total_calls(self) -> int:
        """Return total number of LLM calls.

        Returns:
            Call count.
        """
        return len(self._records)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a prospective LLM call.

        Args:
            model: Model identifier.
            input_tokens: Estimated input token count.
            output_tokens: Estimated output token count.

        Returns:
            Estimated cost in USD.
        """
        input_price, output_price = MODEL_PRICING.get(model, (5.0, 15.0))
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def record_call(self, record: LLMCallRecord) -> None:
        """Record an LLM call and check budget thresholds.

        Args:
            record: The call record to track.

        Raises:
            BudgetExhaustedError: If the budget is fully consumed.
        """
        self._records.append(record)

        pct = (
            (self.total_cost / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0
        )
        if pct >= self.warn_at_percent and not self._warned:
            self._warned = True
            logger.warning(
                "budget_warning",
                used_percent=round(pct, 1),
                total_cost=round(self.total_cost, 4),
                max_cost=self.max_cost_usd,
            )

        if self.total_cost >= self.max_cost_usd:
            raise BudgetExhaustedError(
                f"Budget exhausted: ${self.total_cost:.4f} >= ${self.max_cost_usd:.2f}"
            )

        if self.total_calls >= self.max_llm_calls:
            raise BudgetExhaustedError(
                f"LLM call limit reached: {self.total_calls} >= {self.max_llm_calls}"
            )

    def status(self) -> BudgetStatus:
        """Return current budget status.

        Returns:
            A ``BudgetStatus`` snapshot.
        """
        pct = (
            (self.total_cost / self.max_cost_usd * 100) if self.max_cost_usd > 0 else 0
        )
        remaining = max(0.0, self.max_cost_usd - self.total_cost)
        return BudgetStatus(
            total_cost_usd=round(self.total_cost, 4),
            total_llm_calls=self.total_calls,
            total_input_tokens=sum(r.input_tokens for r in self._records),
            total_output_tokens=sum(r.output_tokens for r in self._records),
            budget_remaining_usd=round(remaining, 4),
            budget_used_percent=round(min(pct, 100.0), 1),
            current_tier=self._current_tier(pct),
        )

    @staticmethod
    def _current_tier(used_percent: float) -> DegradationTier:
        """Determine the current degradation tier.

        Args:
            used_percent: Percentage of budget consumed.

        Returns:
            The active degradation tier.
        """
        if used_percent >= _TIER_THRESHOLDS[DegradationTier.PARTIAL]:
            return DegradationTier.PARTIAL
        if used_percent >= _TIER_THRESHOLDS[DegradationTier.CACHED]:
            return DegradationTier.CACHED
        if used_percent >= _TIER_THRESHOLDS[DegradationTier.REDUCED]:
            return DegradationTier.REDUCED
        return DegradationTier.FULL


# ---------------------------------------------------------------------------
# Degradation Manager
# ---------------------------------------------------------------------------


class DegradationManager:
    """Manages capability degradation based on budget tier.

    Provides model fallback chains and feature flags per tier.

    Attributes:
        tracker: The underlying budget tracker.
    """

    # Model fallback chain per tier
    MODEL_CHAINS: ClassVar[dict[DegradationTier, list[str]]] = {
        DegradationTier.FULL: [
            "claude-sonnet-4-5-20250929",
            "gpt-4o",
        ],
        DegradationTier.REDUCED: [
            "claude-haiku-3-5-20241022",
            "gpt-4o-mini",
        ],
        DegradationTier.CACHED: [
            "claude-haiku-3-5-20241022",
        ],
        DegradationTier.PARTIAL: [
            "gpt-4o-mini",
        ],
    }

    def __init__(self, tracker: BudgetTracker) -> None:
        """Initialize the degradation manager.

        Args:
            tracker: Budget tracker instance.
        """
        self.tracker = tracker

    @property
    def tier(self) -> DegradationTier:
        """Return the current degradation tier.

        Returns:
            Active tier.
        """
        return self.tracker.status().current_tier

    def get_model(self) -> str:
        """Return the preferred model for the current tier.

        Returns:
            Model identifier string.
        """
        chain = self.MODEL_CHAINS.get(
            self.tier, self.MODEL_CHAINS[DegradationTier.FULL]
        )
        return chain[0]

    def get_fallback_chain(self) -> list[str]:
        """Return the full model fallback chain for the current tier.

        Returns:
            Ordered list of model identifiers (preferred first).
        """
        return list(
            self.MODEL_CHAINS.get(self.tier, self.MODEL_CHAINS[DegradationTier.FULL])
        )

    def should_skip_search(self) -> bool:
        """Whether to skip new web searches (use cache only).

        Returns:
            True if in CACHED or PARTIAL tier.
        """
        return self.tier in {DegradationTier.CACHED, DegradationTier.PARTIAL}

    def should_skip_scraping(self) -> bool:
        """Whether to skip web scraping (use cached content only).

        Returns:
            True if in PARTIAL tier.
        """
        return self.tier == DegradationTier.PARTIAL

    def max_search_results(self) -> int:
        """Return the maximum search results allowed at the current tier.

        Returns:
            Integer limit.
        """
        limits: dict[DegradationTier, int] = {
            DegradationTier.FULL: 10,
            DegradationTier.REDUCED: 5,
            DegradationTier.CACHED: 3,
            DegradationTier.PARTIAL: 0,
        }
        return limits.get(self.tier, 10)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExhaustedError(Exception):
    """Raised when the research run's cost budget is fully consumed."""
