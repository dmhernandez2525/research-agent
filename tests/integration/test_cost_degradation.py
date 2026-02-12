"""Integration tests for cost tracking + degradation tier transitions."""

from __future__ import annotations

import pytest

from research_agent.costs import (
    BudgetExhaustedError,
    BudgetTracker,
    DegradationManager,
    DegradationTier,
    LLMCallRecord,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(
    model: str = "claude-sonnet-4-5-20250929",
    cost: float = 0.01,
    step: str = "search",
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> LLMCallRecord:
    return LLMCallRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        step_name=step,
    )


# ---------------------------------------------------------------------------
# Budget -> Degradation tier transitions
# ---------------------------------------------------------------------------


class TestBudgetTierTransitions:
    """Budget consumption drives degradation tier changes."""

    def test_full_tier_at_start(self) -> None:
        """New tracker starts in FULL tier."""
        tracker = BudgetTracker(max_cost_usd=1.00)
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.FULL

    def test_transition_to_reduced_at_80_percent(self) -> None:
        """At 80% budget usage, tier should be REDUCED."""
        tracker = BudgetTracker(max_cost_usd=1.00, max_llm_calls=200)
        mgr = DegradationManager(tracker)

        # Record calls totaling $0.81
        for _ in range(81):
            tracker.record_call(_make_call(cost=0.01))

        assert mgr.tier == DegradationTier.REDUCED

    def test_transition_to_cached_at_95_percent(self) -> None:
        """At 95% budget usage, tier should be CACHED."""
        tracker = BudgetTracker(max_cost_usd=1.00, max_llm_calls=200)
        mgr = DegradationManager(tracker)

        # Record calls totaling $0.96
        for _ in range(96):
            tracker.record_call(_make_call(cost=0.01))

        assert mgr.tier == DegradationTier.CACHED

    def test_budget_exhaustion_raises_at_100_percent(self) -> None:
        """Recording calls past max budget should raise BudgetExhaustedError."""
        tracker = BudgetTracker(max_cost_usd=0.10)

        with pytest.raises(BudgetExhaustedError):
            for _ in range(20):
                tracker.record_call(_make_call(cost=0.01))


# ---------------------------------------------------------------------------
# Degradation -> Model selection
# ---------------------------------------------------------------------------


class TestDegradationModelSelection:
    """Degradation tier determines which models are available."""

    def test_full_tier_uses_sonnet(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.00)
        mgr = DegradationManager(tracker)
        model = mgr.get_model()
        assert "sonnet" in model

    def test_reduced_tier_uses_haiku(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.00, max_llm_calls=200)
        mgr = DegradationManager(tracker)

        for _ in range(81):
            tracker.record_call(_make_call(cost=0.01))

        model = mgr.get_model()
        assert "haiku" in model

    def test_cached_tier_skips_search(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.00, max_llm_calls=200)
        mgr = DegradationManager(tracker)

        for _ in range(96):
            tracker.record_call(_make_call(cost=0.01))

        assert mgr.should_skip_search() is True
        assert mgr.should_skip_scraping() is False

    def test_partial_tier_skips_everything(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.00)
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.PARTIAL)

        assert mgr.should_skip_search() is True
        assert mgr.should_skip_scraping() is True
        assert mgr.max_search_results() == 0

    def test_fallback_chain_length_varies_by_tier(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.00)
        mgr = DegradationManager(tracker)

        # FULL tier has 2 models in chain
        chain_full = mgr.get_fallback_chain()
        assert len(chain_full) == 2

        # Force to PARTIAL, only 1 model
        mgr.force_degrade(DegradationTier.PARTIAL)
        chain_partial = mgr.get_fallback_chain()
        assert len(chain_partial) == 1


# ---------------------------------------------------------------------------
# Force degrade and recovery
# ---------------------------------------------------------------------------


class TestForceDegradeAndRecovery:
    """Manual tier forcing and recovery flow."""

    def test_force_degrade_one_level(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.00)
        mgr = DegradationManager(tracker)

        assert mgr.tier == DegradationTier.FULL
        mgr.force_degrade()
        assert mgr.tier == DegradationTier.REDUCED

    def test_force_degrade_to_specific_tier(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.00)
        mgr = DegradationManager(tracker)

        mgr.force_degrade(DegradationTier.CACHED)
        assert mgr.tier == DegradationTier.CACHED

    def test_recovery_succeeds_below_threshold(self) -> None:
        """Recovery should succeed when budget usage is below 75%."""
        tracker = BudgetTracker(max_cost_usd=10.00)
        mgr = DegradationManager(tracker)

        # Force degrade, then recover (budget is at 0%)
        mgr.force_degrade(DegradationTier.REDUCED)
        assert mgr.tier == DegradationTier.REDUCED

        recovered = mgr.try_recover()
        assert recovered is True
        assert mgr.tier == DegradationTier.FULL

    def test_recovery_fails_above_threshold(self) -> None:
        """Recovery should fail when budget is above 75%."""
        tracker = BudgetTracker(max_cost_usd=1.00, max_llm_calls=200)
        mgr = DegradationManager(tracker)

        # Use 76% of budget
        for _ in range(76):
            tracker.record_call(_make_call(cost=0.01))

        mgr.force_degrade(DegradationTier.CACHED)
        recovered = mgr.try_recover()
        assert recovered is False
        assert mgr.tier == DegradationTier.CACHED


# ---------------------------------------------------------------------------
# Cost tracking across steps
# ---------------------------------------------------------------------------


class TestCostTrackingAcrossSteps:
    """Cost per step aggregation integrates with budget status."""

    def test_cost_per_step_aggregation(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.00)

        tracker.record_call(_make_call(cost=0.05, step="plan"))
        tracker.record_call(_make_call(cost=0.10, step="search"))
        tracker.record_call(_make_call(cost=0.08, step="search"))
        tracker.record_call(_make_call(cost=0.15, step="summarize"))

        breakdown = tracker.cost_per_step()
        assert breakdown["plan"] == pytest.approx(0.05, abs=1e-6)
        assert breakdown["search"] == pytest.approx(0.18, abs=1e-6)
        assert breakdown["summarize"] == pytest.approx(0.15, abs=1e-6)

    def test_status_reflects_cumulative_calls(self) -> None:
        tracker = BudgetTracker(max_cost_usd=2.00, max_llm_calls=100)

        for _ in range(5):
            tracker.record_call(
                _make_call(cost=0.02, input_tokens=500, output_tokens=200)
            )

        status = tracker.status()
        assert status.total_llm_calls == 5
        assert status.total_cost_usd == pytest.approx(0.10, abs=1e-4)
        assert status.total_input_tokens == 2500
        assert status.total_output_tokens == 1000
        assert status.budget_remaining_usd == pytest.approx(1.90, abs=1e-4)
        assert status.budget_used_percent == pytest.approx(5.0, abs=0.1)

    def test_pre_check_prevents_overspend(self) -> None:
        tracker = BudgetTracker(max_cost_usd=0.05)

        tracker.record_call(_make_call(cost=0.03))
        estimated = tracker.estimate_cost(
            "claude-sonnet-4-5-20250929", input_tokens=10000, output_tokens=5000
        )

        with pytest.raises(BudgetExhaustedError):
            tracker.check_budget(estimated_cost=estimated)
