"""Unit tests for research_agent.costs - budget tracking and cost estimation."""

from __future__ import annotations

import pytest

from research_agent.costs import (
    MODEL_PRICING,
    BudgetExhaustedError,
    BudgetStatus,
    BudgetTracker,
    DegradationManager,
    DegradationTier,
    LLMCallRecord,
)

# ---------------------------------------------------------------------------
# TestLLMCallRecord
# ---------------------------------------------------------------------------


class TestLLMCallRecord:
    """LLMCallRecord validates individual call data."""

    def test_basic_record(self) -> None:
        record = LLMCallRecord(
            model="claude-haiku-3-5-20241022",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            step_name="search",
        )
        assert record.model == "claude-haiku-3-5-20241022"
        assert record.cost_usd == 0.001

    def test_defaults(self) -> None:
        record = LLMCallRecord(model="test")
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.cost_usd == 0.0
        assert record.step_name == ""

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError):
            LLMCallRecord(model="test", input_tokens=-1)

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValueError):
            LLMCallRecord(model="test", cost_usd=-0.01)


# ---------------------------------------------------------------------------
# TestBudgetStatus
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    """BudgetStatus captures a snapshot of budget consumption."""

    def test_defaults(self) -> None:
        status = BudgetStatus(budget_remaining_usd=2.0)
        assert status.total_cost_usd == 0.0
        assert status.total_llm_calls == 0
        assert status.current_tier == DegradationTier.FULL

    def test_full_status(self) -> None:
        status = BudgetStatus(
            total_cost_usd=1.5,
            total_llm_calls=10,
            total_input_tokens=5000,
            total_output_tokens=2000,
            budget_remaining_usd=0.5,
            budget_used_percent=75.0,
            current_tier=DegradationTier.FULL,
        )
        assert status.budget_remaining_usd == 0.5
        assert status.budget_used_percent == 75.0


# ---------------------------------------------------------------------------
# TestModelPricing
# ---------------------------------------------------------------------------


class TestModelPricing:
    """MODEL_PRICING has entries for all supported models."""

    def test_has_anthropic_models(self) -> None:
        assert "claude-sonnet-4-5-20250929" in MODEL_PRICING
        assert "claude-haiku-3-5-20241022" in MODEL_PRICING

    def test_has_openai_models(self) -> None:
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING

    def test_pricing_format(self) -> None:
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert input_price > 0, f"{model} input price should be positive"
            assert output_price > 0, f"{model} output price should be positive"

    def test_haiku_cheaper_than_sonnet(self) -> None:
        haiku_in, haiku_out = MODEL_PRICING["claude-haiku-3-5-20241022"]
        sonnet_in, sonnet_out = MODEL_PRICING["claude-sonnet-4-5-20250929"]
        assert haiku_in < sonnet_in
        assert haiku_out < sonnet_out


# ---------------------------------------------------------------------------
# TestBudgetTrackerInit
# ---------------------------------------------------------------------------


class TestBudgetTrackerInit:
    """BudgetTracker initializes with configurable limits."""

    def test_default_values(self) -> None:
        tracker = BudgetTracker()
        assert tracker.max_cost_usd == 2.00
        assert tracker.max_llm_calls == 50
        assert tracker.warn_at_percent == 80

    def test_custom_values(self) -> None:
        tracker = BudgetTracker(max_cost_usd=5.0, max_llm_calls=100, warn_at_percent=90)
        assert tracker.max_cost_usd == 5.0
        assert tracker.max_llm_calls == 100
        assert tracker.warn_at_percent == 90

    def test_initial_totals(self) -> None:
        tracker = BudgetTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_calls == 0


# ---------------------------------------------------------------------------
# TestEstimateCost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """estimate_cost calculates expected cost from token counts."""

    def test_known_model(self) -> None:
        tracker = BudgetTracker()
        # Haiku: input $0.80/1M, output $4.00/1M
        cost = tracker.estimate_cost("claude-haiku-3-5-20241022", 1000, 500)
        expected = (1000 * 0.80 + 500 * 4.00) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_uses_fallback(self) -> None:
        tracker = BudgetTracker()
        # Unknown model uses (5.0, 15.0) fallback
        cost = tracker.estimate_cost("unknown-model", 1000, 500)
        expected = (1000 * 5.0 + 500 * 15.0) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens(self) -> None:
        tracker = BudgetTracker()
        assert tracker.estimate_cost("gpt-4o", 0, 0) == 0.0

    def test_sonnet_more_expensive_than_haiku(self) -> None:
        tracker = BudgetTracker()
        haiku_cost = tracker.estimate_cost("claude-haiku-3-5-20241022", 1000, 1000)
        sonnet_cost = tracker.estimate_cost("claude-sonnet-4-5-20250929", 1000, 1000)
        assert sonnet_cost > haiku_cost


# ---------------------------------------------------------------------------
# TestCheckBudget
# ---------------------------------------------------------------------------


class TestCheckBudget:
    """check_budget validates budget before spending."""

    def test_passes_within_budget(self) -> None:
        tracker = BudgetTracker(max_cost_usd=2.0)
        tracker.check_budget(estimated_cost=1.0)  # should not raise

    def test_raises_when_over_budget(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        with pytest.raises(BudgetExhaustedError, match="Budget would be exceeded"):
            tracker.check_budget(estimated_cost=1.5)

    def test_raises_when_exactly_at_budget(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        with pytest.raises(BudgetExhaustedError):
            tracker.check_budget(estimated_cost=1.0)

    def test_raises_when_calls_exhausted(self) -> None:
        tracker = BudgetTracker(max_cost_usd=100.0, max_llm_calls=2)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.001))
        # Second record_call triggers the limit (total_calls == 2 >= max_llm_calls)
        with pytest.raises(BudgetExhaustedError, match="call limit"):
            tracker.record_call(LLMCallRecord(model="test", cost_usd=0.001))
        # check_budget should also refuse after limit is reached
        with pytest.raises(BudgetExhaustedError, match="call limit"):
            tracker.check_budget()

    def test_accumulates_prior_cost(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.8))
        with pytest.raises(BudgetExhaustedError):
            tracker.check_budget(estimated_cost=0.3)


# ---------------------------------------------------------------------------
# TestRecordCall
# ---------------------------------------------------------------------------


class TestRecordCall:
    """record_call tracks calls and enforces budget."""

    def test_increments_totals(self) -> None:
        tracker = BudgetTracker()
        tracker.record_call(
            LLMCallRecord(
                model="test", input_tokens=100, output_tokens=50, cost_usd=0.01
            )
        )
        assert tracker.total_calls == 1
        assert tracker.total_cost == 0.01

    def test_accumulates_multiple_calls(self) -> None:
        tracker = BudgetTracker()
        for _ in range(3):
            tracker.record_call(LLMCallRecord(model="test", cost_usd=0.1))
        assert tracker.total_calls == 3
        assert abs(tracker.total_cost - 0.3) < 1e-10

    def test_raises_when_budget_exhausted(self) -> None:
        tracker = BudgetTracker(max_cost_usd=0.05)
        with pytest.raises(BudgetExhaustedError, match="Budget exhausted"):
            tracker.record_call(LLMCallRecord(model="test", cost_usd=0.10))

    def test_raises_when_call_limit_reached(self) -> None:
        tracker = BudgetTracker(max_cost_usd=100.0, max_llm_calls=1)
        with pytest.raises(BudgetExhaustedError, match="call limit"):
            tracker.record_call(LLMCallRecord(model="test", cost_usd=0.001))

    def test_warns_at_threshold(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0, warn_at_percent=80)
        # Spend 85% of budget
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.85))
        assert tracker._warned is True

    def test_no_warn_below_threshold(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0, warn_at_percent=80)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.5))
        assert tracker._warned is False

    def test_warns_only_once(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0, warn_at_percent=50)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.6))
        assert tracker._warned is True
        # Recording more calls should not reset warned
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.1))
        assert tracker._warned is True


# ---------------------------------------------------------------------------
# TestCostPerStep
# ---------------------------------------------------------------------------


class TestCostPerStep:
    """cost_per_step aggregates costs by graph step name."""

    def test_single_step(self) -> None:
        tracker = BudgetTracker()
        tracker.record_call(
            LLMCallRecord(model="test", cost_usd=0.01, step_name="plan")
        )
        assert tracker.cost_per_step() == {"plan": 0.01}

    def test_multiple_steps(self) -> None:
        tracker = BudgetTracker()
        tracker.record_call(
            LLMCallRecord(model="test", cost_usd=0.01, step_name="plan")
        )
        tracker.record_call(
            LLMCallRecord(model="test", cost_usd=0.02, step_name="search")
        )
        tracker.record_call(
            LLMCallRecord(model="test", cost_usd=0.03, step_name="plan")
        )
        breakdown = tracker.cost_per_step()
        assert abs(breakdown["plan"] - 0.04) < 1e-10
        assert abs(breakdown["search"] - 0.02) < 1e-10

    def test_empty_step_name_uses_unknown(self) -> None:
        tracker = BudgetTracker()
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.01))
        assert "unknown" in tracker.cost_per_step()

    def test_empty_records(self) -> None:
        tracker = BudgetTracker()
        assert tracker.cost_per_step() == {}


# ---------------------------------------------------------------------------
# TestBudgetStatus
# ---------------------------------------------------------------------------


class TestBudgetTrackerStatus:
    """status() returns accurate BudgetStatus snapshots."""

    def test_initial_status(self) -> None:
        tracker = BudgetTracker(max_cost_usd=2.0)
        status = tracker.status()
        assert status.total_cost_usd == 0.0
        assert status.total_llm_calls == 0
        assert status.budget_remaining_usd == 2.0
        assert status.budget_used_percent == 0.0
        assert status.current_tier == DegradationTier.FULL

    def test_after_spending(self) -> None:
        tracker = BudgetTracker(max_cost_usd=2.0)
        tracker.record_call(
            LLMCallRecord(
                model="test", input_tokens=500, output_tokens=200, cost_usd=0.5
            )
        )
        status = tracker.status()
        assert status.total_cost_usd == 0.5
        assert status.total_llm_calls == 1
        assert status.total_input_tokens == 500
        assert status.total_output_tokens == 200
        assert status.budget_remaining_usd == 1.5
        assert status.budget_used_percent == 25.0

    def test_token_accumulation(self) -> None:
        tracker = BudgetTracker()
        tracker.record_call(
            LLMCallRecord(model="a", input_tokens=100, output_tokens=50, cost_usd=0.001)
        )
        tracker.record_call(
            LLMCallRecord(
                model="b", input_tokens=200, output_tokens=100, cost_usd=0.001
            )
        )
        status = tracker.status()
        assert status.total_input_tokens == 300
        assert status.total_output_tokens == 150


# ---------------------------------------------------------------------------
# TestCurrentTier
# ---------------------------------------------------------------------------


class TestCurrentTier:
    """_current_tier maps budget percentage to degradation tier."""

    def test_full_tier(self) -> None:
        assert BudgetTracker._current_tier(0.0) == DegradationTier.FULL
        assert BudgetTracker._current_tier(50.0) == DegradationTier.FULL
        assert BudgetTracker._current_tier(79.9) == DegradationTier.FULL

    def test_reduced_tier(self) -> None:
        assert BudgetTracker._current_tier(80.0) == DegradationTier.REDUCED
        assert BudgetTracker._current_tier(90.0) == DegradationTier.REDUCED
        assert BudgetTracker._current_tier(94.9) == DegradationTier.REDUCED

    def test_cached_tier(self) -> None:
        assert BudgetTracker._current_tier(95.0) == DegradationTier.CACHED
        assert BudgetTracker._current_tier(99.9) == DegradationTier.CACHED

    def test_partial_tier(self) -> None:
        assert BudgetTracker._current_tier(100.0) == DegradationTier.PARTIAL
        assert BudgetTracker._current_tier(150.0) == DegradationTier.PARTIAL


# ---------------------------------------------------------------------------
# TestBudgetExhaustedError
# ---------------------------------------------------------------------------


class TestBudgetExhaustedError:
    """BudgetExhaustedError is raised correctly."""

    def test_is_exception(self) -> None:
        assert issubclass(BudgetExhaustedError, Exception)

    def test_message(self) -> None:
        err = BudgetExhaustedError("budget exceeded")
        assert str(err) == "budget exceeded"


# ---------------------------------------------------------------------------
# TestDegradationManagerInit
# ---------------------------------------------------------------------------


class TestDegradationManagerInit:
    """DegradationManager initializes with a budget tracker."""

    def test_initial_tier_is_full(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.FULL

    def test_has_tracker(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert mgr.tracker is tracker


# ---------------------------------------------------------------------------
# TestDegradationTierFromBudget
# ---------------------------------------------------------------------------


class TestDegradationTierFromBudget:
    """Tier is computed from budget percentage when not forced."""

    def test_full_tier(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.FULL

    def test_reduced_tier_at_80_percent(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.80))
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.REDUCED

    def test_cached_tier_at_95_percent(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.95))
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.CACHED


# ---------------------------------------------------------------------------
# TestDegradationModelChains
# ---------------------------------------------------------------------------


class TestDegradationModelChains:
    """Model chains differ by tier."""

    def test_full_tier_uses_sonnet(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert "sonnet" in mgr.get_model()

    def test_reduced_tier_uses_haiku(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.85))
        mgr = DegradationManager(tracker)
        assert "haiku" in mgr.get_model()

    def test_fallback_chain_returns_list(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        chain = mgr.get_fallback_chain()
        assert isinstance(chain, list)
        assert len(chain) >= 1

    def test_full_chain_has_two_models(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert len(mgr.get_fallback_chain()) == 2


# ---------------------------------------------------------------------------
# TestFeatureFlags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Feature flags control which operations are allowed per tier."""

    def test_full_tier_allows_search(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert mgr.should_skip_search() is False
        assert mgr.should_skip_scraping() is False

    def test_reduced_tier_allows_search(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.85))
        mgr = DegradationManager(tracker)
        assert mgr.should_skip_search() is False
        assert mgr.should_skip_scraping() is False

    def test_cached_tier_skips_search(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.96))
        mgr = DegradationManager(tracker)
        assert mgr.should_skip_search() is True
        assert mgr.should_skip_scraping() is False

    def test_partial_tier_skips_all(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.PARTIAL)
        assert mgr.should_skip_search() is True
        assert mgr.should_skip_scraping() is True


# ---------------------------------------------------------------------------
# TestMaxSearchResults
# ---------------------------------------------------------------------------


class TestMaxSearchResults:
    """max_search_results varies by tier."""

    def test_full_tier_returns_10(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert mgr.max_search_results() == 10

    def test_reduced_tier_returns_5(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.85))
        mgr = DegradationManager(tracker)
        assert mgr.max_search_results() == 5

    def test_cached_tier_returns_3(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.96))
        mgr = DegradationManager(tracker)
        assert mgr.max_search_results() == 3

    def test_partial_tier_returns_0(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.PARTIAL)
        assert mgr.max_search_results() == 0


# ---------------------------------------------------------------------------
# TestForceDegrade
# ---------------------------------------------------------------------------


class TestForceDegrade:
    """force_degrade overrides the computed tier."""

    def test_force_specific_tier(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.CACHED)
        assert mgr.tier == DegradationTier.CACHED

    def test_force_one_step_down(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        assert mgr.tier == DegradationTier.FULL
        mgr.force_degrade()
        assert mgr.tier == DegradationTier.REDUCED

    def test_force_multiple_steps(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        mgr.force_degrade()  # FULL -> REDUCED
        mgr.force_degrade()  # REDUCED -> CACHED
        assert mgr.tier == DegradationTier.CACHED

    def test_force_at_lowest_stays(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.PARTIAL)
        mgr.force_degrade()  # Already at PARTIAL, stays
        assert mgr.tier == DegradationTier.PARTIAL


# ---------------------------------------------------------------------------
# TestTryRecover
# ---------------------------------------------------------------------------


class TestTryRecover:
    """try_recover upgrades tier when budget pressure eases."""

    def test_recovers_when_below_threshold(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.0)
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.REDUCED)
        # Budget at 0%, well below 75% threshold
        assert mgr.try_recover() is True
        assert mgr.tier == DegradationTier.FULL

    def test_no_recovery_above_threshold(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.80))
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.CACHED)
        # Budget at 80%, above 75% threshold
        assert mgr.try_recover() is False
        assert mgr.tier == DegradationTier.CACHED

    def test_no_recovery_when_not_forced(self) -> None:
        tracker = BudgetTracker()
        mgr = DegradationManager(tracker)
        # No forced tier, recovery is automatic via budget
        assert mgr.try_recover() is False

    def test_recovers_one_step_at_a_time(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.0)
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.CACHED)
        mgr.try_recover()  # CACHED -> REDUCED
        assert mgr.tier == DegradationTier.REDUCED
        mgr.try_recover()  # REDUCED -> FULL
        assert mgr.tier == DegradationTier.FULL

    def test_recover_from_full_clears_forced(self) -> None:
        tracker = BudgetTracker(max_cost_usd=10.0)
        mgr = DegradationManager(tracker)
        mgr.force_degrade(DegradationTier.REDUCED)
        mgr.try_recover()
        # After recovering to FULL, forced tier should be cleared
        assert mgr._forced_tier is None


# ---------------------------------------------------------------------------
# TestTierTransitionLogging
# ---------------------------------------------------------------------------


class TestTierTransitionLogging:
    """Tier changes are logged via structlog."""

    def test_transition_detected(self) -> None:
        tracker = BudgetTracker(max_cost_usd=1.0)
        mgr = DegradationManager(tracker)
        # Start at FULL
        assert mgr.tier == DegradationTier.FULL
        # Spend to reach REDUCED
        tracker.record_call(LLMCallRecord(model="test", cost_usd=0.85))
        # Accessing tier should detect the transition
        assert mgr.tier == DegradationTier.REDUCED
        # Internal last_tier should be updated
        assert mgr._last_tier == DegradationTier.REDUCED
