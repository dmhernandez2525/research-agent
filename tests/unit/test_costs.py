"""Unit tests for research_agent.costs - budget tracking, degradation, tier transitions."""

from __future__ import annotations

from typing import Any

import pytest

# TODO: Uncomment once the costs module is implemented.
# from research_agent.costs import CostTracker


class TestBudgetTracking:
    """CostTracker should accurately accumulate token-level costs."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_initial_cost_is_zero(self, sample_config: dict[str, Any]) -> None:
        """A fresh CostTracker should report zero accumulated cost."""
        # TODO: tracker = CostTracker(budget=sample_config["costs"]["max_cost_per_run"])
        #       assert tracker.total_cost == 0.0

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_record_usage_increments_cost(self, sample_config: dict[str, Any]) -> None:
        """Recording token usage should increase the accumulated cost."""
        # TODO: Call tracker.record(input_tokens=100, output_tokens=50, model="...")
        #       and assert tracker.total_cost > 0.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_budget_exceeded_raises(self, sample_config: dict[str, Any]) -> None:
        """Exceeding the budget should raise a BudgetExceededError."""
        # TODO: Set a tiny budget, record a large usage, and assert the
        #       appropriate exception is raised.


class TestDegradationTriggers:
    """When cost approaches the budget, the system should degrade gracefully."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_warning_at_threshold(self, sample_config: dict[str, Any]) -> None:
        """A warning should be emitted when cost exceeds warn_at_percentage."""
        # TODO: Set warn_at_percentage=80, spend 81% of budget, and verify
        #       tracker.should_warn is True.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_no_warning_below_threshold(self, sample_config: dict[str, Any]) -> None:
        """Below the warning threshold, should_warn should be False."""
        # TODO: Spend only 50% of budget and assert should_warn is False.


class TestTierTransitions:
    """Cost pressure should trigger model tier downgrades."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_tier_downgrades_when_budget_tight(
        self, sample_config: dict[str, Any]
    ) -> None:
        """When budget is nearly exhausted, the tracker should suggest a cheaper tier."""
        # TODO: Spend 90% of budget, call tracker.recommended_tier(),
        #       and assert it returns a cheaper model identifier.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_tier_stays_when_budget_healthy(
        self, sample_config: dict[str, Any]
    ) -> None:
        """With ample budget remaining, the tier should stay at the default."""
        # TODO: Spend 10% of budget, call tracker.recommended_tier(),
        #       and assert it returns the original model.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.costs exists")
    def test_tier_transition_logged(self, sample_config: dict[str, Any]) -> None:
        """A tier transition event should be recorded for observability."""
        # TODO: Force a tier change and inspect tracker.events or log
        #       output to confirm the transition was recorded.
