"""Unit tests for research_agent.state - ResearchState creation and serialization."""

from __future__ import annotations

from typing import Any

import pytest

# TODO: Uncomment once the state module is implemented.
# from research_agent.state import ResearchState


class TestStateCreation:
    """Verify ResearchState can be constructed with valid data."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_create_state_with_query(self, sample_state: dict[str, Any]) -> None:
        """A ResearchState should be constructable from a query string."""
        # TODO: Instantiate ResearchState(query=sample_state["query"])
        #       and assert query is stored correctly.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_state_field_defaults(self) -> None:
        """Unset optional fields should receive sensible defaults."""
        # TODO: Create a state with only the required query field and
        #       assert iteration==0, cost_so_far==0.0, sub_queries==[], etc.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_state_rejects_negative_iteration(self) -> None:
        """Iteration count must be non-negative."""
        # TODO: Attempt to create a state with iteration=-1 and assert
        #       ValidationError is raised.


class TestStateSerialization:
    """Verify that ResearchState can be serialized and deserialized."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_round_trip_json(self, sample_state: dict[str, Any]) -> None:
        """State should survive a JSON round-trip without data loss."""
        # TODO: Create state, call .model_dump_json(), parse back with
        #       ResearchState.model_validate_json(), and assert equality.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_serialization_excludes_none_checkpoint_id(
        self, sample_state: dict[str, Any]
    ) -> None:
        """When checkpoint_id is None it should be excluded from the JSON output."""
        # TODO: Create state with checkpoint_id=None, dump to dict with
        #       exclude_none=True, and assert "checkpoint_id" not in output.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.state exists")
    def test_state_accepts_populated_sub_queries(self) -> None:
        """A state pre-loaded with sub_queries should preserve them."""
        # TODO: Provide sub_queries=["q1", "q2"] and verify they persist
        #       through serialization.
