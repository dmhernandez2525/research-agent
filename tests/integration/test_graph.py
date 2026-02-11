"""Integration tests for the full LangGraph research agent graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

# TODO: Uncomment once the graph module is implemented.
# from research_agent.graph import build_graph, run_graph

pytestmark = pytest.mark.integration


class TestFullGraphExecution:
    """Run the complete research graph with mocked external APIs."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_graph_completes_without_error(
        self,
        sample_state: dict[str, Any],
        sample_config: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """The graph should run to completion and return a final state."""
        # TODO: Build the graph with mocked LLM and search APIs,
        #       invoke it with sample_state, and assert the returned
        #       state has a non-empty "report" field.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_graph_respects_max_iterations(
        self,
        sample_state: dict[str, Any],
        sample_config: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """The graph should stop after max_iterations even if quality is low."""
        # TODO: Set max_iterations=1, run graph, and verify iteration
        #       count in the final state is <= 1.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_graph_budget_enforcement(
        self,
        sample_state: dict[str, Any],
        sample_config: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """The graph should halt when the cost budget is exhausted."""
        # TODO: Set a very low budget (0.001), mock expensive LLM calls,
        #       run the graph, and assert it terminated due to budget.


class TestGraphNodeTransitions:
    """Verify correct transitions between graph nodes."""

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_planner_to_searcher_transition(
        self,
        sample_state: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """After planning, the graph should transition to the searcher node."""
        # TODO: Build the graph, run a single step from the planner,
        #       and verify the next node is "searcher".

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_evaluator_triggers_revision(
        self,
        sample_state: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """If the evaluator scores the report low, the graph should loop back."""
        # TODO: Mock the evaluator to return a low score, run the graph,
        #       and verify that a second iteration occurred.

    @pytest.mark.skip(reason="TODO: Implement once research_agent.graph exists")
    def test_evaluator_accepts_good_report(
        self,
        sample_state: dict[str, Any],
        mock_llm: MagicMock,
    ) -> None:
        """If the evaluator scores the report high, the graph should end."""
        # TODO: Mock the evaluator to return a high score, run the graph,
        #       and verify it terminated after one iteration.
