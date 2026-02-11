"""Unit tests for research_agent.nodes.planner - query decomposition and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

# TODO: Uncomment once the planner node is implemented.
# from research_agent.nodes.planner import decompose_query, validate_plan


class TestQueryDecomposition:
    """The planner should break a broad query into focused sub-queries."""

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_decomposition_produces_multiple_sub_queries(
        self,
        mock_llm: MagicMock,
        sample_state: dict[str, Any],
    ) -> None:
        """A complex query should yield at least 2 sub-queries."""
        # TODO: Call decompose_query with the sample query and mock LLM.
        #       Assert the returned list has len >= 2.

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_decomposition_sub_queries_are_non_empty(
        self,
        mock_llm: MagicMock,
        sample_state: dict[str, Any],
    ) -> None:
        """Every sub-query should be a non-empty, stripped string."""
        # TODO: Call decompose_query, iterate over results, and assert
        #       each item is a non-empty string with no leading/trailing whitespace.

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_decomposition_respects_max_sub_queries(
        self,
        mock_llm: MagicMock,
        sample_state: dict[str, Any],
    ) -> None:
        """The number of sub-queries should not exceed the configured maximum."""
        # TODO: Set max_sub_queries=3, call decompose_query, and assert
        #       len(result) <= 3.


class TestStructuredOutputValidation:
    """The planner's LLM output should conform to the expected schema."""

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_valid_json_output_parsed_correctly(self, mock_llm: MagicMock) -> None:
        """Well-formed JSON from the LLM should parse into a plan object."""
        # TODO: Configure mock_llm to return structured JSON, call
        #       validate_plan, and assert it returns a valid plan.

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_malformed_json_triggers_retry_or_error(self, mock_llm: MagicMock) -> None:
        """Malformed JSON should trigger a retry or raise a structured error."""
        # TODO: Configure mock_llm to return invalid JSON, call
        #       validate_plan, and assert the appropriate error handling.

    @pytest.mark.skip(reason="TODO: Implement once nodes.planner exists")
    def test_missing_required_fields_rejected(self, mock_llm: MagicMock) -> None:
        """A plan missing required fields should fail validation."""
        # TODO: Provide JSON without the "sub_queries" key and assert
        #       validation raises an error.
