"""Unit tests for research_agent.nodes.planner - query decomposition and validation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from research_agent.nodes.planner import (
    PlannerOutput,
    _decompose_query,
    _fallback_single_subtopic,
    _load_prompt,
    plan_node,
)
from research_agent.state import Subtopic

# PlannerOutput uses Subtopic via TYPE_CHECKING, so we must rebuild
# the model after Subtopic is available at runtime.
PlannerOutput.model_rebuild()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock litellm response with the given content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _valid_planner_json(
    num_subtopics: int = 3,
    reasoning: str = "Breaking down the query into components.",
) -> str:
    """Generate valid planner JSON for testing."""
    subtopics = [
        {
            "id": i,
            "question": f"Sub-question {i}?",
            "rationale": f"Rationale for sub-question {i}.",
        }
        for i in range(1, num_subtopics + 1)
    ]
    return json.dumps({"subtopics": subtopics, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# PlannerOutput model validation
# ---------------------------------------------------------------------------


class TestPlannerOutput:
    """PlannerOutput validates subtopics list constraints."""

    def test_valid_construction(self) -> None:
        sq = Subtopic(id=1, question="What is RAG?")
        output = PlannerOutput(subtopics=[sq])
        assert len(output.subtopics) == 1
        assert output.subtopics[0].question == "What is RAG?"

    def test_default_reasoning_is_empty(self) -> None:
        sq = Subtopic(id=1, question="What is RAG?")
        output = PlannerOutput(subtopics=[sq])
        assert output.reasoning == ""

    def test_custom_reasoning(self) -> None:
        sq = Subtopic(id=1, question="What is RAG?")
        output = PlannerOutput(
            subtopics=[sq],
            reasoning="Breaking down the query into core components.",
        )
        assert "core components" in output.reasoning

    def test_multiple_subtopics(self) -> None:
        sqs = [
            Subtopic(id=1, question="What is RAG?"),
            Subtopic(id=2, question="How does retrieval work?"),
            Subtopic(id=3, question="What are RAG limitations?"),
        ]
        output = PlannerOutput(subtopics=sqs)
        assert len(output.subtopics) == 3

    def test_empty_subtopics_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput(subtopics=[])

    def test_max_subtopics_accepted(self) -> None:
        sqs = [Subtopic(id=i, question=f"Question {i}") for i in range(1, 11)]
        output = PlannerOutput(subtopics=sqs)
        assert len(output.subtopics) == 10

    def test_exceeds_max_subtopics_raises(self) -> None:
        sqs = [Subtopic(id=i, question=f"Question {i}") for i in range(1, 12)]
        with pytest.raises(ValidationError):
            PlannerOutput(subtopics=sqs)

    def test_missing_subtopics_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Subtopic model (used by PlannerOutput)
# ---------------------------------------------------------------------------


class TestSubtopicInPlanner:
    """Subtopic model validates fields and provides defaults."""

    def test_valid_subtopic(self) -> None:
        sq = Subtopic(id=1, question="What is RAG?")
        assert sq.id == 1
        assert sq.question == "What is RAG?"

    def test_default_rationale_empty(self) -> None:
        sq = Subtopic(id=1, question="Q?")
        assert sq.rationale == ""

    def test_custom_rationale(self) -> None:
        sq = Subtopic(id=1, question="Q?", rationale="Covers the fundamentals")
        assert sq.rationale == "Covers the fundamentals"

    def test_default_search_queries_empty(self) -> None:
        sq = Subtopic(id=1, question="Q?")
        assert sq.search_queries == []

    def test_default_status_pending(self) -> None:
        sq = Subtopic(id=1, question="Q?")
        assert sq.status == "pending"

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            Subtopic(question="Q?")  # type: ignore[call-arg]

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            Subtopic(id=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# _load_prompt
# ---------------------------------------------------------------------------


class TestLoadPrompt:
    """Planner prompt loading from YAML."""

    def test_returns_system_and_user_keys(self) -> None:
        prompts = _load_prompt()
        assert "system" in prompts
        assert "user" in prompts

    def test_system_prompt_has_content(self) -> None:
        prompts = _load_prompt()
        assert len(prompts["system"]) > 50

    def test_user_prompt_has_query_placeholder(self) -> None:
        prompts = _load_prompt()
        assert "{query}" in prompts["user"]


# ---------------------------------------------------------------------------
# _fallback_single_subtopic
# ---------------------------------------------------------------------------


class TestFallbackSingleSubtopic:
    """Fallback subtopic creation when LLM fails."""

    def test_returns_single_item_list(self) -> None:
        result = _fallback_single_subtopic("test query")
        assert len(result) == 1

    def test_uses_original_query(self) -> None:
        result = _fallback_single_subtopic("What is machine learning?")
        assert result[0]["question"] == "What is machine learning?"

    def test_has_required_fields(self) -> None:
        result = _fallback_single_subtopic("test")
        item = result[0]
        assert item["id"] == 1
        assert "rationale" in item
        assert item["search_queries"] == []
        assert item["status"] == "pending"


# ---------------------------------------------------------------------------
# _decompose_query
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    """LLM-powered query decomposition."""

    @pytest.mark.asyncio()
    async def test_successful_decomposition(self) -> None:
        mock_response = _make_mock_response(_valid_planner_json(3))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _decompose_query("What is RAG?")

        assert isinstance(result, PlannerOutput)
        assert len(result.subtopics) == 3
        assert result.subtopics[0].id == 1
        assert result.subtopics[1].id == 2
        assert result.subtopics[2].id == 3

    @pytest.mark.asyncio()
    async def test_renumbers_subtopic_ids(self) -> None:
        """Subtopic IDs are renumbered sequentially regardless of LLM output."""
        data = {
            "subtopics": [
                {"id": 10, "question": "First?", "rationale": "R1"},
                {"id": 20, "question": "Second?", "rationale": "R2"},
            ],
            "reasoning": "Strategy.",
        }
        mock_response = _make_mock_response(json.dumps(data))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _decompose_query("topic")

        assert result.subtopics[0].id == 1
        assert result.subtopics[1].id == 2

    @pytest.mark.asyncio()
    async def test_preserves_reasoning(self) -> None:
        mock_response = _make_mock_response(
            _valid_planner_json(2, reasoning="Covering breadth and depth.")
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _decompose_query("topic")

        assert "breadth and depth" in result.reasoning

    @pytest.mark.asyncio()
    async def test_handles_json_in_code_fence(self) -> None:
        """JSON wrapped in markdown code fences is extracted correctly."""
        raw_json = _valid_planner_json(2)
        fenced = f"```json\n{raw_json}\n```"
        mock_response = _make_mock_response(fenced)

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _decompose_query("topic")

        assert len(result.subtopics) == 2

    @pytest.mark.asyncio()
    async def test_raises_on_invalid_json(self) -> None:
        mock_response = _make_mock_response("This is not JSON at all.")

        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(ValueError),
        ):
            await _decompose_query("topic")

    @pytest.mark.asyncio()
    async def test_raises_on_llm_error(self) -> None:
        with (
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ),
            pytest.raises(RuntimeError, match="API error"),
        ):
            await _decompose_query("topic")

    @pytest.mark.asyncio()
    async def test_uses_sonnet_model(self) -> None:
        mock_response = _make_mock_response(_valid_planner_json(1))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_completion:
            await _decompose_query("topic")

        call_kwargs = mock_completion.call_args
        assert call_kwargs[1]["model"] == "anthropic/claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# plan_node (async)
# ---------------------------------------------------------------------------


class TestPlanNode:
    """plan_node decomposes query and returns state update."""

    @pytest.mark.asyncio()
    async def test_successful_plan(self) -> None:
        state: dict[str, Any] = {"query": "What is RAG?"}
        mock_response = _make_mock_response(_valid_planner_json(3))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await plan_node(state)

        assert result["step"] == "plan"
        assert result["step_index"] == 0
        assert result["current_subtopic_index"] == 0
        assert len(result["subtopics"]) == 3

    @pytest.mark.asyncio()
    async def test_subtopics_are_dicts(self) -> None:
        state: dict[str, Any] = {"query": "What is RAG?"}
        mock_response = _make_mock_response(_valid_planner_json(2))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await plan_node(state)

        for st in result["subtopics"]:
            assert isinstance(st, dict)
            assert "id" in st
            assert "question" in st
            assert "rationale" in st
            assert "search_queries" in st
            assert "status" in st

    @pytest.mark.asyncio()
    async def test_fallback_on_llm_error(self) -> None:
        state: dict[str, Any] = {"query": "What is machine learning?"}

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ):
            result = await plan_node(state)

        assert result["step"] == "plan"
        assert len(result["subtopics"]) == 1
        assert result["subtopics"][0]["question"] == "What is machine learning?"

    @pytest.mark.asyncio()
    async def test_fallback_on_invalid_json(self) -> None:
        state: dict[str, Any] = {"query": "test topic"}
        mock_response = _make_mock_response("I cannot parse this")

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await plan_node(state)

        assert len(result["subtopics"]) == 1
        assert result["subtopics"][0]["question"] == "test topic"

    @pytest.mark.asyncio()
    async def test_empty_query_uses_fallback(self) -> None:
        state: dict[str, Any] = {"query": ""}
        result = await plan_node(state)

        assert result["step"] == "plan"
        assert len(result["subtopics"]) == 1
        assert result["subtopics"][0]["question"] == "General research"

    @pytest.mark.asyncio()
    async def test_whitespace_only_query_uses_fallback(self) -> None:
        state: dict[str, Any] = {"query": "   "}
        result = await plan_node(state)

        assert len(result["subtopics"]) == 1
        assert result["subtopics"][0]["question"] == "General research"

    @pytest.mark.asyncio()
    async def test_missing_query_uses_fallback(self) -> None:
        state: dict[str, Any] = {}
        result = await plan_node(state)

        assert result["step"] == "plan"
        assert len(result["subtopics"]) == 1
