"""Unit tests for research_agent.nodes.planner - query decomposition and validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from research_agent.nodes.planner import PlannerOutput, plan_node
from research_agent.state import Subtopic

# PlannerOutput uses Subtopic via TYPE_CHECKING, so we must rebuild
# the model after Subtopic is available at runtime.
PlannerOutput.model_rebuild()

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

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            Subtopic(question="Q?")  # type: ignore[call-arg]

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            Subtopic(id=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# plan_node stub behavior
# ---------------------------------------------------------------------------


class TestPlanNode:
    """plan_node raises NotImplementedError (stub behavior)."""

    def test_raises_not_implemented(self) -> None:
        state: dict[str, Any] = {
            "query": "What is RAG?",
            "step": "plan",
            "step_index": 0,
        }
        with pytest.raises(NotImplementedError, match="plan_node"):
            plan_node(state)

    def test_raises_with_empty_query(self) -> None:
        state: dict[str, Any] = {"query": "", "step": "plan", "step_index": 0}
        with pytest.raises(NotImplementedError, match="plan_node"):
            plan_node(state)

    def test_raises_with_missing_query(self) -> None:
        state: dict[str, Any] = {"step": "plan", "step_index": 0}
        with pytest.raises(NotImplementedError, match="plan_node"):
            plan_node(state)
