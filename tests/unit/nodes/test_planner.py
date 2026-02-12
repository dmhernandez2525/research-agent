"""Unit tests for research_agent.nodes.planner - query decomposition and validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from research_agent.nodes.planner import PlannerOutput, plan_node
from research_agent.state import SubQuestion

# PlannerOutput uses SubQuestion via TYPE_CHECKING, so we must rebuild
# the model after SubQuestion is available at runtime.
PlannerOutput.model_rebuild()

# ---------------------------------------------------------------------------
# PlannerOutput model validation
# ---------------------------------------------------------------------------


class TestPlannerOutput:
    """PlannerOutput validates sub_questions list constraints."""

    def test_valid_construction(self) -> None:
        sq = SubQuestion(id=1, question="What is RAG?")
        output = PlannerOutput(sub_questions=[sq])
        assert len(output.sub_questions) == 1
        assert output.sub_questions[0].question == "What is RAG?"

    def test_default_reasoning_is_empty(self) -> None:
        sq = SubQuestion(id=1, question="What is RAG?")
        output = PlannerOutput(sub_questions=[sq])
        assert output.reasoning == ""

    def test_custom_reasoning(self) -> None:
        sq = SubQuestion(id=1, question="What is RAG?")
        output = PlannerOutput(
            sub_questions=[sq],
            reasoning="Breaking down the query into core components.",
        )
        assert "core components" in output.reasoning

    def test_multiple_sub_questions(self) -> None:
        sqs = [
            SubQuestion(id=1, question="What is RAG?"),
            SubQuestion(id=2, question="How does retrieval work?"),
            SubQuestion(id=3, question="What are RAG limitations?"),
        ]
        output = PlannerOutput(sub_questions=sqs)
        assert len(output.sub_questions) == 3

    def test_empty_sub_questions_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput(sub_questions=[])

    def test_max_sub_questions_accepted(self) -> None:
        sqs = [SubQuestion(id=i, question=f"Question {i}") for i in range(1, 11)]
        output = PlannerOutput(sub_questions=sqs)
        assert len(output.sub_questions) == 10

    def test_exceeds_max_sub_questions_raises(self) -> None:
        sqs = [SubQuestion(id=i, question=f"Question {i}") for i in range(1, 12)]
        with pytest.raises(ValidationError):
            PlannerOutput(sub_questions=sqs)

    def test_missing_sub_questions_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SubQuestion model (used by PlannerOutput)
# ---------------------------------------------------------------------------


class TestSubQuestionInPlanner:
    """SubQuestion model validates fields and provides defaults."""

    def test_valid_sub_question(self) -> None:
        sq = SubQuestion(id=1, question="What is RAG?")
        assert sq.id == 1
        assert sq.question == "What is RAG?"

    def test_default_rationale_empty(self) -> None:
        sq = SubQuestion(id=1, question="Q?")
        assert sq.rationale == ""

    def test_custom_rationale(self) -> None:
        sq = SubQuestion(id=1, question="Q?", rationale="Covers the fundamentals")
        assert sq.rationale == "Covers the fundamentals"

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            SubQuestion(question="Q?")  # type: ignore[call-arg]

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            SubQuestion(id=1)  # type: ignore[call-arg]


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
