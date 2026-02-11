"""Query decomposition node.

Takes the user query and decomposes it into focused sub-questions using
structured LLM output with a Pydantic model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from research_agent.state import ResearchState, SubQuestion

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Structured output from the planning LLM call."""

    sub_questions: list[SubQuestion] = Field(
        description="Ordered list of sub-questions to investigate.",
        min_length=1,
        max_length=10,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition strategy.",
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def plan_node(state: ResearchState) -> dict[str, Any]:
    """Decompose the research query into focused sub-questions.

    Uses structured LLM output to produce a validated list of
    ``SubQuestion`` items, each with an ID and rationale.

    Args:
        state: Current research state containing the user query.

    Returns:
        Partial state update with ``sub_questions``, ``step``, and
        ``step_index``.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    query = state.get("query", "")
    logger.info("plan_start", query=query)

    raise NotImplementedError("plan_node is not yet implemented")
