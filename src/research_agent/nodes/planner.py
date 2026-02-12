"""Query decomposition node.

Takes the user query and decomposes it into focused sub-questions using
LLM-powered structured output with Pydantic validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from research_agent.state import ResearchState, Subtopic

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class PlannerOutput(BaseModel):
    """Structured output from the planning LLM call."""

    subtopics: list[Subtopic] = Field(
        description="Ordered list of sub-questions to investigate.",
        min_length=1,
        max_length=10,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the decomposition strategy.",
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt() -> dict[str, str]:
    """Load the planner prompt templates from YAML.

    Returns:
        Dictionary with 'system' and 'user' prompt templates.
    """
    import yaml

    path = _PROMPTS_DIR / "planner.yaml"
    with path.open() as f:
        result: dict[str, str] = yaml.safe_load(f)
    return result


_PLANNER_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in this format: "
    '{"subtopics": [{"id": 1, "question": "<question>", '
    '"rationale": "<rationale>"}], '
    '"reasoning": "<brief strategy explanation>"}'
)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


async def _decompose_query(query: str) -> PlannerOutput:
    """Decompose a research query into subtopics via LLM.

    Uses the SMART tier (Sonnet) for query decomposition with
    JSON format instructions and Pydantic validation.

    Args:
        query: The user's research query.

    Returns:
        A validated PlannerOutput with subtopics.
    """
    import litellm

    from research_agent.models import _extract_json

    prompt_templates = _load_prompt()

    system_prompt = prompt_templates["system"] + _PLANNER_JSON_INSTRUCTION
    user_prompt = prompt_templates["user"].format(query=query)

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    data = _extract_json(content)

    # Rebuild PlannerOutput to resolve forward references
    PlannerOutput.model_rebuild()
    result = PlannerOutput(**data)

    # Renumber subtopic IDs sequentially
    for i, subtopic in enumerate(result.subtopics, start=1):
        subtopic.id = i

    logger.info(
        "decompose_query_ok",
        num_subtopics=len(result.subtopics),
        reasoning_length=len(result.reasoning),
    )

    return result


def _fallback_single_subtopic(query: str) -> list[dict[str, Any]]:
    """Create a single-subtopic fallback when LLM decomposition fails.

    Args:
        query: The original research query.

    Returns:
        A list containing one subtopic dict derived from the query.
    """
    return [
        {
            "id": 1,
            "question": query,
            "rationale": "Direct investigation of the original query.",
            "search_queries": [],
            "status": "pending",
        }
    ]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """Decompose the research query into focused sub-questions.

    Uses the SMART tier LLM (Sonnet) with structured output to produce
    a validated list of subtopics. Falls back to a single subtopic
    wrapping the original query if decomposition fails.

    Args:
        state: Current research state containing the user query.

    Returns:
        Partial state update with ``subtopics``,
        ``current_subtopic_index``, ``step``, and ``step_index``.
    """
    query = state.get("query", "")
    logger.info("plan_start", query=query)

    if not query.strip():
        logger.warning("plan_empty_query")
        return {
            "subtopics": _fallback_single_subtopic("General research"),
            "current_subtopic_index": 0,
            "step": "plan",
            "step_index": 0,
        }

    try:
        result = await _decompose_query(query)
        subtopics = [
            {
                "id": s.id,
                "question": s.question,
                "rationale": s.rationale,
                "search_queries": s.search_queries,
                "status": s.status,
            }
            for s in result.subtopics
        ]
    except Exception as exc:
        logger.error("plan_decompose_failed", error=str(exc))
        subtopics = _fallback_single_subtopic(query)

    logger.info(
        "plan_complete",
        num_subtopics=len(subtopics),
    )

    return {
        "subtopics": subtopics,
        "current_subtopic_index": 0,
        "step": "plan",
        "step_index": 0,
    }
