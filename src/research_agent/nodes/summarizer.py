"""Per-subtask compression node.

Groups scraped content by sub-question and produces a focused summary for
each group, extracting key findings and citing source URLs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from research_agent.state import Summary

if TYPE_CHECKING:
    from research_agent.state import ResearchState, ScrapedContent

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class SummarizerOutput(BaseModel):
    """Structured output from the summarization LLM call."""

    summary: str = Field(description="200-500 word summary paragraph.")
    key_findings: list[str] = Field(
        description="3-5 key findings as concise bullet points.",
        min_length=1,
        max_length=10,
    )
    disagreements: str = Field(
        default="",
        description="Notable disagreements or gaps between sources.",
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt() -> dict[str, str]:
    """Load the summarizer prompt templates from YAML.

    Returns:
        Dictionary with 'system' and 'user' prompt templates.
    """
    import yaml

    path = _PROMPTS_DIR / "summarizer.yaml"
    with path.open() as f:
        result: dict[str, str] = yaml.safe_load(f)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_content_by_question(
    content: list[ScrapedContent],
) -> dict[int, list[ScrapedContent]]:
    """Group scraped content items by their originating sub-question ID.

    Args:
        content: All scraped content items.

    Returns:
        Mapping of sub_question_id to list of content items.
    """
    groups: dict[int, list[ScrapedContent]] = defaultdict(list)
    for item in content:
        groups[item.sub_question_id].append(item)
    return dict(groups)


def _build_content_block(items: list[ScrapedContent]) -> str:
    """Concatenate scraped content items into a single text block.

    Each item is separated by a horizontal rule and prefixed with
    its source title and URL for citation tracking.

    Args:
        items: Scraped content items to concatenate.

    Returns:
        Formatted content string for the LLM prompt.
    """
    parts: list[str] = []
    for item in items:
        header = f"Source: {item.title} ({item.url})"
        parts.append(f"{header}\n\n{item.content}")
    return "\n\n---\n\n".join(parts)


_SUMMARIZER_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in this format: "
    '{"summary": "<200-500 word summary>", '
    '"key_findings": ["finding1", "finding2", "finding3"], '
    '"disagreements": "<notable disagreements or gaps>"}'
)


async def _summarize_group(
    sub_question_id: int,
    sub_question_text: str,
    content_items: list[ScrapedContent],
) -> Summary:
    """Produce a compressed summary for a group of content items.

    Uses the SMART tier LLM (Sonnet) with structured output to generate
    a summary with key findings, preserving source attribution.

    Args:
        sub_question_id: The sub-question ID.
        sub_question_text: The sub-question text for context.
        content_items: Scraped content for this sub-question.

    Returns:
        A ``Summary`` model with the compressed text and findings.
    """
    import litellm

    from research_agent.models import _extract_json

    prompt_templates = _load_prompt()

    content_block = _build_content_block(content_items)
    user_prompt = prompt_templates["user"].format(
        sub_question=sub_question_text,
        num_sources=len(content_items),
        content=content_block,
    )

    system_prompt = prompt_templates["system"] + _SUMMARIZER_JSON_INSTRUCTION

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    content = response.choices[0].message.content
    data = _extract_json(content)
    result = SummarizerOutput(**data)

    source_urls = list({item.url for item in content_items})

    logger.info(
        "summarize_group_ok",
        sub_question_id=sub_question_id,
        num_sources=len(content_items),
        num_findings=len(result.key_findings),
        summary_words=len(result.summary.split()),
    )

    return Summary(
        sub_question_id=sub_question_id,
        sub_question=sub_question_text,
        summary=result.summary,
        source_urls=source_urls,
        key_findings=result.key_findings,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def summarize_node(state: ResearchState) -> dict[str, Any]:
    """Summarize scraped content for the current subtopic.

    Filters scraped content by the current sub-question ID, generates
    a structured summary via LLM, and increments the subtopic index
    for the next iteration.

    Args:
        state: Current research state with ``scraped_content`` and
            ``sub_questions`` populated.

    Returns:
        Partial state update with ``summaries``, ``current_subtopic_index``,
        ``step``, and ``step_index``.
    """
    scraped_content = state.get("scraped_content", [])
    sub_questions = state.get("sub_questions", [])
    current_idx = state.get("current_subtopic_index", 0)

    logger.info(
        "summarize_start",
        num_content=len(scraped_content),
        num_sub_questions=len(sub_questions),
        current_subtopic_index=current_idx,
    )

    if not sub_questions or current_idx >= len(sub_questions):
        logger.warning(
            "summarize_skip", reason="no sub-questions or index out of range"
        )
        return {
            "summaries": [],
            "step": "summarize",
            "step_index": 3,
            "current_subtopic_index": current_idx + 1,
        }

    sub_q = sub_questions[current_idx]
    sub_q_id = sub_q.get("id", current_idx + 1) if isinstance(sub_q, dict) else sub_q.id
    question = sub_q.get("question", "") if isinstance(sub_q, dict) else sub_q.question

    # Filter content for current sub-question
    grouped = _group_content_by_question(scraped_content)
    current_content = grouped.get(sub_q_id, [])

    if not current_content:
        logger.warning(
            "summarize_no_content",
            sub_question_id=sub_q_id,
            question=question,
        )
        return {
            "summaries": [],
            "step": "summarize",
            "step_index": 3,
            "current_subtopic_index": current_idx + 1,
        }

    try:
        summary = await _summarize_group(sub_q_id, question, current_content)
        summaries = [summary]
    except Exception as exc:
        logger.error(
            "summarize_failed",
            sub_question_id=sub_q_id,
            error=str(exc),
        )
        summaries = []

    logger.info(
        "summarize_complete",
        sub_question_id=sub_q_id,
        num_sources=len(current_content),
        produced_summary=len(summaries) > 0,
    )

    return {
        "summaries": summaries,
        "step": "summarize",
        "step_index": 3,
        "current_subtopic_index": current_idx + 1,
    }
