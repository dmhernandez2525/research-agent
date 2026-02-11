"""Per-subtask compression node.

Groups scraped content by sub-question and produces a focused summary for
each group, extracting key findings and citing source URLs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from research_agent.state import ResearchState, ScrapedContent, Summary

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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


def _summarize_group(
    sub_question_id: int,
    sub_question_text: str,
    content_items: list[ScrapedContent],
) -> Summary:
    """Produce a compressed summary for a group of content items.

    Uses LLM to summarize the combined content, extracting key findings
    and preserving source attribution.

    Args:
        sub_question_id: The sub-question ID.
        sub_question_text: The sub-question text for context.
        content_items: Scraped content for this sub-question.

    Returns:
        A ``Summary`` model with the compressed text.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_summarize_group is not yet implemented")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def summarize_node(state: ResearchState) -> dict[str, Any]:
    """Summarize scraped content grouped by sub-question.

    Produces one ``Summary`` per sub-question, containing compressed
    findings and source citations.

    Args:
        state: Current research state with ``scraped_content`` and
            ``sub_questions`` populated.

    Returns:
        Partial state update with ``summaries``, ``step``, and
        ``step_index``.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    scraped_content = state.get("scraped_content", [])
    sub_questions = state.get("sub_questions", [])
    logger.info(
        "summarize_start",
        num_content=len(scraped_content),
        num_sub_questions=len(sub_questions),
    )

    raise NotImplementedError("summarize_node is not yet implemented")
