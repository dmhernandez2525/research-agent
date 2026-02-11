"""Final report generation node.

Performs one-shot synthesis from all per-subtask summaries into a coherent
research report with proper citations and structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from research_agent.state import ResearchState, Source, Summary

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class SynthesisOutput(BaseModel):
    """Structured output from the synthesis LLM call."""

    title: str = Field(description="Report title.")
    report: str = Field(description="Full Markdown report body.")
    sources: list[Source] = Field(
        default_factory=list,
        description="Cited sources used in the report.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthesis_context(summaries: list[Summary]) -> str:
    """Concatenate per-subtask summaries into a single context block.

    Args:
        summaries: All per-subtask summaries.

    Returns:
        Formatted string context for the synthesis prompt.
    """
    parts: list[str] = []
    for summary in summaries:
        header = f"## Sub-question {summary.sub_question_id}: {summary.sub_question}"
        sources_str = ", ".join(summary.source_urls) if summary.source_urls else "none"
        parts.append(f"{header}\n\n{summary.summary}\n\nSources: {sources_str}")
    return "\n\n---\n\n".join(parts)


def _synthesize_report(
    query: str,
    context: str,
) -> SynthesisOutput:
    """Generate the final research report from summarized context.

    Uses a single LLM call to produce a coherent, well-structured Markdown
    report that synthesizes all sub-question findings.

    Args:
        query: The original research query.
        context: Concatenated sub-question summaries.

    Returns:
        A ``SynthesisOutput`` with the report and source list.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    raise NotImplementedError("_synthesize_report is not yet implemented")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def synthesize_node(state: ResearchState) -> dict[str, Any]:
    """Synthesize all summaries into a final research report.

    Performs one-shot synthesis from all per-subtask summaries, generating
    a structured Markdown report with citations.

    Args:
        state: Current research state with ``summaries`` populated.

    Returns:
        Partial state update with ``final_report``, ``sources``,
        ``step``, and ``step_index``.

    Raises:
        NotImplementedError: Stub -- full implementation pending.
    """
    summaries = state.get("summaries", [])
    query = state.get("query", "")
    logger.info("synthesize_start", num_summaries=len(summaries), query=query)

    raise NotImplementedError("synthesize_node is not yet implemented")
