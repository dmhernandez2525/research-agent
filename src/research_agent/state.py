"""LangGraph research state definition.

Uses TypedDict with ``Annotated[list, operator.add]`` for automatic state
accumulation across graph nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic data models for state fields
# ---------------------------------------------------------------------------


class Subtopic(BaseModel):
    """A decomposed subtopic from the planner."""

    id: int = Field(description="1-based subtopic index.")
    question: str = Field(description="The subtopic question text.")
    rationale: str = Field(default="", description="Why this subtopic matters.")
    search_queries: list[str] = Field(
        default_factory=list, description="Pre-generated search queries."
    )
    status: str = Field(default="pending", description="Processing status.")


class SearchResult(BaseModel):
    """A single web search result."""

    subtopic_id: int = Field(description="ID of the originating subtopic.")
    query: str = Field(description="The search query that produced this result.")
    title: str = Field(default="")
    url: str = Field(description="Result URL.")
    snippet: str = Field(default="", description="Search-engine snippet.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score.")


class ScrapedPage(BaseModel):
    """Extracted content from a scraped web page."""

    url: str = Field(description="Source URL.")
    subtopic_id: int = Field(description="ID of the originating subtopic.")
    title: str = Field(default="")
    content: str = Field(default="", description="Extracted text content.")
    word_count: int = Field(default=0, ge=0)
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Content quality score."
    )


class SubtopicSummary(BaseModel):
    """A per-subtopic summary of scraped content."""

    subtopic_id: int = Field(description="ID of the originating subtopic.")
    sub_question: str = Field(default="", description="The subtopic question text.")
    summary: str = Field(description="Compressed summary text.")
    source_urls: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)


class Source(BaseModel):
    """A cited source in the final report."""

    url: str
    title: str = ""
    accessed_at: str = Field(default="", description="ISO-8601 timestamp.")
    relevance: str = Field(default="", description="How the source was used.")


class ErrorEntry(BaseModel):
    """An error or warning logged during the research run."""

    step: str = Field(description="Graph node where the error occurred.")
    message: str
    recoverable: bool = True


# ---------------------------------------------------------------------------
# LangGraph state (TypedDict with accumulator annotations)
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """Top-level state flowing through the LangGraph research pipeline.

    ``total=False`` is intentional: LangGraph nodes return partial state
    updates containing only the fields they modify. The graph engine
    merges these partial updates into the full state.

    List fields use ``Annotated[list, operator.add]`` so that each node can
    *append* to the list rather than replacing it.
    """

    # Core query -- required fields
    query: str
    step: str
    step_index: int

    # Iteration tracking
    current_subtopic_index: int
    search_retry_count: int
    seen_urls: Annotated[list[str], operator.add]

    # Planner output
    subtopics: list[Subtopic]

    # Search output (accumulates)
    search_results: Annotated[list[SearchResult], operator.add]

    # Scraper output (accumulates)
    scraped_pages: Annotated[list[ScrapedPage], operator.add]

    # Summarizer output (accumulates)
    subtopic_summaries: Annotated[list[SubtopicSummary], operator.add]

    # Synthesizer output
    final_report: str

    # Provenance
    sources: Annotated[list[Source], operator.add]

    # Report quality metadata
    report_metadata: dict[str, Any]

    # Error tracking (accumulates)
    error_log: Annotated[list[ErrorEntry], operator.add]
