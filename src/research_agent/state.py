"""LangGraph research state definition.

Uses TypedDict with ``Annotated[list, operator.add]`` for automatic state
accumulation across graph nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic data models for state fields
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A decomposed sub-question from the planner."""

    id: int = Field(description="1-based sub-question index.")
    question: str = Field(description="The sub-question text.")
    rationale: str = Field(default="", description="Why this sub-question matters.")


class SearchResult(BaseModel):
    """A single web search result."""

    sub_question_id: int = Field(description="ID of the originating sub-question.")
    query: str = Field(description="The search query that produced this result.")
    title: str = Field(default="")
    url: str = Field(description="Result URL.")
    snippet: str = Field(default="", description="Search-engine snippet.")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score.")


class ScrapedContent(BaseModel):
    """Extracted content from a scraped web page."""

    url: str = Field(description="Source URL.")
    sub_question_id: int = Field(description="ID of the originating sub-question.")
    title: str = Field(default="")
    content: str = Field(default="", description="Extracted text content.")
    word_count: int = Field(default=0, ge=0)
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Content quality score."
    )


class Summary(BaseModel):
    """A per-subtask summary of scraped content."""

    sub_question_id: int = Field(description="ID of the originating sub-question.")
    sub_question: str = Field(default="", description="The sub-question text.")
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

    List fields use ``Annotated[list, operator.add]`` so that each node can
    *append* to the list rather than replacing it.
    """

    # Core query
    query: str
    step: str
    step_index: int

    # Planner output
    sub_questions: list[SubQuestion]

    # Search output (accumulates across iterations)
    search_results: Annotated[list[SearchResult], operator.add]

    # Scraper output (accumulates)
    scraped_content: Annotated[list[ScrapedContent], operator.add]

    # Summarizer output (accumulates)
    summaries: Annotated[list[Summary], operator.add]

    # Synthesizer output
    final_report: str

    # Provenance
    sources: Annotated[list[Source], operator.add]

    # Error tracking (accumulates)
    error_log: Annotated[list[ErrorEntry], operator.add]
