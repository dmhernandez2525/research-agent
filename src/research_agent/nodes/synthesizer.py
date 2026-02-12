"""Final report generation node.

Supports two synthesis modes:
- **Single-pass**: All summaries synthesized in one LLM call (default for <= 3 subtopics).
- **Serial**: Section-by-section generation with running context, followed by an
  executive summary pass. Bypasses the single-call word ceiling for complex topics.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from research_agent.state import Source, SubtopicSummary

if TYPE_CHECKING:
    from research_agent.state import ResearchState

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULT_MAX_LENGTH = 10_000  # words
_CITATION_PATTERN = re.compile(r"\[(?:Source\s+)?(\d+)\]")


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


class SectionOutput(BaseModel):
    """Structured output for a single serial synthesis section."""

    section_title: str = Field(description="Section heading.")
    section_body: str = Field(description="Markdown body for the section.")


class ExecutiveSummaryOutput(BaseModel):
    """Structured output for the executive summary pass."""

    executive_summary: str = Field(description="2-4 sentence executive summary.")
    introduction: str = Field(description="Background/introduction paragraph.")
    conclusion: str = Field(description="Conclusion with key takeaways.")
    title: str = Field(description="Report title.")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_prompt() -> dict[str, str]:
    """Load the synthesizer prompt templates from YAML.

    Returns:
        Dictionary with 'system' and 'user' prompt templates.
    """
    import yaml

    path = _PROMPTS_DIR / "synthesizer.yaml"
    with path.open() as f:
        result: dict[str, str] = yaml.safe_load(f)
    return result


# ---------------------------------------------------------------------------
# Citation index
# ---------------------------------------------------------------------------


def _build_citation_index(summaries: list[SubtopicSummary]) -> dict[str, int]:
    """Build a deduplicated citation index from summary source URLs.

    Assigns a 1-based global number to each unique URL across all summaries.

    Args:
        summaries: All per-subtask summaries.

    Returns:
        Mapping of URL to its citation number (1-based).
    """
    seen: dict[str, int] = {}
    counter = 1
    for summary in summaries:
        for url in summary.source_urls:
            if url not in seen:
                seen[url] = counter
                counter += 1
    return seen


def _format_context_with_citations(
    summaries: list[SubtopicSummary],
    citation_index: dict[str, int],
) -> str:
    """Format summaries with numbered citation references.

    Replaces source URLs in each summary's context with [Source N] notation
    and appends the citation legend.

    Args:
        summaries: All per-subtask summaries.
        citation_index: URL-to-number mapping.

    Returns:
        Formatted context string with inline citation references.
    """
    parts: list[str] = []
    for summary in summaries:
        citations = [
            f"[Source {citation_index[url]}]"
            for url in summary.source_urls
            if url in citation_index
        ]
        citation_str = " ".join(citations) if citations else ""

        header = f"## Sub-question {summary.subtopic_id}: {summary.sub_question}"
        findings = "\n".join(f"- {f}" for f in summary.key_findings)

        block = f"{header}\n\n{summary.summary}\n\n**Key Findings:**\n{findings}"
        if citation_str:
            block += f"\n\nCited sources: {citation_str}"
        parts.append(block)

    # Append citation legend
    legend_lines = [
        f"[Source {num}]: {url}"
        for url, num in sorted(citation_index.items(), key=lambda x: x[1])
    ]
    legend = "\n".join(legend_lines)
    parts.append(f"## Citation Legend\n\n{legend}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthesis_context(summaries: list[SubtopicSummary]) -> str:
    """Concatenate per-subtask summaries into a single context block.

    Args:
        summaries: All per-subtask summaries.

    Returns:
        Formatted string context for the synthesis prompt.
    """
    parts: list[str] = []
    for summary in summaries:
        header = f"## Sub-question {summary.subtopic_id}: {summary.sub_question}"
        sources_str = ", ".join(summary.source_urls) if summary.source_urls else "none"
        parts.append(f"{header}\n\n{summary.summary}\n\nSources: {sources_str}")
    return "\n\n---\n\n".join(parts)


def _build_sources_section(
    citation_index: dict[str, int],
    synthesis_sources: list[Source],
) -> str:
    """Build a Markdown sources section from the citation index.

    Uses title information from synthesis_sources where available,
    falling back to the URL for the title.

    Args:
        citation_index: URL-to-number mapping.
        synthesis_sources: Sources returned by the LLM (may have titles).

    Returns:
        Formatted Markdown sources section.
    """
    source_titles: dict[str, str] = {}
    for src in synthesis_sources:
        if src.title:
            source_titles[src.url] = src.title

    lines: list[str] = ["## Sources", ""]
    for url, num in sorted(citation_index.items(), key=lambda x: x[1]):
        title = source_titles.get(url, url)
        lines.append(f"{num}. [{title}]({url})")

    return "\n".join(lines)


def _validate_citations(report: str, citation_index: dict[str, int]) -> list[str]:
    """Validate citation references in the report against the citation index.

    Finds all [Source N] or [N] references and checks they correspond to
    valid entries in the citation index.

    Args:
        report: The generated report text.
        citation_index: URL-to-number mapping.

    Returns:
        List of warning messages for invalid citations.
    """
    max_citation = max(citation_index.values()) if citation_index else 0
    found_numbers = {int(m.group(1)) for m in _CITATION_PATTERN.finditer(report)}

    warnings: list[str] = []
    for num in sorted(found_numbers):
        if num < 1 or num > max_citation:
            warnings.append(f"Citation [Source {num}] references non-existent source")
    return warnings


def _has_sources_section(report: str) -> bool:
    """Check if the report already contains a Sources section.

    Args:
        report: The report Markdown text.

    Returns:
        True if a sources heading exists.
    """
    return bool(re.search(r"^#{1,3}\s+Sources", report, re.MULTILINE | re.IGNORECASE))


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_SYNTHESIZER_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in this format: "
    '{"title": "<report title>", '
    '"report": "<full markdown report body>", '
    '"sources": [{"url": "<url>", "title": "<title>"}]}'
)


async def _synthesize_report(
    query: str,
    context: str,
    max_length: int = _DEFAULT_MAX_LENGTH,
) -> SynthesisOutput:
    """Generate the final research report from summarized context.

    Uses the STRATEGIC tier LLM (Sonnet with higher max_tokens) for
    complex, multi-source synthesis.

    Args:
        query: The original research query.
        context: Formatted sub-question summaries with citations.
        max_length: Target maximum word count for the report.

    Returns:
        A ``SynthesisOutput`` with the report and source list.
    """
    import litellm

    from research_agent.models import _extract_json

    prompt_templates = _load_prompt()

    system_prompt = prompt_templates["system"] + _SYNTHESIZER_JSON_INSTRUCTION
    user_prompt = prompt_templates["user"].format(
        query=query,
        context=context,
    )
    user_prompt += f"\n\nTarget maximum length: {max_length} words."

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8192,
        temperature=0.1,
    )

    content = response.choices[0].message.content
    data = _extract_json(content)
    return SynthesisOutput(**data)


# ---------------------------------------------------------------------------
# Serial synthesis
# ---------------------------------------------------------------------------

_SECTION_JSON_INSTRUCTION = (
    '\n\nRespond with ONLY a JSON object: '
    '{"section_title": "<title>", "section_body": "<markdown body>"}'
)

_EXEC_SUMMARY_JSON_INSTRUCTION = (
    '\n\nRespond with ONLY a JSON object: '
    '{"executive_summary": "<summary>", "introduction": "<intro>", '
    '"conclusion": "<conclusion>", "title": "<report title>"}'
)

_DEFAULT_SERIAL_THRESHOLD = 3


async def _synthesize_section(
    query: str,
    summary: SubtopicSummary,
    citation_index: dict[str, int],
    section_number: int,
    total_sections: int,
    prior_sections: list[str],
) -> SectionOutput:
    """Generate a single report section for one subtopic.

    Args:
        query: Original research query.
        summary: The subtopic summary to synthesize.
        citation_index: Global URL-to-number citation map.
        section_number: 1-based index of this section.
        total_sections: Total number of sections being generated.
        prior_sections: Titles and brief summaries of sections already generated.

    Returns:
        A ``SectionOutput`` with the section title and body.
    """
    import litellm

    from research_agent.models import _extract_json

    prompt_templates = _load_prompt()

    # Build section-specific context with citation references
    citations = [
        f"[Source {citation_index[url]}]"
        for url in summary.source_urls
        if url in citation_index
    ]
    citation_str = " ".join(citations) if citations else ""
    findings = "\n".join(f"- {f}" for f in summary.key_findings)
    section_context = f"{summary.summary}\n\n**Key Findings:**\n{findings}"
    if citation_str:
        section_context += f"\n\nAvailable sources: {citation_str}"

    prior_context = "\n".join(prior_sections) if prior_sections else "None (this is the first section)."

    system_prompt = prompt_templates["section_system"] + _SECTION_JSON_INSTRUCTION
    user_prompt = prompt_templates["section_user"].format(
        query=query,
        section_number=section_number,
        total_sections=total_sections,
        subtopic_question=summary.sub_question or f"Subtopic {summary.subtopic_id}",
        section_context=section_context,
        prior_context=prior_context,
    )

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
    return SectionOutput(**data)


async def _synthesize_executive_summary(
    query: str,
    sections_text: str,
    citation_legend: str,
) -> ExecutiveSummaryOutput:
    """Generate the executive summary, introduction, and conclusion.

    Called after all individual sections have been generated in serial mode.

    Args:
        query: Original research query.
        sections_text: All generated sections concatenated.
        citation_legend: The citation legend text.

    Returns:
        An ``ExecutiveSummaryOutput`` with the framing content.
    """
    import litellm

    from research_agent.models import _extract_json

    prompt_templates = _load_prompt()

    system_prompt = prompt_templates["executive_summary_system"] + _EXEC_SUMMARY_JSON_INSTRUCTION
    user_prompt = prompt_templates["executive_summary_user"].format(
        query=query,
        sections_text=sections_text,
        citation_legend=citation_legend,
    )

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
    return ExecutiveSummaryOutput(**data)


async def _synthesize_serial(
    query: str,
    summaries: list[SubtopicSummary],
    citation_index: dict[str, int],
) -> SynthesisOutput:
    """Generate a report using serial section-by-section synthesis.

    Generates each section individually with running context of prior
    sections, then produces an executive summary, introduction, and
    conclusion in a final pass. Assembles everything into a complete report.

    Args:
        query: Original research query.
        summaries: Per-subtopic summaries to synthesize.
        citation_index: Global URL-to-number citation map.

    Returns:
        A ``SynthesisOutput`` with the assembled report.
    """
    total_sections = len(summaries)
    generated_sections: list[SectionOutput] = []
    prior_summaries: list[str] = []

    for i, summary in enumerate(summaries):
        section = await _synthesize_section(
            query=query,
            summary=summary,
            citation_index=citation_index,
            section_number=i + 1,
            total_sections=total_sections,
            prior_sections=list(prior_summaries),
        )
        generated_sections.append(section)
        prior_summaries.append(f"- Section {i + 1}: {section.section_title}")

        logger.debug(
            "serial_section_complete",
            section_number=i + 1,
            title=section.section_title,
        )

    # Assemble sections text
    sections_parts: list[str] = []
    for section in generated_sections:
        sections_parts.append(f"## {section.section_title}\n\n{section.section_body}")
    sections_text = "\n\n".join(sections_parts)

    # Build citation legend for the exec summary prompt
    legend_lines = [
        f"[Source {num}]: {url}"
        for url, num in sorted(citation_index.items(), key=lambda x: x[1])
    ]
    citation_legend = "\n".join(legend_lines)

    # Generate executive summary + intro + conclusion
    exec_output = await _synthesize_executive_summary(
        query=query,
        sections_text=sections_text,
        citation_legend=citation_legend,
    )

    # Assemble the full report
    report_parts = [
        f"# {exec_output.title}",
        f"## Executive Summary\n\n{exec_output.executive_summary}",
        f"## Introduction\n\n{exec_output.introduction}",
        sections_text,
        f"## Conclusion\n\n{exec_output.conclusion}",
    ]
    report = "\n\n".join(report_parts)

    logger.info(
        "serial_synthesis_complete",
        num_sections=len(generated_sections),
        word_count=len(report.split()),
    )

    return SynthesisOutput(
        title=exec_output.title,
        report=report,
        sources=[],
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def synthesize_node(
    state: ResearchState,
    serial_threshold: int = _DEFAULT_SERIAL_THRESHOLD,
) -> dict[str, Any]:
    """Synthesize all summaries into a final research report.

    Chooses between single-pass and serial synthesis based on the number
    of subtopic summaries. Serial mode generates each section individually
    with running context, then adds an executive summary. Falls back to
    single-pass if serial mode fails.

    Args:
        state: Current research state with ``subtopic_summaries`` populated.
        serial_threshold: Use serial synthesis when subtopic count exceeds this.

    Returns:
        Partial state update with ``final_report``, ``sources``,
        ``step``, and ``step_index``.
    """
    summaries = state.get("subtopic_summaries", [])
    query = state.get("query", "")
    logger.info("synthesize_start", num_summaries=len(summaries), query=query)

    if not summaries:
        logger.warning("synthesize_skip", reason="no summaries available")
        return {
            "final_report": "",
            "sources": [],
            "step": "synthesize",
            "step_index": 4,
        }

    # Build citation index
    citation_index = _build_citation_index(summaries)
    logger.info("citation_index_built", num_sources=len(citation_index))

    # Choose synthesis mode
    use_serial = len(summaries) > serial_threshold

    try:
        if use_serial:
            logger.info(
                "serial_synthesis_selected",
                num_summaries=len(summaries),
                threshold=serial_threshold,
            )
            try:
                result = await _synthesize_serial(query, summaries, citation_index)
            except Exception as serial_exc:
                logger.warning(
                    "serial_synthesis_fallback",
                    error=str(serial_exc),
                )
                # Fall back to single-pass on serial failure
                context = _format_context_with_citations(summaries, citation_index)
                result = await _synthesize_report(query, context)
        else:
            context = _format_context_with_citations(summaries, citation_index)
            result = await _synthesize_report(query, context)
    except Exception as exc:
        logger.error("synthesize_failed", error=str(exc))
        return {
            "final_report": "",
            "sources": [],
            "step": "synthesize",
            "step_index": 4,
        }

    report = result.report

    # Validate citations
    citation_warnings = _validate_citations(report, citation_index)
    for warning in citation_warnings:
        logger.warning("citation_warning", message=warning)

    # Append Sources section if LLM omitted it
    if not _has_sources_section(report):
        sources_section = _build_sources_section(citation_index, result.sources)
        report = f"{report}\n\n{sources_section}"
        logger.info("sources_section_appended")

    # Build Source models with timestamps
    now = datetime.now(tz=UTC).isoformat()
    source_models = [
        Source(
            url=url,
            title=next(
                (s.title for s in result.sources if s.url == url),
                "",
            ),
            accessed_at=now,
        )
        for url in citation_index
    ]

    word_count = len(report.split())
    logger.info(
        "synthesize_complete",
        title=result.title,
        word_count=word_count,
        num_sources=len(source_models),
        citation_warnings=len(citation_warnings),
        mode="serial" if use_serial else "single-pass",
    )

    return {
        "final_report": report,
        "sources": source_models,
        "step": "synthesize",
        "step_index": 4,
    }
