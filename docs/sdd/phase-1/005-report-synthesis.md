# SDD-005: Report Synthesis

## Overview

The synthesis node combines all subtopic summaries into a single coherent research report. In Phase 1, synthesis uses a one-shot approach where all summaries are passed to the LLM in a single call. The output is a structured Markdown document with citations.

## One-Shot Synthesis Architecture

```
subtopic_summaries (all)
        |
        v
+------------------+
|  Build Prompt    |
|  (template +     |
|   summaries +    |
|   instructions)  |
+--------+---------+
         |
         v
+------------------+
|  LLM Call        |
|  (synthesis)     |
+--------+---------+
         |
         v
+------------------+
|  Post-Process    |
|  (citations,     |
|   formatting)    |
+--------+---------+
         |
         v
    final_report.md
```

**Why one-shot for Phase 1:** With a typical research run producing 3-5 subtopics, each summarized to 500-1000 tokens, the total input fits comfortably within context limits (under 10k tokens of summaries). One-shot avoids the complexity of multi-pass synthesis while producing coherent output.

**Phase 3 upgrade:** Serial section-by-section synthesis will be added for long reports where summaries exceed context limits.

## Report Template Structure

The LLM is prompted to produce a report with the following structure:

```markdown
# {Title}

## Executive Summary

{2-3 paragraph overview of key findings}

## Key Findings

### {Subtopic 1 Title}

{Detailed findings with inline citations [1][2]}

### {Subtopic 2 Title}

{Detailed findings with inline citations [3][4]}

...

## Conclusions

{Summary of implications and recommendations}

## Sources

[1] {Author/Site}. "{Title}." {URL}. Accessed {date}.
[2] ...
```

## Synthesis Prompt

```python
SYNTHESIS_PROMPT = """You are a research analyst producing a comprehensive report.

## Research Question
{query}

## Subtopic Summaries

{formatted_summaries}

## Instructions

Write a well-structured research report following this format:

1. **Executive Summary** (2-3 paragraphs): High-level overview of the most important findings across all subtopics. A reader should understand the key takeaways from this section alone.

2. **Key Findings** (one subsection per subtopic): Detailed analysis organized by subtopic. Use inline citations like [1], [2] to reference specific sources. Integrate information across subtopics where relevant.

3. **Conclusions** (1-2 paragraphs): Synthesize the findings into actionable insights or recommendations.

4. **Sources**: Numbered list of all cited sources with title and URL.

## Constraints
- Maximum length: {max_length} words
- Cite every factual claim with a source number
- Do not invent information not present in the summaries
- Use clear, professional language
- Prefer specific data points over vague statements
"""
```

## Citation Management

### Citation Collection

Each `SubtopicSummary` carries a `citations` list of URLs. During prompt construction, all citations across subtopics are collected and assigned global numbers:

```python
def build_citation_index(summaries: list[SubtopicSummary]) -> dict[str, int]:
    """Map URLs to citation numbers. Deduplicate across subtopics."""
    index: dict[str, int] = {}
    counter = 1
    for summary in summaries:
        for url in summary["citations"]:
            if url not in index:
                index[url] = counter
                counter += 1
    return index

def format_summaries_with_citations(
    summaries: list[SubtopicSummary],
    citation_index: dict[str, int],
) -> str:
    """Format summaries for the synthesis prompt with numbered citations."""
    parts = []
    for summary in summaries:
        citation_refs = ", ".join(
            f"[{citation_index[url]}]" for url in summary["citations"]
            if url in citation_index
        )
        parts.append(
            f"### {summary['title']}\n\n"
            f"{summary['summary']}\n\n"
            f"Sources: {citation_refs}"
        )
    return "\n\n".join(parts)
```

### Citation Validation

After the LLM generates the report, citations are validated:

```python
def validate_citations(report: str, citation_index: dict[str, int]) -> list[str]:
    """Return list of warnings for citation issues."""
    warnings = []
    max_citation = max(citation_index.values()) if citation_index else 0

    # Find all citation references in the report
    used = set(int(m) for m in re.findall(r'\[(\d+)\]', report))

    # Check for references to non-existent citations
    for num in used:
        if num > max_citation:
            warnings.append(f"Citation [{num}] referenced but does not exist")

    # Check for unused citations (informational, not blocking)
    all_nums = set(citation_index.values())
    unused = all_nums - used
    if unused:
        warnings.append(f"Citations {unused} defined but not referenced in report")

    return warnings
```

## Length Control

The synthesis prompt includes a `max_length` constraint (default: 10,000 words from `config.yaml`). Additionally, the LLM's `max_tokens` parameter is set to cap output:

```python
def estimate_max_tokens(max_words: int) -> int:
    """Estimate max_tokens from target word count. ~1.3 tokens per word for English."""
    return int(max_words * 1.3)
```

If the generated report exceeds `max_length`, no truncation is applied in Phase 1 -- the constraint in the prompt is sufficient for reasonable adherence. Phase 2 adds post-generation length checking.

## Quality Evaluation Integration

Phase 1 includes a basic quality check. Phase 2 expands this into full LLM-as-judge evaluation.

### Phase 1: Basic Quality Check

```python
def basic_quality_check(report: str, summaries: list[SubtopicSummary]) -> dict:
    """Run basic structural quality checks on the generated report."""
    checks = {
        "has_executive_summary": "## Executive Summary" in report,
        "has_findings": "## Key Findings" in report or "## Findings" in report,
        "has_sources": "## Sources" in report,
        "has_citations": bool(re.search(r'\[\d+\]', report)),
        "subtopics_covered": sum(
            1 for s in summaries
            if s["title"].lower() in report.lower()
        ) / len(summaries) if summaries else 0,
        "word_count": len(report.split()),
    }
    checks["passes"] = (
        checks["has_executive_summary"]
        and checks["has_findings"]
        and checks["has_sources"]
        and checks["has_citations"]
        and checks["subtopics_covered"] >= 0.8
    )
    return checks
```

### Phase 2: LLM-as-Judge (Future)

A second LLM call will score the report on:
- **Coverage** (1-5): Does the report address all subtopics?
- **Accuracy** (1-5): Are claims supported by the provided summaries?
- **Coherence** (1-5): Does the report flow logically?
- **Citation quality** (1-5): Are citations used appropriately?

## Synthesize Node Implementation

```python
async def synthesize_node(state: ResearchState) -> dict:
    summaries = state["subtopic_summaries"]
    query = state["query"]
    config = state["config"]

    # Build citation index
    citation_index = build_citation_index(summaries)

    # Format summaries with citation numbers
    formatted = format_summaries_with_citations(summaries, citation_index)

    # Build synthesis prompt
    prompt = SYNTHESIS_PROMPT.format(
        query=query,
        formatted_summaries=formatted,
        max_length=config["report"]["max_length"],
    )

    # Generate report
    response = await model_router.ainvoke(prompt, max_tokens=estimate_max_tokens(config["report"]["max_length"]))
    report = response.content

    # Append sources section if LLM omitted it
    if "## Sources" not in report:
        report += format_sources_section(citation_index)

    # Validate citations
    warnings = validate_citations(report, citation_index)

    # Basic quality check
    quality = basic_quality_check(report, summaries)

    # Write report to file
    report_path = write_report(report, query, config["report"]["output_dir"])

    return {
        "final_report": report,
        "report_metadata": {
            "report_path": str(report_path),
            "citation_count": len(citation_index),
            "quality_check": quality,
            "warnings": warnings,
            "word_count": len(report.split()),
        },
    }
```

## File Location

```
src/research_agent/
    nodes/
        synthesize.py   # synthesize_node
    synthesis.py        # Citation management, quality checks, prompt template
    report.py           # write_report, format_sources_section
```
