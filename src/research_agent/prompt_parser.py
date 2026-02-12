"""Parser for RESEARCH_PROMPT.md structured prompt files.

Extracts topic, constraints, output requirements, and existing context
from Markdown-formatted prompt files. The parsed content is used to
configure the research pipeline (planner constraints, output format, etc.).

Expected format::

    # Topic
    The research topic or question.

    ## Constraints
    - Constraint one
    - Constraint two

    ## Output Requirements
    - Requirement one
    - Requirement two

    ## Existing Context
    Any pre-existing knowledge or context to incorporate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


@dataclass
class ResearchPrompt:
    """Parsed research prompt with structured fields.

    Attributes:
        topic: The main research topic or question.
        constraints: List of constraints for the research.
        output_requirements: List of output format/content requirements.
        existing_context: Pre-existing context or knowledge.
        raw_text: The full raw Markdown text.
    """

    topic: str = ""
    constraints: list[str] = field(default_factory=list)
    output_requirements: list[str] = field(default_factory=list)
    existing_context: str = ""
    raw_text: str = ""


def _extract_list_items(text: str) -> list[str]:
    """Extract list items from a Markdown text block.

    Handles both ``-`` and ``*`` markers, and numbered lists.

    Args:
        text: Markdown text possibly containing list items.

    Returns:
        List of stripped text items.
    """
    items: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        # Unordered list
        match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if match:
            items.append(match.group(1).strip())
            continue
        # Ordered list
        match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if match:
            items.append(match.group(1).strip())
    return items


def _split_sections(text: str) -> dict[str, str]:
    """Split Markdown text into sections by heading.

    Returns a dict mapping lowercase section names to their body text.
    Content before the first heading is mapped to the key ``"_preamble"``.

    Args:
        text: The full Markdown text.

    Returns:
        Dict of section name (lowercase) to section body.
    """
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return {"_preamble": text.strip()}

    # Content before first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["_preamble"] = preamble

    for i, match in enumerate(matches):
        name = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[name] = body

    return sections


def parse_research_prompt(text: str) -> ResearchPrompt:
    """Parse a RESEARCH_PROMPT.md formatted string.

    Extracts structured fields from the Markdown content. Section
    matching is case-insensitive and supports common heading variations
    (e.g. "Constraints", "Research Constraints", "Output Requirements",
    "Output Format", etc.).

    Args:
        text: The raw Markdown text of the prompt file.

    Returns:
        A ResearchPrompt dataclass with extracted fields.
    """
    sections = _split_sections(text)
    prompt = ResearchPrompt(raw_text=text)

    # Topic: first heading named "topic" or "_preamble"
    for key in ("topic", "research topic", "question", "research question"):
        if key in sections:
            prompt.topic = sections[key].strip()
            break
    if not prompt.topic and "_preamble" in sections:
        prompt.topic = sections["_preamble"].strip()

    # Constraints
    for key in ("constraints", "research constraints", "limitations"):
        if key in sections:
            prompt.constraints = _extract_list_items(sections[key])
            break

    # Output requirements
    for key in (
        "output requirements",
        "output format",
        "requirements",
        "deliverables",
    ):
        if key in sections:
            prompt.output_requirements = _extract_list_items(sections[key])
            break

    # Existing context
    for key in ("existing context", "context", "background", "prior research"):
        if key in sections:
            prompt.existing_context = sections[key].strip()
            break

    logger.info(
        "research_prompt_parsed",
        topic_length=len(prompt.topic),
        num_constraints=len(prompt.constraints),
        num_requirements=len(prompt.output_requirements),
        has_context=bool(prompt.existing_context),
    )

    return prompt


def load_research_prompt(path: Path | str) -> ResearchPrompt:
    """Load and parse a RESEARCH_PROMPT.md file.

    Args:
        path: Path to the Markdown prompt file.

    Returns:
        A parsed ResearchPrompt.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return parse_research_prompt(text)


def format_constraints_for_planner(prompt: ResearchPrompt) -> str:
    """Format constraints as a string for the planner system prompt.

    Produces a bullet-point list of constraints suitable for appending
    to the planner's system prompt.

    Args:
        prompt: The parsed research prompt.

    Returns:
        A formatted string, or empty string if no constraints.
    """
    if not prompt.constraints:
        return ""
    lines = ["Research constraints:"]
    for c in prompt.constraints:
        lines.append(f"- {c}")
    return "\n".join(lines)
