"""Plan editing utilities for human-in-the-loop review.

Provides helpers for editing research plans via $EDITOR or inline,
serializing/deserializing sub-questions to YAML for editing, and
validating edited plans.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EditableSubQuestion(BaseModel):
    """A sub-question in a format suitable for editing."""

    id: int = Field(ge=1)
    question: str = Field(min_length=1)
    rationale: str = Field(default="")


class EditedPlan(BaseModel):
    """Validated result of an edited plan."""

    subtopics: list[EditableSubQuestion] = Field(min_length=1, max_length=20)

    @field_validator("subtopics")
    @classmethod
    def renumber_ids(
        cls,
        v: list[EditableSubQuestion],
    ) -> list[EditableSubQuestion]:
        """Ensure IDs are sequential starting from 1."""
        for i, sq in enumerate(v, start=1):
            sq.id = i
        return v


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------

_EDIT_HEADER = """\
# Research Plan Editor
# Edit the sub-questions below. You can:
#   - Remove lines to drop a sub-question
#   - Reorder entries
#   - Edit question text or rationale
#   - Add new entries (id will be auto-assigned)
# Save and close the editor when done.
# To cancel, delete all entries or leave the file empty.

subtopics:
"""


def _yaml_quote(value: str) -> str:
    """Quote a string for safe YAML embedding.

    Uses double quotes when the value contains special characters,
    otherwise emits the value unquoted.

    Args:
        value: The string to quote.

    Returns:
        A YAML-safe string representation.
    """
    needs_quoting = any(c in value for c in ":#{}[]&*?|>!%@`'\"\\,\n")
    if needs_quoting:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def plan_to_yaml(subtopics: list[dict[str, Any]]) -> str:
    """Serialize subtopics to an editable YAML string.

    Args:
        subtopics: List of subtopic dicts with id, question, rationale.

    Returns:
        YAML string with header comments and subtopic entries.
    """
    lines = [_EDIT_HEADER]
    for sq in subtopics:
        sq_id = sq.get("id", 0)
        question = sq.get("question", "")
        rationale = sq.get("rationale", "")
        lines.append(f"  - id: {sq_id}")
        lines.append(f"    question: {_yaml_quote(question)}")
        lines.append(f"    rationale: {_yaml_quote(rationale)}")
        lines.append("")
    return "\n".join(lines)


def yaml_to_plan(content: str) -> EditedPlan | None:
    """Parse edited YAML back into a validated plan.

    Args:
        content: YAML string from the editor.

    Returns:
        Validated EditedPlan, or None if the content is empty/invalid
        (indicating cancellation).
    """
    content = content.strip()
    if not content:
        return None

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        logger.warning("plan_edit_yaml_parse_error")
        return None

    if not isinstance(data, dict):
        return None

    raw_sqs = data.get("subtopics")
    if not raw_sqs or not isinstance(raw_sqs, list):
        return None

    # Normalize None values to empty strings (YAML maps bare keys to None)
    for sq in raw_sqs:
        if isinstance(sq, dict):
            for key in ("question", "rationale"):
                if sq.get(key) is None:
                    sq[key] = ""

    try:
        return EditedPlan(subtopics=raw_sqs)
    except Exception:
        logger.warning("plan_edit_validation_error")
        return None


# ---------------------------------------------------------------------------
# Editor integration
# ---------------------------------------------------------------------------


def _get_editor() -> str:
    """Get the user's preferred editor.

    Returns:
        Editor command string, falling back to 'vi'.
    """
    return os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"


def edit_plan_in_editor(subtopics: list[dict[str, Any]]) -> EditedPlan | None:
    """Open subtopics in $EDITOR for interactive editing.

    Creates a temporary YAML file, opens it in the user's editor,
    and parses the result on save.

    Args:
        subtopics: Current subtopic list.

    Returns:
        Validated EditedPlan with edited subtopics, or None if
        the user cancelled (empty file or parse failure).
    """
    yaml_content = plan_to_yaml(subtopics)
    editor = _get_editor()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="research-plan-",
        delete=False,
    ) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [editor, tmp_path],
            check=False,
        )

        if result.returncode != 0:
            logger.warning("plan_editor_nonzero_exit", code=result.returncode)
            return None

        with open(tmp_path) as f:
            edited_content = f.read()

        return yaml_to_plan(edited_content)

    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def edit_plan_inline(subtopics: list[dict[str, Any]]) -> EditedPlan | None:
    """Simple inline plan editing via numbered removal.

    Prompts the user to enter sub-question numbers to remove,
    then returns the filtered plan.

    Args:
        subtopics: Current subtopic list.

    Returns:
        Validated EditedPlan, or None on cancellation.
    """
    try:
        user_input = input(
            "\nEnter sub-question numbers to remove (comma-separated), "
            "or 'c' to cancel: "
        )
    except (EOFError, KeyboardInterrupt):
        return None

    user_input = user_input.strip()
    if not user_input or user_input.lower() == "c":
        return None

    try:
        remove_ids = {int(x.strip()) for x in user_input.split(",")}
    except ValueError:
        return None

    filtered = [sq for sq in subtopics if sq.get("id") not in remove_ids]
    if not filtered:
        return None

    try:
        return EditedPlan(
            subtopics=[
                EditableSubQuestion(
                    id=sq.get("id", 0),
                    question=sq.get("question", ""),
                    rationale=sq.get("rationale", ""),
                )
                for sq in filtered
            ],
        )
    except Exception:
        return None
