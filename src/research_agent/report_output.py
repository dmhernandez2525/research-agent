"""Report file output utilities.

Writes final research reports to disk with sanitized filenames and
metadata sidecar files.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILENAME_LENGTH = 80
_UNSAFE_CHARS = re.compile(r"[^\w\s-]")
_WHITESPACE = re.compile(r"[\s]+")


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------


def sanitize_filename(query: str) -> str:
    """Sanitize a research query into a filesystem-safe filename component.

    Applies the following transformations:
    - Lowercase the query
    - Remove non-alphanumeric characters (except hyphens and spaces)
    - Replace whitespace runs with single hyphens
    - Truncate to _MAX_FILENAME_LENGTH characters
    - Strip leading/trailing hyphens

    Args:
        query: The original research query.

    Returns:
        A sanitized string safe for use in filenames.
    """
    sanitized = query.lower().strip()
    sanitized = _UNSAFE_CHARS.sub("", sanitized)
    sanitized = _WHITESPACE.sub("-", sanitized)
    sanitized = sanitized.strip("-")

    if len(sanitized) > _MAX_FILENAME_LENGTH:
        sanitized = sanitized[:_MAX_FILENAME_LENGTH].rstrip("-")

    return sanitized or "report"


def generate_report_filename(query: str, timestamp: datetime | None = None) -> str:
    """Generate a report filename from the query and timestamp.

    Format: ``{sanitized_query}_{timestamp}.md``

    Args:
        query: The original research query.
        timestamp: Optional timestamp (defaults to UTC now).

    Returns:
        The generated filename string.
    """
    ts = timestamp or datetime.now(tz=UTC)
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    sanitized = sanitize_filename(query)
    return f"{sanitized}_{ts_str}.md"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_report(
    report: str,
    query: str,
    output_dir: Path | str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a report to disk with an optional metadata sidecar.

    Creates the output directory if it doesn't exist. Generates a
    filename from the sanitized query and current timestamp. Writes
    a ``.meta.json`` sidecar alongside the report file.

    Args:
        report: The full Markdown report content.
        query: The original research query (used for filename).
        output_dir: Directory to write the report into.
        metadata: Optional metadata dict to include in the sidecar.

    Returns:
        The path to the written report file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=UTC)
    filename = generate_report_filename(query, now)
    report_path = out_dir / filename

    # Write the report
    report_path.write_text(report, encoding="utf-8")

    # Write metadata sidecar
    meta = {
        "query": query,
        "generated_at": now.isoformat(),
        "word_count": len(report.split()),
        "filename": filename,
    }
    if metadata:
        meta.update(metadata)

    meta_path = report_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    logger.info(
        "report_written",
        path=str(report_path),
        meta_path=str(meta_path),
        word_count=meta["word_count"],
    )

    return report_path
