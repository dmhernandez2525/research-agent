"""Progressive markdown output for partial research reports.

Appends subtopic summaries as they complete, producing a readable
partial report even if synthesis never runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Progressive Markdown Writer
# ---------------------------------------------------------------------------


class ProgressWriter:
    """Appends completed subtopic summaries to a progressive markdown file.

    The file is human-readable at any point during the research run,
    providing a partial report even if the agent crashes before synthesis.

    Attributes:
        path: Path to the progress.md file.
    """

    def __init__(self, path: Path, title: str = "") -> None:
        """Initialize the progress writer.

        If the file does not exist yet and a title is provided,
        writes a header line.

        Args:
            path: Path to the progress markdown file.
            title: Optional report title for the header.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists() and title:
            self._write_header(title)

    def _write_header(self, title: str) -> None:
        """Write the initial report header.

        Args:
            title: Report title.
        """
        header = f"# {title}\n\n"
        header += f"*Research in progress. Started {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M UTC')}.*\n\n"
        self.path.write_text(header, encoding="utf-8")

    def append_subtopic(
        self,
        title: str,
        summary: str,
        citations: list[str] | None = None,
        key_findings: list[str] | None = None,
    ) -> None:
        """Append a completed subtopic summary to the progress file.

        Args:
            title: Subtopic heading text.
            summary: The summary paragraph(s).
            citations: Optional list of source URLs or references.
            key_findings: Optional list of key findings.
        """
        section = f"\n## {title}\n\n"
        section += f"{summary}\n"

        if key_findings:
            section += "\n**Key Findings:**\n"
            for finding in key_findings:
                section += f"- {finding}\n"

        if citations:
            section += "\n**Sources:**\n"
            for citation in citations:
                section += f"- {citation}\n"

        section += "\n---\n"

        with self.path.open("a", encoding="utf-8") as f:
            f.write(section)
            f.flush()

        logger.info("progress_subtopic_appended", title=title)

    def append_error_note(self, node: str, message: str) -> None:
        """Append an error note to the progress file.

        Args:
            node: Graph node where the error occurred.
            message: Error description.
        """
        section = f"\n> **Note:** Error in *{node}* step: {message}\n\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(section)
            f.flush()

    def append_status(self, message: str) -> None:
        """Append a status update to the progress file.

        Args:
            message: Status message text.
        """
        line = f"\n*{message}*\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def read(self) -> str:
        """Read the current progress file content.

        Returns:
            The full markdown content, or empty string if file does not exist.
        """
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def subtopic_count(self) -> int:
        """Count the number of subtopics written so far.

        Returns:
            Number of level-2 headings (## ...) in the file.
        """
        content = self.read()
        if not content:
            return 0
        return sum(1 for line in content.splitlines() if line.startswith("## "))
