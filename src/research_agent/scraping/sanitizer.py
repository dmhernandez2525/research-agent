"""HTML sanitization for prompt injection defense.

Strips scripts, styles, hidden elements, data attributes, and boundary
markers that could be used for prompt injection attacks.
"""

from __future__ import annotations

import re
from typing import ClassVar

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SanitizationResult(BaseModel):
    """Result of an HTML sanitization pass."""

    original_length: int = Field(
        ge=0, description="Original HTML length in characters."
    )
    sanitized_length: int = Field(ge=0, description="Sanitized HTML length.")
    elements_removed: int = Field(ge=0, description="Number of elements stripped.")
    injection_markers_found: int = Field(
        default=0, ge=0, description="Number of potential injection markers detected."
    )
    sanitized_html: str = Field(default="", description="The sanitized HTML output.")


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


class HTMLSanitizer:
    """Sanitizes HTML to defend against prompt injection in scraped content.

    Removes:
        - ``<script>`` and ``<style>`` elements
        - Hidden elements (``display:none``, ``visibility:hidden``, etc.)
        - ``data-*`` attributes
        - Boundary / delimiter markers that could confuse LLM context
        - Event handler attributes (``on*``)
        - ``<iframe>``, ``<object>``, ``<embed>`` elements

    Attributes:
        strip_comments: Whether to remove HTML comments.
        max_output_length: Maximum output length (truncate if exceeded).
    """

    # Patterns for elements to remove entirely (tag + content)
    STRIP_ELEMENTS: ClassVar[list[str]] = [
        r"<script[\s>].*?</script>",
        r"<style[\s>].*?</style>",
        r"<iframe[\s>].*?</iframe>",
        r"<object[\s>].*?</object>",
        r"<embed[^>]*>",
        r"<noscript[\s>].*?</noscript>",
    ]

    # Patterns for hidden elements
    HIDDEN_PATTERNS: ClassVar[list[str]] = [
        r'style\s*=\s*"[^"]*display\s*:\s*none[^"]*"',
        r"style\s*=\s*'[^']*display\s*:\s*none[^']*'",
        r'style\s*=\s*"[^"]*visibility\s*:\s*hidden[^"]*"',
        r"style\s*=\s*'[^']*visibility\s*:\s*hidden[^']*'",
        r'aria-hidden\s*=\s*"true"',
        r'hidden\s*=\s*"[^"]*"',
    ]

    # Potential prompt injection boundary markers
    INJECTION_MARKERS: ClassVar[list[str]] = [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
        r"Human:",
        r"Assistant:",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
    ]

    def __init__(
        self,
        strip_comments: bool = True,
        max_output_length: int = 500_000,
    ) -> None:
        """Initialize the HTML sanitizer.

        Args:
            strip_comments: Whether to remove HTML comments.
            max_output_length: Truncate output if it exceeds this length.
        """
        self.strip_comments = strip_comments
        self.max_output_length = max_output_length

    def sanitize(self, html: str) -> SanitizationResult:
        """Sanitize HTML content for safe LLM consumption.

        Args:
            html: Raw HTML string.

        Returns:
            A ``SanitizationResult`` with the cleaned HTML and statistics.
        """
        original_length = len(html)
        elements_removed = 0
        working = html

        # Strip dangerous elements
        for pattern in self.STRIP_ELEMENTS:
            matches = re.findall(pattern, working, re.DOTALL | re.IGNORECASE)
            elements_removed += len(matches)
            working = re.sub(pattern, "", working, flags=re.DOTALL | re.IGNORECASE)

        # Strip HTML comments
        if self.strip_comments:
            comments = re.findall(r"<!--.*?-->", working, re.DOTALL)
            elements_removed += len(comments)
            working = re.sub(r"<!--.*?-->", "", working, flags=re.DOTALL)

        # Strip event handler attributes
        working = re.sub(r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']", "", working)

        # Strip data-* attributes
        working = re.sub(r"\s+data-[\w-]+\s*=\s*[\"'][^\"']*[\"']", "", working)

        # Detect and neutralize injection markers
        injection_count = 0
        for marker in self.INJECTION_MARKERS:
            found = re.findall(marker, working)
            injection_count += len(found)
            working = re.sub(marker, "[REMOVED]", working)

        if injection_count > 0:
            logger.warning(
                "injection_markers_detected",
                count=injection_count,
            )

        # Strip elements with hidden styles
        for pattern in self.HIDDEN_PATTERNS:
            # Remove the attribute rather than the whole element
            working = re.sub(pattern, "", working, flags=re.IGNORECASE)

        # Truncate if needed
        if len(working) > self.max_output_length:
            working = working[: self.max_output_length]
            logger.info(
                "html_truncated",
                original_length=len(working),
                max_length=self.max_output_length,
            )

        return SanitizationResult(
            original_length=original_length,
            sanitized_length=len(working),
            elements_removed=elements_removed,
            injection_markers_found=injection_count,
            sanitized_html=working,
        )

    def sanitize_for_embedding(self, html: str) -> str:
        """Sanitize and convert HTML to plain text for embedding.

        A more aggressive sanitization that strips all HTML tags and
        returns clean text suitable for embedding models.

        Args:
            html: Raw HTML string.

        Returns:
            Clean plain text.
        """
        result = self.sanitize(html)
        text = re.sub(r"<[^>]+>", " ", result.sanitized_html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
