"""Content freshness and validity scoring for scraped web pages.

Extracts publication dates from HTML meta tags and page content,
calculates content age, and scores freshness on a 0.0-1.0 scale.
Detects archived, expired, and removed pages.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import ClassVar

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FreshnessResult(BaseModel):
    """Result of content freshness analysis."""

    publication_date: str | None = Field(
        default=None, description="Detected publication date (ISO-8601)."
    )
    age_days: int | None = Field(
        default=None, ge=0, description="Content age in days from now."
    )
    freshness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Freshness score (1.0 = very fresh, 0.0 = very old).",
    )
    is_archived: bool = Field(
        default=False, description="Whether the page appears archived or removed."
    )
    date_source: str = Field(
        default="none", description="Where the date was extracted from."
    )


# ---------------------------------------------------------------------------
# Date extraction patterns
# ---------------------------------------------------------------------------

# Meta tag attribute patterns (name/property -> content)
_META_DATE_ATTRS: list[str] = [
    "article:published_time",
    "article:modified_time",
    "og:article:published_time",
    "datePublished",
    "date",
    "DC.date",
    "DC.date.issued",
    "sailthru.date",
    "pubdate",
    "publishdate",
    "publish_date",
    "last-modified",
]

# ISO-8601 date pattern (YYYY-MM-DD with optional time)
_ISO_DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})"
    r"(?:T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?"
)

# Common date formats in page content
_CONTENT_DATE_PATTERNS: list[tuple[str, str]] = [
    # "Published: January 15, 2024" or "Updated: Jan 15, 2024"
    (
        r"(?:published|posted|updated|modified|date)\s*[:|-]\s*"
        r"(\w+\s+\d{1,2},?\s+\d{4})",
        "%B %d, %Y",
    ),
    (
        r"(?:published|posted|updated|modified|date)\s*[:|-]\s*"
        r"(\w+\s+\d{1,2},?\s+\d{4})",
        "%b %d, %Y",
    ),
    # "15 January 2024"
    (r"(\d{1,2}\s+\w+\s+\d{4})", "%d %B %Y"),
    # "01/15/2024" or "15/01/2024"
    (r"(\d{1,2}/\d{1,2}/\d{4})", "%m/%d/%Y"),
]


# ---------------------------------------------------------------------------
# Archive / removed page detection
# ---------------------------------------------------------------------------

_ARCHIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"this\s+(page|article|content)\s+(has\s+been|is|was)\s+"
        r"(removed|deleted|archived|expired|taken\s+down)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(page|content|article)\s+(not?\s+found|no\s+longer\s+(available|exists?))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(404|410)\s+(not\s+found|gone)",
        re.IGNORECASE,
    ),
    re.compile(
        r"this\s+link\s+(has\s+)?expired",
        re.IGNORECASE,
    ),
    re.compile(
        r"we\s+(couldn.t|could\s+not)\s+find\s+(the|this)\s+page",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class FreshnessScorer:
    """Scores content freshness based on publication date and page signals.

    Extracts dates from HTML meta tags and page content, calculates age,
    and applies a decay function to produce a 0.0-1.0 freshness score.

    Attributes:
        max_age_days: Content older than this gets a score of 0.0.
        half_life_days: Age at which the freshness score reaches 0.5.
    """

    # Default freshness score when no date can be extracted
    DEFAULT_SCORE: ClassVar[float] = 0.5

    def __init__(
        self,
        max_age_days: int = 730,
        half_life_days: int = 180,
    ) -> None:
        """Initialize the freshness scorer.

        Args:
            max_age_days: Maximum content age (days) before score drops to 0.
            half_life_days: Content age at which score is 0.5.
        """
        self.max_age_days = max_age_days
        self.half_life_days = half_life_days

    def score(
        self, html: str, reference_date: datetime | None = None
    ) -> FreshnessResult:
        """Score content freshness from HTML.

        Args:
            html: Raw HTML content.
            reference_date: Date to measure age against (defaults to now UTC).

        Returns:
            A ``FreshnessResult`` with date, age, score, and archive status.
        """
        if not html or not html.strip():
            return FreshnessResult()

        now = reference_date or datetime.now(tz=UTC)

        # Check for archived/removed pages
        is_archived = self._detect_archived(html)
        if is_archived:
            return FreshnessResult(
                freshness_score=0.0,
                is_archived=True,
                date_source="archive_detection",
            )

        # Try to extract a publication date
        pub_date, source = self._extract_date(html)

        if pub_date is None:
            return FreshnessResult(
                freshness_score=self.DEFAULT_SCORE,
                date_source="none",
            )

        # Calculate age in days
        age_days = max(0, (now - pub_date).days)

        # Apply decay function
        freshness = self._decay(age_days)

        return FreshnessResult(
            publication_date=pub_date.date().isoformat(),
            age_days=age_days,
            freshness_score=round(freshness, 3),
            is_archived=False,
            date_source=source,
        )

    def _extract_date(self, html: str) -> tuple[datetime | None, str]:
        """Extract the most likely publication date from HTML.

        Tries meta tags first, then falls back to content patterns.

        Args:
            html: Raw HTML content.

        Returns:
            Tuple of (datetime or None, source description).
        """
        # Try meta tags
        date = self._extract_from_meta(html)
        if date:
            return date, "meta_tag"

        # Try JSON-LD
        date = self._extract_from_json_ld(html)
        if date:
            return date, "json_ld"

        # Try content patterns
        date = self._extract_from_content(html)
        if date:
            return date, "content_pattern"

        return None, "none"

    def _extract_from_meta(self, html: str) -> datetime | None:
        """Extract date from HTML meta tags.

        Args:
            html: Raw HTML.

        Returns:
            Parsed datetime or None.
        """
        for attr in _META_DATE_ATTRS:
            # Match both name= and property= attributes
            pattern = (
                rf"<meta\s+(?:[^>]*?)"
                rf'(?:name|property)\s*=\s*["\']?{re.escape(attr)}["\']?'
                rf"(?:[^>]*?)"
                rf'content\s*=\s*["\']?([^"\'>\s]+)'
            )
            match = re.search(pattern, html, re.IGNORECASE)
            if not match:
                # Try reversed order (content before name)
                pattern_rev = (
                    rf"<meta\s+(?:[^>]*?)"
                    rf'content\s*=\s*["\']([^"\']+)["\']'
                    rf"(?:[^>]*?)"
                    rf'(?:name|property)\s*=\s*["\']?{re.escape(attr)}["\']?'
                )
                match = re.search(pattern_rev, html, re.IGNORECASE)

            if match:
                date_str = match.group(1).strip()
                parsed = self._parse_iso_date(date_str)
                if parsed:
                    return parsed
        return None

    def _extract_from_json_ld(self, html: str) -> datetime | None:
        """Extract date from JSON-LD structured data.

        Args:
            html: Raw HTML.

        Returns:
            Parsed datetime or None.
        """
        # Find datePublished or dateModified in JSON-LD
        for field in ("datePublished", "dateModified", "dateCreated"):
            pattern = rf'"{field}"\s*:\s*"([^"]+)"'
            match = re.search(pattern, html)
            if match:
                parsed = self._parse_iso_date(match.group(1))
                if parsed:
                    return parsed
        return None

    def _extract_from_content(self, html: str) -> datetime | None:
        """Extract date from visible page content.

        Args:
            html: Raw HTML.

        Returns:
            Parsed datetime or None.
        """
        # Strip tags for content scanning
        text = re.sub(r"<[^>]+>", " ", html)

        for pattern, date_fmt in _CONTENT_DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip().replace(",", "")
                try:
                    return datetime.strptime(
                        date_str, date_fmt.replace(",", "")
                    ).replace(tzinfo=UTC)
                except ValueError:
                    continue
        return None

    def _parse_iso_date(self, date_str: str) -> datetime | None:
        """Parse an ISO-8601 date string.

        Args:
            date_str: Date string to parse.

        Returns:
            Parsed datetime with UTC timezone, or None.
        """
        match = _ISO_DATE_RE.search(date_str)
        if not match:
            return None
        try:
            dt = datetime.fromisoformat(match.group(0))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            return None

    def _decay(self, age_days: int) -> float:
        """Apply exponential decay to calculate freshness score.

        Args:
            age_days: Content age in days.

        Returns:
            Freshness score between 0.0 and 1.0.
        """
        if age_days <= 0:
            return 1.0
        if age_days >= self.max_age_days:
            return 0.0

        # Exponential decay: score = 2^(-age / half_life)
        import math

        return max(0.0, min(1.0, math.pow(2, -age_days / self.half_life_days)))

    def _detect_archived(self, html: str) -> bool:
        """Check if the page appears to be archived or removed.

        Args:
            html: Raw HTML.

        Returns:
            True if archive/removal signals are detected.
        """
        text = re.sub(r"<[^>]+>", " ", html)
        return any(p.search(text) for p in _ARCHIVE_PATTERNS)
