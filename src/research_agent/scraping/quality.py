"""Content quality scoring for scraped web pages.

Scores content across five dimensions:
    - Word count
    - Link density (ratio of link text to total text)
    - Boilerplate detection
    - Content density (text-to-HTML ratio)
    - Sentence length distribution
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


class QualityMetrics(BaseModel):
    """Per-dimension quality metrics for a scraped page."""

    word_count: int = Field(default=0, ge=0)
    word_count_score: float = Field(default=0.0, ge=0.0, le=1.0)

    link_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of link text to total text."
    )
    link_density_score: float = Field(default=0.0, ge=0.0, le=1.0)

    boilerplate_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated boilerplate fraction."
    )
    boilerplate_score: float = Field(default=0.0, ge=0.0, le=1.0)

    content_density: float = Field(
        default=0.0, ge=0.0, description="Text-to-tag ratio."
    )
    content_density_score: float = Field(default=0.0, ge=0.0, le=1.0)

    avg_sentence_length: float = Field(default=0.0, ge=0.0)
    sentence_length_score: float = Field(default=0.0, ge=0.0, le=1.0)

    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted composite score."
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ContentQualityScorer:
    """Scores extracted content quality across multiple dimensions.

    Attributes:
        min_words: Minimum word count for a non-zero score.
        ideal_words: Word count that achieves a perfect score.
        max_link_density: Link density above which content is penalized.
        ideal_sentence_length: Ideal average sentence length in words.
    """

    # Dimension weights (must sum to 1.0)
    WEIGHTS: ClassVar[dict[str, float]] = {
        "word_count": 0.25,
        "link_density": 0.20,
        "boilerplate": 0.20,
        "content_density": 0.15,
        "sentence_length": 0.20,
    }

    # Boilerplate indicator phrases
    BOILERPLATE_PATTERNS: ClassVar[list[str]] = [
        r"cookie\s+policy",
        r"privacy\s+policy",
        r"terms\s+(of\s+)?(service|use)",
        r"all\s+rights\s+reserved",
        r"subscribe\s+to\s+(our\s+)?newsletter",
        r"sign\s+up\s+for",
        r"follow\s+us\s+on",
        r"share\s+(this|on)",
        r"copyright\s+\d{4}",
        r"powered\s+by",
    ]

    def __init__(
        self,
        min_words: int = 50,
        ideal_words: int = 1500,
        max_link_density: float = 0.4,
        ideal_sentence_length: float = 20.0,
    ) -> None:
        """Initialize the quality scorer.

        Args:
            min_words: Minimum word count threshold.
            ideal_words: Ideal word count for full score.
            max_link_density: Max link density before penalization.
            ideal_sentence_length: Ideal average sentence length.
        """
        self.min_words = min_words
        self.ideal_words = ideal_words
        self.max_link_density = max_link_density
        self.ideal_sentence_length = ideal_sentence_length

    def score(
        self,
        text: str,
        raw_html: str = "",
        link_text: str = "",
    ) -> QualityMetrics:
        """Score the quality of extracted text content.

        Args:
            text: Extracted plain text content.
            raw_html: Original HTML (used for content density calculation).
            link_text: Concatenated text of all anchor elements.

        Returns:
            A ``QualityMetrics`` with per-dimension and overall scores.
        """
        words = text.split()
        word_count = len(words)

        # Word count score
        word_count_score = self._score_word_count(word_count)

        # Link density
        link_density = len(link_text) / max(len(text), 1)
        link_density_score = self._score_link_density(link_density)

        # Boilerplate
        boilerplate_ratio = self._detect_boilerplate(text)
        boilerplate_score = max(0.0, 1.0 - boilerplate_ratio * 2)

        # Content density
        content_density = len(text) / max(len(raw_html), 1) if raw_html else 0.5
        content_density_score = min(1.0, content_density * 3)

        # Sentence length (split on sentence-ending punctuation followed by space
        # or end-of-string, to avoid splitting on abbreviations and decimals)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )
        sentence_length_score = self._score_sentence_length(avg_sentence_length)

        # Overall
        overall = (
            self.WEIGHTS["word_count"] * word_count_score
            + self.WEIGHTS["link_density"] * link_density_score
            + self.WEIGHTS["boilerplate"] * boilerplate_score
            + self.WEIGHTS["content_density"] * content_density_score
            + self.WEIGHTS["sentence_length"] * sentence_length_score
        )

        return QualityMetrics(
            word_count=word_count,
            word_count_score=round(word_count_score, 3),
            link_density=round(link_density, 3),
            link_density_score=round(link_density_score, 3),
            boilerplate_ratio=round(boilerplate_ratio, 3),
            boilerplate_score=round(boilerplate_score, 3),
            content_density=round(content_density, 3),
            content_density_score=round(content_density_score, 3),
            avg_sentence_length=round(avg_sentence_length, 1),
            sentence_length_score=round(sentence_length_score, 3),
            overall_score=round(min(max(overall, 0.0), 1.0), 3),
        )

    def _score_word_count(self, word_count: int) -> float:
        """Score based on word count.

        Args:
            word_count: Number of words.

        Returns:
            Score between 0.0 and 1.0.
        """
        if word_count < self.min_words:
            return 0.0
        return min(1.0, word_count / self.ideal_words)

    def _score_link_density(self, link_density: float) -> float:
        """Score based on link text density (lower is better).

        Args:
            link_density: Ratio of link text to total text.

        Returns:
            Score between 0.0 and 1.0.
        """
        if link_density > self.max_link_density:
            return 0.0
        return 1.0 - (link_density / self.max_link_density)

    def _score_sentence_length(self, avg_length: float) -> float:
        """Score based on average sentence length (closer to ideal is better).

        Args:
            avg_length: Average sentence length in words.

        Returns:
            Score between 0.0 and 1.0.
        """
        if avg_length == 0:
            return 0.0
        deviation = abs(avg_length - self.ideal_sentence_length)
        return max(0.0, 1.0 - deviation / self.ideal_sentence_length)

    def _detect_boilerplate(self, text: str) -> float:
        """Estimate the fraction of text that is boilerplate.

        Args:
            text: The full text content.

        Returns:
            Estimated boilerplate ratio (0.0 - 1.0).
        """
        if not text:
            return 0.0

        lower_text = text.lower()
        matches = sum(
            1 for pattern in self.BOILERPLATE_PATTERNS if re.search(pattern, lower_text)
        )
        # Rough heuristic: each pattern match = ~5% boilerplate
        return min(1.0, matches * 0.05)
