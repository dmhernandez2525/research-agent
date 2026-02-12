"""Paywall and access-control detection for scraped web content.

Detects paywalled, login-gated, and metered content by scanning HTML
for common paywall signals before full extraction, preventing the
pipeline from ingesting partial or truncated articles.
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


class PaywallSignal(BaseModel):
    """A single detected paywall indicator."""

    pattern_name: str = Field(description="Name of the matched pattern.")
    matched_text: str = Field(default="", description="Text snippet that matched.")
    weight: float = Field(
        default=1.0, ge=0.0, le=5.0, description="Signal strength weight."
    )


class PaywallResult(BaseModel):
    """Result of paywall detection analysis."""

    is_paywalled: bool = Field(
        default=False, description="Whether the content appears paywalled."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the detection."
    )
    detected_signals: list[PaywallSignal] = Field(default_factory=list)
    total_weight: float = Field(
        default=0.0, ge=0.0, description="Sum of all signal weights."
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PaywallDetector:
    """Detects paywalled and login-gated content in HTML.

    Scans raw HTML for common paywall indicators including subscription
    prompts, login gates, metered paywalls, and truncated content markers.
    Uses weighted pattern matching: each pattern has a weight reflecting
    its reliability as a paywall signal.

    Attributes:
        threshold: Minimum total weight to flag content as paywalled.
    """

    # Pattern tuples: (name, regex, weight)
    # Higher weight = stronger signal that content is paywalled.
    PAYWALL_PATTERNS: ClassVar[list[tuple[str, str, float]]] = [
        # Hard paywalls (strong signals)
        (
            "subscription_required",
            r"subscribe\s+to\s+(read|continue|access|unlock)",
            3.0,
        ),
        (
            "subscribers_only",
            r"(this\s+)?(article|content|story)\s+is\s+(for\s+)?(subscribers?|members?)\s+only",
            3.0,
        ),
        ("premium_content", r"premium\s+(content|article|access)", 2.5),
        ("paywall_class", r'class\s*=\s*["\'][^"\']*paywall[^"\']*["\']', 2.5),
        ("paywall_id", r'id\s*=\s*["\'][^"\']*paywall[^"\']*["\']', 2.5),
        # Login gates
        (
            "login_to_read",
            r"(log\s*in|sign\s*in)\s+to\s+(read|continue|access|view)",
            2.0,
        ),
        (
            "create_account",
            r"create\s+(a\s+)?(free\s+)?account\s+to\s+(read|continue|access)",
            2.0,
        ),
        ("registration_wall", r'class\s*=\s*["\'][^"\']*regwall[^"\']*["\']', 2.5),
        # Metered paywalls
        (
            "free_articles_remaining",
            r"(you\s+have\s+)?\d+\s+(free\s+)?(articles?|stories?)\s+remaining",
            2.0,
        ),
        (
            "article_limit_reached",
            r"(you.ve|you\s+have)\s+reached\s+(your|the)\s+(monthly\s+)?(article|reading)\s+limit",
            2.5,
        ),
        # Soft signals (need additional context)
        ("subscribe_now_button", r"subscribe\s+now", 1.0),
        ("unlock_article", r"unlock\s+(this\s+)?(article|story|content)", 2.0),
        (
            "continue_reading_cta",
            r"(continue|keep)\s+reading\s+(with|for|by)\s+(a\s+)?subscription",
            2.5,
        ),
        ("trial_offer", r"(start|begin)\s+(your\s+)?(free\s+)?trial", 1.0),
        # Truncation markers
        ("content_truncated", r'class\s*=\s*["\'][^"\']*truncat[^"\']*["\']', 1.5),
        ("read_more_premium", r"read\s+more\s+with\s+(a\s+)?subscription", 2.5),
        # Common paywall overlay patterns
        (
            "overlay_modal",
            r'class\s*=\s*["\'][^"\']*(?:paywall|subscribe)[-_]?(?:modal|overlay|popup|gate)[^"\']*["\']',
            3.0,
        ),
    ]

    # Patterns that indicate free/open content (reduce confidence)
    OPEN_ACCESS_PATTERNS: ClassVar[list[tuple[str, str, float]]] = [
        (
            "open_access_badge",
            r'class\s*=\s*["\'][^"\']*open[-_]?access[^"\']*["\']',
            2.0,
        ),
        ("creative_commons", r"creative\s+commons", 1.5),
        ("free_to_read", r"free\s+to\s+read", 1.5),
    ]

    def __init__(self, threshold: float = 3.0) -> None:
        """Initialize the paywall detector.

        Args:
            threshold: Minimum total signal weight to classify as paywalled.
                Default of 3.0 requires at least one strong signal or
                multiple weak signals.
        """
        self.threshold = threshold

    def detect(self, html: str) -> PaywallResult:
        """Analyze HTML for paywall indicators.

        Args:
            html: Raw HTML content to analyze.

        Returns:
            A ``PaywallResult`` with detection outcome and signal details.
        """
        if not html or not html.strip():
            return PaywallResult()

        signals: list[PaywallSignal] = []
        lower_html = html.lower()

        # Scan for paywall patterns
        for name, pattern, weight in self.PAYWALL_PATTERNS:
            match = re.search(pattern, lower_html, re.IGNORECASE)
            if match:
                signals.append(
                    PaywallSignal(
                        pattern_name=name,
                        matched_text=match.group(0)[:100],
                        weight=weight,
                    )
                )

        total_weight = sum(s.weight for s in signals)

        # Check for open access counter-signals
        open_weight = 0.0
        for _name, pattern, weight in self.OPEN_ACCESS_PATTERNS:
            if re.search(pattern, lower_html, re.IGNORECASE):
                open_weight += weight

        # Reduce total weight by open access signals
        adjusted_weight = max(0.0, total_weight - open_weight)

        # Calculate confidence (sigmoid-like scaling)
        if adjusted_weight <= 0:
            confidence = 0.0
        else:
            confidence = min(1.0, adjusted_weight / (self.threshold * 2))

        is_paywalled = adjusted_weight >= self.threshold

        if is_paywalled:
            logger.info(
                "paywall_detected",
                signal_count=len(signals),
                total_weight=round(adjusted_weight, 2),
                confidence=round(confidence, 2),
            )

        return PaywallResult(
            is_paywalled=is_paywalled,
            confidence=round(confidence, 3),
            detected_signals=signals,
            total_weight=round(adjusted_weight, 2),
        )

    def is_accessible(self, html: str) -> bool:
        """Quick check: is the content likely accessible (not paywalled)?

        Args:
            html: Raw HTML content.

        Returns:
            ``True`` if content appears freely accessible.
        """
        return not self.detect(html).is_paywalled
