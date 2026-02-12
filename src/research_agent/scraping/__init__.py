"""Web scraping utilities: quality scoring, sanitization, and access detection."""

from __future__ import annotations

from research_agent.scraping.paywall import (
    PaywallDetector,
    PaywallResult,
    PaywallSignal,
)
from research_agent.scraping.quality import ContentQualityScorer, QualityMetrics
from research_agent.scraping.sanitizer import HTMLSanitizer, SanitizationResult

__all__ = [
    "ContentQualityScorer",
    "HTMLSanitizer",
    "PaywallDetector",
    "PaywallResult",
    "PaywallSignal",
    "QualityMetrics",
    "SanitizationResult",
]
