"""Unit tests for research_agent.scraping.paywall - paywall detection."""

from __future__ import annotations

import pytest

from research_agent.scraping.paywall import (
    PaywallDetector,
    PaywallResult,
    PaywallSignal,
)

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestPaywallSignal:
    """PaywallSignal Pydantic model validation."""

    def test_construction(self) -> None:
        signal = PaywallSignal(pattern_name="test", weight=2.0)
        assert signal.pattern_name == "test"
        assert signal.weight == 2.0
        assert signal.matched_text == ""

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValueError):
            PaywallSignal(pattern_name="bad", weight=6.0)

    def test_weight_lower_bound(self) -> None:
        signal = PaywallSignal(pattern_name="ok", weight=0.0)
        assert signal.weight == 0.0


class TestPaywallResult:
    """PaywallResult Pydantic model validation."""

    def test_default_result(self) -> None:
        result = PaywallResult()
        assert result.is_paywalled is False
        assert result.confidence == 0.0
        assert result.detected_signals == []
        assert result.total_weight == 0.0

    def test_paywalled_result(self) -> None:
        result = PaywallResult(is_paywalled=True, confidence=0.8, total_weight=5.0)
        assert result.is_paywalled is True
        assert result.confidence == 0.8


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestPaywallDetector:
    """PaywallDetector pattern matching and threshold logic."""

    def test_default_threshold(self) -> None:
        detector = PaywallDetector()
        assert detector.threshold == 3.0

    def test_custom_threshold(self) -> None:
        detector = PaywallDetector(threshold=5.0)
        assert detector.threshold == 5.0

    def test_empty_html_returns_not_paywalled(self) -> None:
        detector = PaywallDetector()
        result = detector.detect("")
        assert result.is_paywalled is False
        assert result.confidence == 0.0

    def test_whitespace_only_returns_not_paywalled(self) -> None:
        detector = PaywallDetector()
        result = detector.detect("   \n  ")
        assert result.is_paywalled is False

    def test_clean_html_returns_not_paywalled(self) -> None:
        detector = PaywallDetector()
        html = "<html><body><p>Free content about technology.</p></body></html>"
        result = detector.detect(html)
        assert result.is_paywalled is False

    def test_subscription_required_detected(self) -> None:
        detector = PaywallDetector()
        html = "<p>Subscribe to read this article in full.</p>"
        result = detector.detect(html)
        assert result.is_paywalled is True
        assert any(
            s.pattern_name == "subscription_required" for s in result.detected_signals
        )

    def test_subscribers_only_detected(self) -> None:
        detector = PaywallDetector()
        html = "<p>This article is for subscribers only.</p>"
        result = detector.detect(html)
        assert result.is_paywalled is True

    def test_premium_content_signal_detected(self) -> None:
        detector = PaywallDetector()
        html = (
            '<div class="premium-banner">Premium content requires a subscription.</div>'
        )
        result = detector.detect(html)
        assert any(s.pattern_name == "premium_content" for s in result.detected_signals)
        # Single 2.5 signal is below default threshold 3.0
        assert result.total_weight >= 2.5

    def test_paywall_css_class_with_cta_detected(self) -> None:
        detector = PaywallDetector()
        html = (
            '<div class="paywall-container">'
            "<p>Subscribe to read the full article.</p>"
            "</div>"
        )
        result = detector.detect(html)
        # paywall_class (2.5) + subscription_required (3.0) = 5.5 >= 3.0
        assert result.is_paywalled is True

    def test_login_gate_detected(self) -> None:
        detector = PaywallDetector()
        html = "<p>Log in to read this article.</p>"
        result = detector.detect(html)
        signal_names = {s.pattern_name for s in result.detected_signals}
        assert "login_to_read" in signal_names

    def test_metered_paywall_detected(self) -> None:
        detector = PaywallDetector()
        html = "<p>You have 2 free articles remaining this month.</p>"
        result = detector.detect(html)
        assert any(
            s.pattern_name == "free_articles_remaining" for s in result.detected_signals
        )

    def test_article_limit_detected(self) -> None:
        detector = PaywallDetector()
        html = "<p>You've reached your monthly article limit.</p>"
        result = detector.detect(html)
        assert any(
            s.pattern_name == "article_limit_reached" for s in result.detected_signals
        )

    def test_overlay_modal_detected(self) -> None:
        detector = PaywallDetector()
        html = '<div class="paywall-overlay">Subscribe to continue</div>'
        result = detector.detect(html)
        assert result.is_paywalled is True

    def test_weak_signals_alone_below_threshold(self) -> None:
        detector = PaywallDetector()
        # "subscribe now" alone has weight 1.0, below default threshold 3.0
        html = "<button>Subscribe Now</button>"
        result = detector.detect(html)
        assert result.is_paywalled is False
        assert len(result.detected_signals) > 0

    def test_multiple_weak_signals_above_threshold(self) -> None:
        detector = PaywallDetector()
        html = (
            "<button>Subscribe Now</button>"
            "<p>Start your free trial today.</p>"
            "<p>Unlock this article to keep reading.</p>"
        )
        result = detector.detect(html)
        # subscribe_now (1.0) + trial_offer (1.0) + unlock_article (2.0) = 4.0 >= 3.0
        assert result.is_paywalled is True

    def test_open_access_reduces_weight(self) -> None:
        detector = PaywallDetector()
        # Paywall signal + creative commons counter-signal
        html = (
            "<p>Subscribe to read more.</p>"
            "<span>This work is licensed under Creative Commons.</span>"
        )
        result = detector.detect(html)
        # subscription_required (3.0) - creative_commons (1.5) = 1.5 < 3.0
        assert result.is_paywalled is False

    def test_open_access_class_reduces_weight(self) -> None:
        detector = PaywallDetector()
        html = (
            "<p>Subscribe to continue reading.</p>"
            '<div class="open-access-badge">Open Access</div>'
        )
        result = detector.detect(html)
        # subscription_required (3.0) - open_access_badge (2.0) = 1.0 < 3.0
        assert result.is_paywalled is False

    def test_confidence_scales_with_weight(self) -> None:
        detector = PaywallDetector(threshold=3.0)
        # Strong paywall signal: weight=3.0, confidence = 3.0 / 6.0 = 0.5
        html = "<p>Subscribe to read this article.</p>"
        result = detector.detect(html)
        assert result.confidence > 0.0
        assert result.confidence <= 1.0

    def test_is_accessible_returns_true_for_clean_html(self) -> None:
        detector = PaywallDetector()
        html = "<p>A free blog post about Python programming.</p>"
        assert detector.is_accessible(html) is True

    def test_is_accessible_returns_false_for_paywalled(self) -> None:
        detector = PaywallDetector()
        html = '<div class="paywall-modal">You must subscribe.</div>'
        assert detector.is_accessible(html) is False

    def test_total_weight_reflects_all_signals(self) -> None:
        detector = PaywallDetector()
        html = (
            '<div class="paywall-gate">'
            "<p>This article is for members only.</p>"
            "<p>Subscribe to unlock this content.</p>"
            "</div>"
        )
        result = detector.detect(html)
        expected_min = 3.0  # At least one strong signal
        assert result.total_weight >= expected_min

    def test_matched_text_truncated_at_100(self) -> None:
        detector = PaywallDetector()
        long_text = "Subscribe to " + "read " * 50 + "this article."
        html = f"<p>{long_text}</p>"
        result = detector.detect(html)
        for signal in result.detected_signals:
            assert len(signal.matched_text) <= 100
