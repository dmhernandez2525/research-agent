"""Unit tests for research_agent.scraping.freshness - content freshness scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from research_agent.scraping.freshness import FreshnessResult, FreshnessScorer

# ---------------------------------------------------------------------------
# FreshnessResult model tests
# ---------------------------------------------------------------------------


class TestFreshnessResult:
    """FreshnessResult Pydantic model validation."""

    def test_default_construction(self) -> None:
        result = FreshnessResult()
        assert result.publication_date is None
        assert result.age_days is None
        assert result.freshness_score == 0.5
        assert result.is_archived is False
        assert result.date_source == "none"

    def test_custom_construction(self) -> None:
        result = FreshnessResult(
            publication_date="2024-01-15",
            age_days=30,
            freshness_score=0.9,
            date_source="meta_tag",
        )
        assert result.publication_date == "2024-01-15"
        assert result.age_days == 30

    def test_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            FreshnessResult(freshness_score=1.5)

    def test_age_days_non_negative(self) -> None:
        with pytest.raises(ValueError):
            FreshnessResult(age_days=-1)


# ---------------------------------------------------------------------------
# FreshnessScorer tests
# ---------------------------------------------------------------------------


class TestFreshnessScorer:
    """FreshnessScorer date extraction and decay scoring."""

    def test_default_init(self) -> None:
        scorer = FreshnessScorer()
        assert scorer.max_age_days == 730
        assert scorer.half_life_days == 180

    def test_custom_init(self) -> None:
        scorer = FreshnessScorer(max_age_days=365, half_life_days=90)
        assert scorer.max_age_days == 365
        assert scorer.half_life_days == 90

    def test_empty_html_returns_default(self) -> None:
        scorer = FreshnessScorer()
        result = scorer.score("")
        assert result.freshness_score == 0.5
        assert result.date_source == "none"

    def test_no_date_returns_default_score(self) -> None:
        scorer = FreshnessScorer()
        html = "<html><body><p>No date info here.</p></body></html>"
        result = scorer.score(html)
        assert result.freshness_score == FreshnessScorer.DEFAULT_SCORE
        assert result.date_source == "none"

    def test_meta_tag_date_extraction(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta property="article:published_time" content="2024-06-01">'
        result = scorer.score(html, reference_date=ref)
        assert result.publication_date == "2024-06-01"
        assert result.age_days == 14
        assert result.date_source == "meta_tag"

    def test_meta_tag_with_time(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta property="article:published_time" content="2024-06-10T12:00:00Z">'
        result = scorer.score(html, reference_date=ref)
        assert result.publication_date == "2024-06-10"
        # June 10 12:00 to June 15 00:00 = 4 days 12 hours = 4 days
        assert result.age_days == 4

    def test_json_ld_date_extraction(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 7, 1, tzinfo=UTC)
        html = """
        <script type="application/ld+json">
        {"@type": "Article", "datePublished": "2024-06-15"}
        </script>
        """
        result = scorer.score(html, reference_date=ref)
        assert result.publication_date == "2024-06-15"
        assert result.date_source == "json_ld"

    def test_content_pattern_date_extraction(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 3, 1, tzinfo=UTC)
        html = "<p>Published: January 15, 2024</p>"
        result = scorer.score(html, reference_date=ref)
        assert result.publication_date == "2024-01-15"
        assert result.date_source == "content_pattern"

    def test_meta_tag_takes_priority_over_json_ld(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 7, 1, tzinfo=UTC)
        html = """
        <meta property="article:published_time" content="2024-06-01">
        <script type="application/ld+json">
        {"@type": "Article", "datePublished": "2024-05-01"}
        </script>
        """
        result = scorer.score(html, reference_date=ref)
        assert result.publication_date == "2024-06-01"
        assert result.date_source == "meta_tag"

    def test_fresh_content_scores_high(self) -> None:
        scorer = FreshnessScorer(half_life_days=180)
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta name="date" content="2024-06-14">'
        result = scorer.score(html, reference_date=ref)
        assert result.freshness_score > 0.9

    def test_old_content_scores_low(self) -> None:
        scorer = FreshnessScorer(max_age_days=730, half_life_days=180)
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta name="date" content="2023-01-01">'
        result = scorer.score(html, reference_date=ref)
        assert result.freshness_score < 0.5

    def test_very_old_content_scores_zero(self) -> None:
        scorer = FreshnessScorer(max_age_days=365)
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta name="date" content="2022-01-01">'
        result = scorer.score(html, reference_date=ref)
        assert result.freshness_score == 0.0

    def test_today_content_scores_one(self) -> None:
        scorer = FreshnessScorer()
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        html = '<meta name="date" content="2024-06-15">'
        result = scorer.score(html, reference_date=ref)
        assert result.freshness_score == 1.0

    def test_half_life_gives_half_score(self) -> None:
        scorer = FreshnessScorer(half_life_days=180)
        ref = datetime(2024, 6, 15, tzinfo=UTC)
        # 180 days before ref
        pub_date = ref - timedelta(days=180)
        html = f'<meta name="date" content="{pub_date.date().isoformat()}">'
        result = scorer.score(html, reference_date=ref)
        assert abs(result.freshness_score - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Archive detection tests
# ---------------------------------------------------------------------------


class TestArchiveDetection:
    """FreshnessScorer detects archived and removed pages."""

    def test_page_removed_detected(self) -> None:
        scorer = FreshnessScorer()
        html = "<h1>This page has been removed</h1>"
        result = scorer.score(html)
        assert result.is_archived is True
        assert result.freshness_score == 0.0

    def test_page_archived_detected(self) -> None:
        scorer = FreshnessScorer()
        html = "<p>This article has been archived.</p>"
        result = scorer.score(html)
        assert result.is_archived is True

    def test_page_not_found_detected(self) -> None:
        scorer = FreshnessScorer()
        html = "<h1>404 Not Found</h1>"
        result = scorer.score(html)
        assert result.is_archived is True

    def test_link_expired_detected(self) -> None:
        scorer = FreshnessScorer()
        html = "<p>This link has expired.</p>"
        result = scorer.score(html)
        assert result.is_archived is True

    def test_couldnt_find_page_detected(self) -> None:
        scorer = FreshnessScorer()
        html = "<p>We couldn't find this page.</p>"
        result = scorer.score(html)
        assert result.is_archived is True

    def test_normal_content_not_flagged(self) -> None:
        scorer = FreshnessScorer()
        html = "<p>This is a normal article about software development.</p>"
        result = scorer.score(html)
        assert result.is_archived is False

    def test_archive_detection_overrides_date(self) -> None:
        scorer = FreshnessScorer()
        html = (
            '<meta name="date" content="2024-06-01"><p>This page has been removed.</p>'
        )
        result = scorer.score(html)
        assert result.is_archived is True
        assert result.freshness_score == 0.0
        assert result.date_source == "archive_detection"
