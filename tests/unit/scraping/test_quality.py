"""Unit tests for research_agent.scraping.quality - content quality scoring."""

from __future__ import annotations

import pytest

from research_agent.scraping.quality import ContentQualityScorer, QualityMetrics

# ---------------------------------------------------------------------------
# QualityMetrics model tests
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    """QualityMetrics Pydantic model validation."""

    def test_default_construction(self) -> None:
        metrics = QualityMetrics()
        assert metrics.word_count == 0
        assert metrics.overall_score == 0.0

    def test_all_scores_bounded(self) -> None:
        metrics = QualityMetrics(
            word_count_score=1.0,
            link_density_score=1.0,
            boilerplate_score=1.0,
            content_density_score=1.0,
            sentence_length_score=1.0,
            overall_score=1.0,
        )
        assert all(
            getattr(metrics, f) >= 0.0
            for f in [
                "word_count_score",
                "link_density_score",
                "boilerplate_score",
                "content_density_score",
                "sentence_length_score",
                "overall_score",
            ]
        )

    def test_score_upper_bound_rejected(self) -> None:
        with pytest.raises(ValueError):
            QualityMetrics(overall_score=1.5)


# ---------------------------------------------------------------------------
# ContentQualityScorer tests
# ---------------------------------------------------------------------------


class TestContentQualityScorer:
    """ContentQualityScorer scoring across dimensions."""

    def test_default_init(self) -> None:
        scorer = ContentQualityScorer()
        assert scorer.min_words == 50
        assert scorer.ideal_words == 1500

    def test_custom_init(self) -> None:
        scorer = ContentQualityScorer(min_words=100, ideal_words=2000)
        assert scorer.min_words == 100
        assert scorer.ideal_words == 2000

    def test_empty_text_low_word_count(self) -> None:
        scorer = ContentQualityScorer()
        metrics = scorer.score("")
        assert metrics.word_count == 0
        assert metrics.word_count_score == 0.0
        assert metrics.sentence_length_score == 0.0

    def test_short_text_below_minimum(self) -> None:
        scorer = ContentQualityScorer(min_words=50)
        metrics = scorer.score("A few words only.")
        assert metrics.word_count_score == 0.0

    def test_ideal_length_text_high_score(self) -> None:
        scorer = ContentQualityScorer(min_words=50, ideal_words=500)
        text = "This is a well-written sentence. " * 100  # ~500 words
        metrics = scorer.score(text)
        assert metrics.word_count_score > 0.5
        assert metrics.overall_score > 0.3

    def test_high_link_density_penalized(self) -> None:
        scorer = ContentQualityScorer()
        text = "Normal content here. " * 50
        link_text = text  # 100% link density
        metrics = scorer.score(text, link_text=link_text)
        assert metrics.link_density_score == 0.0

    def test_no_link_density_full_score(self) -> None:
        scorer = ContentQualityScorer()
        text = "Normal content here. " * 50
        metrics = scorer.score(text, link_text="")
        assert metrics.link_density_score == 1.0

    def test_boilerplate_detection(self) -> None:
        scorer = ContentQualityScorer()
        text = (
            "Cookie policy applies. Terms of service. "
            "Subscribe to our newsletter. All rights reserved. "
            "Follow us on Twitter. Share this article. "
            "Copyright 2024. Powered by WordPress."
        )
        metrics = scorer.score(text)
        assert metrics.boilerplate_ratio > 0.0
        assert metrics.boilerplate_score < 1.0

    def test_no_boilerplate_full_score(self) -> None:
        scorer = ContentQualityScorer()
        text = "Research findings indicate that the algorithm performs well. " * 30
        metrics = scorer.score(text)
        assert metrics.boilerplate_ratio == 0.0
        assert metrics.boilerplate_score == 1.0

    def test_content_density_with_html(self) -> None:
        scorer = ContentQualityScorer()
        text = "Hello world"
        html = "<div><p>Hello world</p></div>" * 10
        metrics = scorer.score(text, raw_html=html)
        assert metrics.content_density > 0.0
        assert metrics.content_density_score > 0.0

    def test_content_density_without_html(self) -> None:
        scorer = ContentQualityScorer()
        text = "Some text content. " * 50
        metrics = scorer.score(text)
        assert metrics.content_density == 0.5  # Default when no HTML

    def test_sentence_length_scoring(self) -> None:
        scorer = ContentQualityScorer(ideal_sentence_length=20.0)
        # ~20 words per sentence = ideal
        text = (" ".join(["word"] * 20) + ". ") * 10
        metrics = scorer.score(text)
        assert metrics.sentence_length_score > 0.7

    def test_very_short_sentences_penalized(self) -> None:
        scorer = ContentQualityScorer(ideal_sentence_length=20.0)
        text = "Yes. No. OK. Fine. Sure. Right. Good. Done. Next. Stop."
        metrics = scorer.score(text)
        assert metrics.sentence_length_score < 0.5

    def test_overall_score_is_weighted_combination(self) -> None:
        scorer = ContentQualityScorer()
        text = "A good sentence with proper length and structure. " * 50
        metrics = scorer.score(text)
        # Overall should be between 0 and 1
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_weights_sum_to_one(self) -> None:
        total = sum(ContentQualityScorer.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6
