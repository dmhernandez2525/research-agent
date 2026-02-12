"""Unit tests for research_agent.evaluation - LLM-as-judge scoring."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from research_agent.evaluation import (
    _DIMENSION_DESCRIPTIONS,
    EVALUATION_DIMENSIONS,
    MAX_REVISION_CYCLES,
    QUALITY_THRESHOLD,
    DimensionScore,
    EvaluationParseError,
    EvaluationResult,
    LLMCallable,
    ReportEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_QUERY = "What are the benefits of renewable energy?"
_SAMPLE_REPORT = (
    "Renewable energy sources such as solar, wind, and hydroelectric power "
    "offer significant environmental and economic benefits. They reduce "
    "greenhouse gas emissions, create jobs, and diversify energy supply."
)


def _make_llm_response(
    scores: dict[str, float] | None = None,
    overall: str = "Solid report.",
    recommendations: list[str] | None = None,
) -> str:
    """Build a well-formed JSON response mimicking an LLM evaluation."""
    if scores is None:
        scores = {
            "Factual Accuracy": 4.0,
            "Completeness": 3.5,
            "Coverage": 4.0,
            "Coherence": 4.5,
            "Bias": 3.0,
        }
    dims = [
        {"dimension": name, "score": score, "reasoning": f"Score {score} for {name}."}
        for name, score in scores.items()
    ]
    return json.dumps(
        {
            "dimensions": dims,
            "overall_reasoning": overall,
            "recommendations": recommendations or ["Expand on sources."],
        }
    )


def _make_async_callable(response: str) -> LLMCallable:
    """Return an async callable that returns a fixed response."""

    async def _call(prompt: str) -> str:
        return response

    return _call


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------


class TestDimensionScore:
    """DimensionScore model validation and weighted score."""

    def test_valid_score(self) -> None:
        ds = DimensionScore(dimension="Factual Accuracy", score=4.0, weight=0.30)
        assert ds.dimension == "Factual Accuracy"
        assert ds.score == 4.0
        assert ds.weight == 0.30

    def test_weighted_score(self) -> None:
        ds = DimensionScore(dimension="Test", score=4.0, weight=0.25)
        assert ds.weighted_score == pytest.approx(1.0)

    def test_min_score(self) -> None:
        ds = DimensionScore(dimension="Test", score=1.0, weight=0.10)
        assert ds.weighted_score == pytest.approx(0.10)

    def test_max_score(self) -> None:
        ds = DimensionScore(dimension="Test", score=5.0, weight=1.0)
        assert ds.weighted_score == pytest.approx(5.0)

    def test_score_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(dimension="Test", score=0.5, weight=0.10)

    def test_score_above_5_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(dimension="Test", score=5.5, weight=0.10)

    def test_default_reasoning_empty(self) -> None:
        ds = DimensionScore(dimension="Test", score=3.0, weight=0.10)
        assert ds.reasoning == ""

    def test_reasoning_preserved(self) -> None:
        ds = DimensionScore(
            dimension="Test", score=3.0, weight=0.10, reasoning="Good."
        )
        assert ds.reasoning == "Good."


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    """EvaluationResult model validation."""

    def test_default_values(self) -> None:
        er = EvaluationResult(query="Test query")
        assert er.overall_score == 0.0
        assert er.dimensions == []
        assert er.recommendations == []
        assert er.overall_reasoning == ""

    def test_full_result(self) -> None:
        dims = [
            DimensionScore(dimension="Accuracy", score=4.0, weight=0.50),
            DimensionScore(dimension="Completeness", score=3.0, weight=0.50),
        ]
        er = EvaluationResult(
            query="Test query",
            dimensions=dims,
            overall_score=3.5,
            overall_reasoning="Decent report.",
            recommendations=["Add more sources."],
        )
        assert len(er.dimensions) == 2
        assert er.overall_score == 3.5
        assert len(er.recommendations) == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify evaluation constants meet spec."""

    def test_dimensions_sum_to_one(self) -> None:
        total = sum(w for _, w in EVALUATION_DIMENSIONS)
        assert total == pytest.approx(1.0)

    def test_five_dimensions(self) -> None:
        assert len(EVALUATION_DIMENSIONS) == 5

    def test_dimension_names(self) -> None:
        names = [name for name, _ in EVALUATION_DIMENSIONS]
        assert "Factual Accuracy" in names
        assert "Completeness" in names
        assert "Coverage" in names
        assert "Coherence" in names
        assert "Bias" in names

    def test_quality_threshold(self) -> None:
        assert QUALITY_THRESHOLD == 3.5

    def test_max_revision_cycles(self) -> None:
        assert MAX_REVISION_CYCLES == 2

    def test_dimension_weights(self) -> None:
        weight_map = dict(EVALUATION_DIMENSIONS)
        assert weight_map["Factual Accuracy"] == pytest.approx(0.30)
        assert weight_map["Completeness"] == pytest.approx(0.25)
        assert weight_map["Coverage"] == pytest.approx(0.20)
        assert weight_map["Coherence"] == pytest.approx(0.15)
        assert weight_map["Bias"] == pytest.approx(0.10)

    def test_all_dimensions_have_descriptions(self) -> None:
        for name, _ in EVALUATION_DIMENSIONS:
            assert name in _DIMENSION_DESCRIPTIONS
            assert len(_DIMENSION_DESCRIPTIONS[name]) > 0


# ---------------------------------------------------------------------------
# ReportEvaluator - Init & Weight Validation
# ---------------------------------------------------------------------------


class TestReportEvaluatorInit:
    """ReportEvaluator initialization and weight validation."""

    def test_default_dimensions(self) -> None:
        evaluator = ReportEvaluator()
        assert len(evaluator.dimensions) == 5

    def test_custom_dimensions(self) -> None:
        custom = [("Accuracy", 0.60), ("Style", 0.40)]
        evaluator = ReportEvaluator(dimensions=custom)
        assert len(evaluator.dimensions) == 2

    def test_weights_must_sum_to_one(self) -> None:
        bad_dims = [("A", 0.50), ("B", 0.30)]
        with pytest.raises(ValueError, match=r"weights must sum to 1\.0"):
            ReportEvaluator(dimensions=bad_dims)

    def test_weights_tolerance(self) -> None:
        # Within 0.01 tolerance should be accepted
        dims = [("A", 0.505), ("B", 0.500)]
        evaluator = ReportEvaluator(dimensions=dims)
        assert len(evaluator.dimensions) == 2

    def test_weights_outside_tolerance_rejected(self) -> None:
        dims = [("A", 0.50), ("B", 0.40)]
        with pytest.raises(ValueError):
            ReportEvaluator(dimensions=dims)


# ---------------------------------------------------------------------------
# ReportEvaluator - Prompt Building
# ---------------------------------------------------------------------------


class TestBuildEvaluationPrompt:
    """_build_evaluation_prompt constructs a structured evaluation prompt."""

    def test_contains_query(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        assert _SAMPLE_QUERY in prompt

    def test_contains_report(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        assert _SAMPLE_REPORT in prompt

    def test_contains_all_dimension_names(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        for name, _ in EVALUATION_DIMENSIONS:
            assert name in prompt

    def test_contains_dimension_weights(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        assert "30%" in prompt
        assert "25%" in prompt
        assert "20%" in prompt
        assert "15%" in prompt
        assert "10%" in prompt

    def test_contains_scoring_scale(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        assert "1-5" in prompt
        assert "1 = Very Poor" in prompt
        assert "5 = Excellent" in prompt

    def test_requests_json_format(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        assert "JSON" in prompt
        assert '"dimensions"' in prompt
        assert '"score"' in prompt
        assert '"reasoning"' in prompt

    def test_contains_dimension_descriptions(self) -> None:
        evaluator = ReportEvaluator()
        prompt = evaluator._build_evaluation_prompt(_SAMPLE_QUERY, _SAMPLE_REPORT)
        for desc in _DIMENSION_DESCRIPTIONS.values():
            assert desc in prompt

    def test_custom_dimensions_in_prompt(self) -> None:
        custom = [("Speed", 0.60), ("Depth", 0.40)]
        evaluator = ReportEvaluator(dimensions=custom)
        prompt = evaluator._build_evaluation_prompt("test query", "test report")
        assert "Speed" in prompt
        assert "Depth" in prompt
        assert "60%" in prompt
        assert "40%" in prompt

    def test_custom_dimension_uses_fallback_description(self) -> None:
        custom = [("Novel Dimension", 0.50), ("Another", 0.50)]
        evaluator = ReportEvaluator(dimensions=custom)
        prompt = evaluator._build_evaluation_prompt("q", "r")
        assert "Evaluate this dimension." in prompt


# ---------------------------------------------------------------------------
# ReportEvaluator - Response Parsing
# ---------------------------------------------------------------------------


class TestParseEvaluationResponse:
    """_parse_evaluation_response handles valid and malformed JSON."""

    def test_valid_response(self) -> None:
        evaluator = ReportEvaluator()
        raw = _make_llm_response()
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        assert isinstance(result, EvaluationResult)
        assert len(result.dimensions) == 5
        assert result.query == _SAMPLE_QUERY

    def test_scores_preserved(self) -> None:
        scores = {
            "Factual Accuracy": 5.0,
            "Completeness": 4.0,
            "Coverage": 3.0,
            "Coherence": 2.0,
            "Bias": 1.0,
        }
        evaluator = ReportEvaluator()
        raw = _make_llm_response(scores=scores)
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        score_map = {d.dimension: d.score for d in result.dimensions}
        assert score_map["Factual Accuracy"] == 5.0
        assert score_map["Completeness"] == 4.0
        assert score_map["Coverage"] == 3.0
        assert score_map["Coherence"] == 2.0
        assert score_map["Bias"] == 1.0

    def test_overall_score_computed(self) -> None:
        scores = {
            "Factual Accuracy": 4.0,
            "Completeness": 4.0,
            "Coverage": 4.0,
            "Coherence": 4.0,
            "Bias": 4.0,
        }
        evaluator = ReportEvaluator()
        raw = _make_llm_response(scores=scores)
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        # All 4s with weights summing to 1.0 -> overall = 4.0
        assert result.overall_score == pytest.approx(4.0)

    def test_overall_reasoning_captured(self) -> None:
        evaluator = ReportEvaluator()
        raw = _make_llm_response(overall="Very thorough analysis.")
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        assert result.overall_reasoning == "Very thorough analysis."

    def test_recommendations_captured(self) -> None:
        evaluator = ReportEvaluator()
        raw = _make_llm_response(recommendations=["Add sources.", "Fix formatting."])
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        assert len(result.recommendations) == 2
        assert "Add sources." in result.recommendations

    def test_invalid_json_raises(self) -> None:
        evaluator = ReportEvaluator()
        with pytest.raises(EvaluationParseError, match="not valid JSON"):
            evaluator._parse_evaluation_response("not json at all", _SAMPLE_QUERY)

    def test_missing_dimensions_key_raises(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({"overall_reasoning": "test"})
        with pytest.raises(EvaluationParseError, match="Missing 'dimensions'"):
            evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)

    def test_missing_score_raises(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "reasoning": "Good."}
            ]
        })
        with pytest.raises(EvaluationParseError, match="Missing score"):
            evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)

    def test_markdown_fences_stripped(self) -> None:
        evaluator = ReportEvaluator()
        inner = _make_llm_response()
        raw = f"```json\n{inner}\n```"
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        assert len(result.dimensions) == 5

    def test_score_clamped_to_max(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 7.0, "reasoning": ""},
                {"dimension": "Completeness", "score": 3.0, "reasoning": ""},
                {"dimension": "Coverage", "score": 3.0, "reasoning": ""},
                {"dimension": "Coherence", "score": 3.0, "reasoning": ""},
                {"dimension": "Bias", "score": 3.0, "reasoning": ""},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        score_map = {d.dimension: d.score for d in result.dimensions}
        assert score_map["Factual Accuracy"] == 5.0

    def test_score_clamped_to_min(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": -1.0, "reasoning": ""},
                {"dimension": "Completeness", "score": 3.0, "reasoning": ""},
                {"dimension": "Coverage", "score": 3.0, "reasoning": ""},
                {"dimension": "Coherence", "score": 3.0, "reasoning": ""},
                {"dimension": "Bias", "score": 3.0, "reasoning": ""},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        score_map = {d.dimension: d.score for d in result.dimensions}
        assert score_map["Factual Accuracy"] == 1.0

    def test_missing_dimension_defaults_to_1(self) -> None:
        evaluator = ReportEvaluator()
        # Only provide 4 of 5 dimensions
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 4.0, "reasoning": ""},
                {"dimension": "Completeness", "score": 4.0, "reasoning": ""},
                {"dimension": "Coverage", "score": 4.0, "reasoning": ""},
                {"dimension": "Coherence", "score": 4.0, "reasoning": ""},
                # Missing: Bias
            ],
        })
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        score_map = {d.dimension: d.score for d in result.dimensions}
        assert score_map["Bias"] == 1.0
        assert "defaulted" in next(
            d.reasoning for d in result.dimensions if d.dimension == "Bias"
        ).lower()

    def test_unexpected_dimension_ignored(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 4.0, "reasoning": ""},
                {"dimension": "Completeness", "score": 4.0, "reasoning": ""},
                {"dimension": "Coverage", "score": 4.0, "reasoning": ""},
                {"dimension": "Coherence", "score": 4.0, "reasoning": ""},
                {"dimension": "Bias", "score": 4.0, "reasoning": ""},
                {"dimension": "Novelty", "score": 5.0, "reasoning": ""},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        dim_names = {d.dimension for d in result.dimensions}
        assert "Novelty" not in dim_names
        assert len(result.dimensions) == 5

    def test_empty_recommendations_ok(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 4.0, "reasoning": ""},
                {"dimension": "Completeness", "score": 4.0, "reasoning": ""},
                {"dimension": "Coverage", "score": 4.0, "reasoning": ""},
                {"dimension": "Coherence", "score": 4.0, "reasoning": ""},
                {"dimension": "Bias", "score": 4.0, "reasoning": ""},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, _SAMPLE_QUERY)
        assert result.recommendations == []


# ---------------------------------------------------------------------------
# ReportEvaluator - compute_overall_score
# ---------------------------------------------------------------------------


class TestComputeOverallScore:
    """compute_overall_score produces correct weighted averages."""

    def test_empty_list(self) -> None:
        assert ReportEvaluator.compute_overall_score([]) == 0.0

    def test_single_dimension(self) -> None:
        dims = [DimensionScore(dimension="A", score=4.0, weight=1.0)]
        assert ReportEvaluator.compute_overall_score(dims) == pytest.approx(4.0)

    def test_weighted_average(self) -> None:
        dims = [
            DimensionScore(dimension="A", score=5.0, weight=0.60),
            DimensionScore(dimension="B", score=3.0, weight=0.40),
        ]
        # 5*0.6 + 3*0.4 = 3.0 + 1.2 = 4.2
        assert ReportEvaluator.compute_overall_score(dims) == pytest.approx(4.2)

    def test_all_max_scores(self) -> None:
        dims = [
            DimensionScore(dimension=name, score=5.0, weight=weight)
            for name, weight in EVALUATION_DIMENSIONS
        ]
        assert ReportEvaluator.compute_overall_score(dims) == pytest.approx(5.0)

    def test_all_min_scores(self) -> None:
        dims = [
            DimensionScore(dimension=name, score=1.0, weight=weight)
            for name, weight in EVALUATION_DIMENSIONS
        ]
        assert ReportEvaluator.compute_overall_score(dims) == pytest.approx(1.0)

    def test_mixed_scores(self) -> None:
        dims = [
            DimensionScore(dimension="Factual Accuracy", score=4.0, weight=0.30),
            DimensionScore(dimension="Completeness", score=3.0, weight=0.25),
            DimensionScore(dimension="Coverage", score=5.0, weight=0.20),
            DimensionScore(dimension="Coherence", score=2.0, weight=0.15),
            DimensionScore(dimension="Bias", score=4.0, weight=0.10),
        ]
        # 4*0.3 + 3*0.25 + 5*0.20 + 2*0.15 + 4*0.10
        # = 1.2 + 0.75 + 1.0 + 0.3 + 0.4 = 3.65
        assert ReportEvaluator.compute_overall_score(dims) == pytest.approx(3.65)


# ---------------------------------------------------------------------------
# ReportEvaluator - format_scorecard
# ---------------------------------------------------------------------------


class TestFormatScorecard:
    """format_scorecard produces valid Markdown."""

    def _make_result(self) -> EvaluationResult:
        dims = [
            DimensionScore(
                dimension=name, score=4.0, weight=weight, reasoning="Good."
            )
            for name, weight in EVALUATION_DIMENSIONS
        ]
        return EvaluationResult(
            query=_SAMPLE_QUERY,
            dimensions=dims,
            overall_score=4.0,
            overall_reasoning="Solid report.",
            recommendations=["Add more sources."],
        )

    def test_contains_header(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        assert "# Evaluation Scorecard" in card

    def test_contains_query(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        assert _SAMPLE_QUERY in card

    def test_contains_overall_score(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        assert "4.0/5.0" in card

    def test_contains_table_rows(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        for name, _ in EVALUATION_DIMENSIONS:
            assert name in card

    def test_contains_assessment(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        assert "Solid report." in card

    def test_contains_recommendations(self) -> None:
        evaluator = ReportEvaluator()
        card = evaluator.format_scorecard(self._make_result())
        assert "Add more sources." in card

    def test_no_assessment_when_empty(self) -> None:
        evaluator = ReportEvaluator()
        result = self._make_result()
        result.overall_reasoning = ""
        card = evaluator.format_scorecard(result)
        assert "**Assessment:**" not in card

    def test_no_recommendations_when_empty(self) -> None:
        evaluator = ReportEvaluator()
        result = self._make_result()
        result.recommendations = []
        card = evaluator.format_scorecard(result)
        assert "**Recommendations:**" not in card


# ---------------------------------------------------------------------------
# ReportEvaluator - evaluate (async, with mock LLM)
# ---------------------------------------------------------------------------


class TestEvaluateAsync:
    """evaluate() end-to-end with a mock LLM callable."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self) -> None:
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable(_make_llm_response())
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert isinstance(result, EvaluationResult)
        assert len(result.dimensions) == 5

    @pytest.mark.asyncio
    async def test_evaluate_computes_overall(self) -> None:
        scores = {
            "Factual Accuracy": 4.0,
            "Completeness": 4.0,
            "Coverage": 4.0,
            "Coherence": 4.0,
            "Bias": 4.0,
        }
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable(_make_llm_response(scores=scores))
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert result.overall_score == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_evaluate_without_callable_raises(self) -> None:
        evaluator = ReportEvaluator()
        with pytest.raises(ValueError, match="llm_callable is required"):
            await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT)

    @pytest.mark.asyncio
    async def test_evaluate_parse_error_propagates(self) -> None:
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable("this is not json")
        with pytest.raises(EvaluationParseError):
            await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)

    @pytest.mark.asyncio
    async def test_evaluate_above_threshold(self) -> None:
        scores = {name: 4.5 for name, _ in EVALUATION_DIMENSIONS}
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable(_make_llm_response(scores=scores))
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert result.overall_score >= QUALITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_evaluate_below_threshold(self) -> None:
        scores = {name: 2.0 for name, _ in EVALUATION_DIMENSIONS}
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable(_make_llm_response(scores=scores))
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert result.overall_score < QUALITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_evaluate_passes_query_through(self) -> None:
        evaluator = ReportEvaluator()
        mock_llm = _make_async_callable(_make_llm_response())
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert result.query == _SAMPLE_QUERY

    @pytest.mark.asyncio
    async def test_evaluate_with_markdown_fenced_response(self) -> None:
        evaluator = ReportEvaluator()
        inner = _make_llm_response()
        fenced = f"```json\n{inner}\n```"
        mock_llm = _make_async_callable(fenced)
        result = await evaluator.evaluate(_SAMPLE_QUERY, _SAMPLE_REPORT, mock_llm)
        assert len(result.dimensions) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEvaluationEdgeCases:
    """Edge case handling for evaluation."""

    def test_custom_two_dimensions(self) -> None:
        custom = [("Speed", 0.50), ("Quality", 0.50)]
        evaluator = ReportEvaluator(dimensions=custom)
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Speed", "score": 3.0, "reasoning": "Ok."},
                {"dimension": "Quality", "score": 5.0, "reasoning": "Great."},
            ],
            "overall_reasoning": "Good overall.",
        })
        result = evaluator._parse_evaluation_response(raw, "test")
        # 3*0.5 + 5*0.5 = 4.0
        assert result.overall_score == pytest.approx(4.0)

    def test_integer_scores_accepted(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 4, "reasoning": ""},
                {"dimension": "Completeness", "score": 3, "reasoning": ""},
                {"dimension": "Coverage", "score": 5, "reasoning": ""},
                {"dimension": "Coherence", "score": 2, "reasoning": ""},
                {"dimension": "Bias", "score": 4, "reasoning": ""},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, "test")
        assert len(result.dimensions) == 5

    def test_reasoning_per_dimension_captured(self) -> None:
        evaluator = ReportEvaluator()
        raw = json.dumps({
            "dimensions": [
                {"dimension": "Factual Accuracy", "score": 4.0,
                 "reasoning": "Well-sourced claims."},
                {"dimension": "Completeness", "score": 3.0,
                 "reasoning": "Missing some angles."},
                {"dimension": "Coverage", "score": 4.0,
                 "reasoning": "Broad source base."},
                {"dimension": "Coherence", "score": 4.5,
                 "reasoning": "Clear structure."},
                {"dimension": "Bias", "score": 3.0,
                 "reasoning": "Slightly one-sided."},
            ],
        })
        result = evaluator._parse_evaluation_response(raw, "test")
        reasoning_map = {d.dimension: d.reasoning for d in result.dimensions}
        assert reasoning_map["Factual Accuracy"] == "Well-sourced claims."
        assert reasoning_map["Bias"] == "Slightly one-sided."
