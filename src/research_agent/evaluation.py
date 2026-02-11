"""Self-evaluation using LLM-as-judge with 5-dimension scoring.

Dimensions and weights:
    - Factual Accuracy  (30%)
    - Completeness      (25%)
    - Coverage          (20%)
    - Coherence         (15%)
    - Bias              (10%)
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str = Field(description="Name of the evaluation dimension.")
    score: float = Field(ge=0.0, le=10.0, description="Score on a 0-10 scale.")
    weight: float = Field(
        ge=0.0, le=1.0, description="Weight of this dimension in the overall score."
    )
    reasoning: str = Field(
        default="", description="Explanation for the assigned score."
    )

    @property
    def weighted_score(self) -> float:
        """Return the weighted contribution of this dimension.

        Returns:
            score * weight.
        """
        return self.score * self.weight


class EvaluationResult(BaseModel):
    """Full evaluation result with per-dimension scores."""

    query: str = Field(description="The original research query.")
    dimensions: list[DimensionScore] = Field(
        default_factory=list, description="Per-dimension scores."
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Weighted overall score (0-10)."
    )
    overall_reasoning: str = Field(
        default="", description="High-level assessment of the report."
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific recommendations for improvement.",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


EVALUATION_DIMENSIONS: list[tuple[str, float]] = [
    ("Factual Accuracy", 0.30),
    ("Completeness", 0.25),
    ("Coverage", 0.20),
    ("Coherence", 0.15),
    ("Bias", 0.10),
]
"""Evaluation dimensions with their weights (must sum to 1.0)."""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ReportEvaluator:
    """Evaluates research reports using LLM-as-judge scoring.

    Uses a secondary LLM call to score a generated report across five
    dimensions, producing a weighted overall quality score.
    """

    def __init__(
        self,
        dimensions: list[tuple[str, float]] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            dimensions: Optional custom dimensions as ``(name, weight)``
                tuples. Defaults to ``EVALUATION_DIMENSIONS``.
        """
        self.dimensions = dimensions or list(EVALUATION_DIMENSIONS)
        total_weight = sum(w for _, w in self.dimensions)
        if abs(total_weight - 1.0) > 0.01:
            msg = f"Dimension weights must sum to 1.0, got {total_weight:.2f}"
            raise ValueError(msg)

    def _build_evaluation_prompt(
        self,
        query: str,
        report: str,
    ) -> str:
        """Build the evaluation prompt for the LLM judge.

        Args:
            query: The original research query.
            report: The generated report text.

        Returns:
            Formatted evaluation prompt string.

        Raises:
            NotImplementedError: Stub -- full implementation pending.
        """
        raise NotImplementedError("_build_evaluation_prompt is not yet implemented")

    async def evaluate(
        self,
        query: str,
        report: str,
    ) -> EvaluationResult:
        """Evaluate a research report using LLM-as-judge.

        Args:
            query: The original research query.
            report: The generated report text.

        Returns:
            An ``EvaluationResult`` with per-dimension and overall scores.

        Raises:
            NotImplementedError: Stub -- full implementation pending.
        """
        raise NotImplementedError("evaluate is not yet implemented")

    @staticmethod
    def compute_overall_score(dimensions: list[DimensionScore]) -> float:
        """Compute the weighted overall score from dimension scores.

        Args:
            dimensions: List of scored dimensions.

        Returns:
            Weighted average score (0-10).
        """
        if not dimensions:
            return 0.0
        return sum(d.weighted_score for d in dimensions)

    def format_scorecard(self, result: EvaluationResult) -> str:
        """Format an evaluation result as a human-readable scorecard.

        Args:
            result: The evaluation result.

        Returns:
            Formatted Markdown scorecard string.
        """
        lines: list[str] = [
            "# Evaluation Scorecard",
            f"**Query:** {result.query}",
            f"**Overall Score:** {result.overall_score:.1f}/10.0",
            "",
            "| Dimension | Score | Weight | Weighted |",
            "|-----------|-------|--------|----------|",
        ]
        for dim in result.dimensions:
            lines.append(
                f"| {dim.dimension} | {dim.score:.1f} | {dim.weight:.0%} "
                f"| {dim.weighted_score:.2f} |"
            )
        lines.append("")
        if result.overall_reasoning:
            lines.append(f"**Assessment:** {result.overall_reasoning}")
            lines.append("")
        if result.recommendations:
            lines.append("**Recommendations:**")
            for rec in result.recommendations:
                lines.append(f"- {rec}")
        return "\n".join(lines)
