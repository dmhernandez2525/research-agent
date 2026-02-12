"""Self-evaluation using LLM-as-judge with 5-dimension scoring.

Dimensions and weights:
    - Factual Accuracy  (30%)
    - Completeness      (25%)
    - Coverage          (20%)
    - Coherence         (15%)
    - Bias              (10%)
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Protocol

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str = Field(description="Name of the evaluation dimension.")
    score: float = Field(
        ge=1.0, le=5.0, description="Score on a 1-5 scale per SDD-005."
    )
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
        default=0.0, ge=0.0, le=5.0, description="Weighted overall score (1-5)."
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

QUALITY_THRESHOLD: float = 3.5
"""Minimum weighted overall score (1-5) for a report to pass evaluation.
Reports below this trigger up to 2 auto-revision cycles per SDD-005."""

MAX_REVISION_CYCLES: int = 2
"""Maximum number of auto-revision attempts for reports below threshold."""

# Per-dimension descriptions used in the evaluation prompt.
_DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "Factual Accuracy": (
        "Are claims supported by the cited sources? Are there any "
        "unsupported assertions or factual errors?"
    ),
    "Completeness": (
        "Does the report address all aspects of the research query? "
        "Are there significant gaps in coverage?"
    ),
    "Coverage": (
        "Does the report draw on a sufficient breadth of sources? "
        "Are multiple perspectives represented?"
    ),
    "Coherence": (
        "Is the report well-organized with clear logical flow? "
        "Are transitions smooth and arguments well-structured?"
    ),
    "Bias": (
        "Does the report present a balanced perspective? "
        "Are opposing viewpoints fairly represented?"
    ),
}


class LLMCallable(Protocol):
    """Protocol for an async callable that sends a prompt to an LLM."""

    async def __call__(self, prompt: str) -> str: ...


class RevisionCallable(Protocol):
    """Protocol for an async callable that revises a report given feedback."""

    async def __call__(self, report: str, feedback: str) -> str: ...


class EvaluationParseError(Exception):
    """Raised when the LLM evaluation response cannot be parsed."""


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

        Constructs a prompt that asks the LLM to score the report across
        all configured dimensions on a 1-5 scale, returning structured
        JSON with per-dimension scores, reasoning, and recommendations.

        Args:
            query: The original research query.
            report: The generated report text.

        Returns:
            Formatted evaluation prompt string.
        """
        dim_lines: list[str] = []
        for name, weight in self.dimensions:
            desc = _DIMENSION_DESCRIPTIONS.get(name, "Evaluate this dimension.")
            pct = f"{weight:.0%}"
            dim_lines.append(f"  - {name} (weight: {pct}): {desc}")

        dimensions_block = "\n".join(dim_lines)

        return textwrap.dedent("""\
            You are an expert research report evaluator. Score the following
            report on each dimension using a 1-5 scale where:
              1 = Very Poor, 2 = Poor, 3 = Adequate, 4 = Good, 5 = Excellent

            Dimensions to evaluate:
            {dimensions}

            Research Query:
            {query}

            Report:
            {report}

            Respond with ONLY valid JSON in this exact format (no markdown fencing):
            {{
              "dimensions": [
                {{
                  "dimension": "<dimension name>",
                  "score": <1-5>,
                  "reasoning": "<1-2 sentence explanation>"
                }}
              ],
              "overall_reasoning": "<1-2 sentence overall assessment>",
              "recommendations": ["<specific improvement 1>", "<specific improvement 2>"]
            }}
        """).format(
            dimensions=dimensions_block,
            query=query,
            report=report,
        )

    def _parse_evaluation_response(
        self,
        raw: str,
        query: str,
    ) -> EvaluationResult:
        """Parse the LLM's JSON response into an EvaluationResult.

        Validates that every configured dimension has a score and that
        scores are within the 1-5 range.

        Args:
            raw: Raw JSON string from the LLM.
            query: The original research query (for the result).

        Returns:
            Parsed and validated EvaluationResult.

        Raises:
            EvaluationParseError: If the response is malformed or missing
                required fields.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (e.g., ```json) and closing fence (```)
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise EvaluationParseError(
                f"LLM response is not valid JSON: {exc}"
            ) from exc

        if "dimensions" not in data:
            raise EvaluationParseError("Missing 'dimensions' key in response")

        # Build a weight lookup from configured dimensions
        weight_map = {name: weight for name, weight in self.dimensions}

        dim_scores: list[DimensionScore] = []
        for entry in data["dimensions"]:
            name = entry.get("dimension", "")
            score_val = entry.get("score")
            reasoning = entry.get("reasoning", "")

            if name not in weight_map:
                logger.warning(
                    "unexpected_dimension_in_response",
                    dimension=name,
                )
                continue

            if score_val is None:
                raise EvaluationParseError(
                    f"Missing score for dimension '{name}'"
                )

            score_float = float(score_val)
            score_float = max(1.0, min(5.0, score_float))

            dim_scores.append(
                DimensionScore(
                    dimension=name,
                    score=score_float,
                    weight=weight_map[name],
                    reasoning=reasoning,
                )
            )

        # Check for missing dimensions
        returned_names = {d.dimension for d in dim_scores}
        for name, weight in self.dimensions:
            if name not in returned_names:
                logger.warning(
                    "missing_dimension_in_response",
                    dimension=name,
                )
                dim_scores.append(
                    DimensionScore(
                        dimension=name,
                        score=1.0,
                        weight=weight,
                        reasoning="Not scored by evaluator; defaulted to 1.0.",
                    )
                )

        overall = self.compute_overall_score(dim_scores)

        return EvaluationResult(
            query=query,
            dimensions=dim_scores,
            overall_score=round(overall, 2),
            overall_reasoning=data.get("overall_reasoning", ""),
            recommendations=data.get("recommendations", []),
        )

    async def evaluate(
        self,
        query: str,
        report: str,
        llm_callable: LLMCallable | None = None,
    ) -> EvaluationResult:
        """Evaluate a research report using LLM-as-judge.

        Builds an evaluation prompt, sends it to the LLM, and parses the
        structured JSON response into scored dimensions.

        Args:
            query: The original research query.
            report: The generated report text.
            llm_callable: Async callable that sends a prompt to an LLM
                and returns the response string. Required for now; later
                phases will wire this to the ModelRouter automatically.

        Returns:
            An ``EvaluationResult`` with per-dimension and overall scores.

        Raises:
            EvaluationParseError: If the LLM response cannot be parsed.
            ValueError: If no llm_callable is provided.
        """
        if llm_callable is None:
            msg = "llm_callable is required (ModelRouter integration is a later phase)"
            raise ValueError(msg)

        prompt = self._build_evaluation_prompt(query, report)
        logger.info("evaluation_prompt_built", query=query, prompt_len=len(prompt))

        raw_response = await llm_callable(prompt)
        logger.debug("evaluation_response_received", response_len=len(raw_response))

        result = self._parse_evaluation_response(raw_response, query)
        logger.info(
            "evaluation_complete",
            overall_score=result.overall_score,
            passed=result.overall_score >= QUALITY_THRESHOLD,
        )
        return result

    @staticmethod
    def compute_overall_score(dimensions: list[DimensionScore]) -> float:
        """Compute the weighted overall score from dimension scores.

        Args:
            dimensions: List of scored dimensions.

        Returns:
            Weighted average score (1-5 range when weights sum to 1.0).
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
            f"**Overall Score:** {result.overall_score:.1f}/5.0",
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


# ---------------------------------------------------------------------------
# Revision models
# ---------------------------------------------------------------------------


class RevisionRecord(BaseModel):
    """Record of a single revision cycle."""

    cycle: int = Field(ge=0, description="Revision cycle number (0 = initial).")
    report: str = Field(description="Report text for this cycle.")
    evaluation: EvaluationResult = Field(description="Evaluation result for this cycle.")


class RevisionResult(BaseModel):
    """Full result of the auto-revision process."""

    best_report: str = Field(description="Report text with the highest overall score.")
    best_evaluation: EvaluationResult = Field(
        description="Evaluation of the best report."
    )
    total_cycles: int = Field(ge=0, description="Number of revision cycles performed.")
    passed: bool = Field(description="Whether the best report met the quality threshold.")
    history: list[RevisionRecord] = Field(
        default_factory=list, description="Full revision history."
    )
    stop_reason: str = Field(
        default="", description="Why the revision loop stopped."
    )


# ---------------------------------------------------------------------------
# Revision Manager
# ---------------------------------------------------------------------------


_DIMINISHING_RETURNS_THRESHOLD = 0.1
"""Minimum score improvement between cycles to continue revising."""


class RevisionManager:
    """Orchestrates auto-revision cycles for below-threshold reports.

    Evaluates a report, and if it falls below ``QUALITY_THRESHOLD``,
    feeds evaluation feedback into a revision callable for up to
    ``MAX_REVISION_CYCLES`` iterations. Stops early on diminishing
    returns (score improvement below threshold) or if the report passes.

    Attributes:
        evaluator: The ``ReportEvaluator`` used for scoring.
        max_cycles: Maximum number of revision cycles.
        quality_threshold: Minimum score to pass.
        min_improvement: Minimum score gain to continue revising.
    """

    def __init__(
        self,
        evaluator: ReportEvaluator | None = None,
        max_cycles: int = MAX_REVISION_CYCLES,
        quality_threshold: float = QUALITY_THRESHOLD,
        min_improvement: float = _DIMINISHING_RETURNS_THRESHOLD,
    ) -> None:
        """Initialize the revision manager.

        Args:
            evaluator: Evaluator to use. Defaults to a fresh ``ReportEvaluator``.
            max_cycles: Maximum revision cycles (default from constant).
            quality_threshold: Score threshold for passing (default 3.5).
            min_improvement: Minimum score improvement to continue (default 0.1).
        """
        self.evaluator = evaluator or ReportEvaluator()
        self.max_cycles = max_cycles
        self.quality_threshold = quality_threshold
        self.min_improvement = min_improvement

    @staticmethod
    def _build_revision_feedback(evaluation: EvaluationResult) -> str:
        """Format evaluation results into actionable revision feedback.

        Args:
            evaluation: The evaluation result to convert to feedback.

        Returns:
            Human-readable feedback string for the revision prompt.
        """
        lines: list[str] = [
            f"Overall score: {evaluation.overall_score:.1f}/5.0 "
            f"(threshold: {QUALITY_THRESHOLD:.1f})",
            "",
            "Per-dimension feedback:",
        ]
        for dim in evaluation.dimensions:
            lines.append(f"  - {dim.dimension} ({dim.score:.1f}/5): {dim.reasoning}")

        if evaluation.overall_reasoning:
            lines.append("")
            lines.append(f"Assessment: {evaluation.overall_reasoning}")

        if evaluation.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in evaluation.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def should_revise(
        self,
        score: float,
        cycle: int,
        previous_score: float | None = None,
    ) -> bool:
        """Determine whether another revision cycle is warranted.

        Args:
            score: Current overall score.
            cycle: Current cycle number (0-indexed).
            previous_score: Score from the previous cycle (None for initial).

        Returns:
            True if revision should continue.
        """
        if score >= self.quality_threshold:
            return False
        if cycle >= self.max_cycles:
            return False
        if previous_score is not None:
            improvement = score - previous_score
            if improvement < self.min_improvement:
                return False
        return True

    async def run(
        self,
        query: str,
        report: str,
        llm_callable: LLMCallable,
        revision_callable: RevisionCallable,
    ) -> RevisionResult:
        """Run the evaluation-revision loop.

        Evaluates the initial report, then iterates with revisions
        until the report passes, max cycles are reached, or diminishing
        returns are detected.

        Args:
            query: The original research query.
            report: The initial report text.
            llm_callable: Async callable for LLM evaluation.
            revision_callable: Async callable for report revision.

        Returns:
            ``RevisionResult`` with the best report and full history.
        """
        history: list[RevisionRecord] = []
        current_report = report
        best_report = report
        best_score = 0.0
        best_evaluation: EvaluationResult | None = None
        previous_score: float | None = None
        stop_reason = ""

        for cycle in range(self.max_cycles + 1):
            evaluation = await self.evaluator.evaluate(
                query, current_report, llm_callable
            )

            record = RevisionRecord(
                cycle=cycle,
                report=current_report,
                evaluation=evaluation,
            )
            history.append(record)

            if evaluation.overall_score > best_score:
                best_score = evaluation.overall_score
                best_report = current_report
                best_evaluation = evaluation

            logger.info(
                "revision_cycle_complete",
                cycle=cycle,
                score=evaluation.overall_score,
                best_score=best_score,
                passed=evaluation.overall_score >= self.quality_threshold,
            )

            if not self.should_revise(evaluation.overall_score, cycle, previous_score):
                if evaluation.overall_score >= self.quality_threshold:
                    stop_reason = "passed"
                elif cycle >= self.max_cycles:
                    stop_reason = "max_cycles_reached"
                else:
                    stop_reason = "diminishing_returns"
                break

            feedback = self._build_revision_feedback(evaluation)
            current_report = await revision_callable(current_report, feedback)
            previous_score = evaluation.overall_score

        if best_evaluation is None:
            best_evaluation = EvaluationResult(query=query)

        return RevisionResult(
            best_report=best_report,
            best_evaluation=best_evaluation,
            total_cycles=len(history) - 1,
            passed=best_score >= self.quality_threshold,
            history=history,
            stop_reason=stop_reason,
        )
