"""Observation masking and context compaction.

Implements a rolling window of raw conversation turns with three-stage
progressive masking based on context utilization:

- Stage 1 (75%): Mask raw tool outputs outside window
- Stage 2 (80%): Compress summaries outside window
- Stage 3 (85%): Replace large text with file pointers
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Stage activation thresholds (percentage of max_tokens)
_STAGE_1_THRESHOLD = 75.0
_STAGE_2_THRESHOLD = 80.0
_STAGE_3_THRESHOLD = 85.0

# Minimum content length (chars) to be eligible for file pointer replacement
_FILE_POINTER_MIN_CHARS = 200


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MaskingStage(IntEnum):
    """Progressive masking stages, activated by context utilization."""

    NONE = 0
    STAGE_1 = 1  # Mask raw tool outputs outside window
    STAGE_2 = 2  # Compress assistant summaries outside window
    STAGE_3 = 3  # Replace large text with file pointers


class Turn(BaseModel):
    """A single conversation turn in the research agent's history."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the turn author.",
    )
    content: str = Field(description="Turn content (text or tool output).")
    token_count: int = Field(default=0, ge=0, description="Estimated token count.")
    step_name: str = Field(
        default="", description="Graph node that produced this turn."
    )
    masked: bool = Field(
        default=False,
        description="Whether this turn's content has been replaced with a summary.",
    )


class CompactionResult(BaseModel):
    """Result of a context compaction operation."""

    original_tokens: int = Field(ge=0)
    compacted_tokens: int = Field(ge=0)
    turns_masked: int = Field(ge=0)
    turns_total: int = Field(ge=0)
    stage_applied: MaskingStage = Field(default=MaskingStage.NONE)


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


class ContextManager:
    """Manages a rolling window of conversation turns with observation masking.

    Keeps the most recent ``window_size`` turns in full detail and replaces
    older tool outputs with compact summaries to reduce context usage.

    Attributes:
        window_size: Number of recent turns to keep unmasked.
        max_tokens: Soft token budget for the full context.
    """

    def __init__(
        self,
        window_size: int = 10,
        max_tokens: int = 100_000,
        compaction_cooldown_turns: int = 3,
    ) -> None:
        """Initialize the context manager.

        Args:
            window_size: Number of recent turns to keep in full detail.
            max_tokens: Soft token budget for the context window.
            compaction_cooldown_turns: Minimum number of new turns between
                compaction attempts to prevent repeated O(n) scans.
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.compaction_cooldown_turns = compaction_cooldown_turns
        self._turns: list[Turn] = []
        self._turns_since_compaction: int = 0
        self._compaction_pending = False

    @property
    def turn_count(self) -> int:
        """Return the number of turns tracked.

        Returns:
            Total number of turns.
        """
        return len(self._turns)

    @property
    def turns(self) -> list[Turn]:
        """Return the current list of turns (read-only view).

        Returns:
            List of all turns (masked and unmasked).
        """
        return list(self._turns)

    @property
    def total_tokens(self) -> int:
        """Return estimated total token count across all turns.

        Returns:
            Sum of token counts.
        """
        return sum(t.token_count for t in self._turns)

    @property
    def utilization_percent(self) -> float:
        """Return context utilization as a percentage of max_tokens.

        Returns:
            Utilization between 0.0 and 100.0+.
        """
        if self.max_tokens == 0:
            return 0.0
        return (self.total_tokens / self.max_tokens) * 100.0

    @property
    def active_stage(self) -> MaskingStage:
        """Determine the active masking stage based on context utilization.

        Returns:
            The highest applicable masking stage.
        """
        pct = self.utilization_percent
        if pct >= _STAGE_3_THRESHOLD:
            return MaskingStage.STAGE_3
        if pct >= _STAGE_2_THRESHOLD:
            return MaskingStage.STAGE_2
        if pct >= _STAGE_1_THRESHOLD:
            return MaskingStage.STAGE_1
        return MaskingStage.NONE

    def add_turn(self, turn: Turn) -> None:
        """Append a new turn and trigger compaction if needed.

        Compaction is skipped during the cooldown period to prevent
        repeated O(n) scans when no maskable turns are available.

        Args:
            turn: The turn to add.
        """
        self._turns.append(turn)
        self._turns_since_compaction += 1

        if self._compaction_pending:
            if self._turns_since_compaction < self.compaction_cooldown_turns:
                return
            self._compaction_pending = False

        if self.total_tokens > self.max_tokens:
            result = self.compact()
            self._turns_since_compaction = 0
            if result.turns_masked == 0:
                self._compaction_pending = True

    def compact(self) -> CompactionResult:
        """Apply progressive masking stages to reduce context size.

        Stage 1 (75%): Mask raw tool outputs outside the window.
        Stage 2 (80%): Compress assistant summaries outside the window.
        Stage 3 (85%): Replace large content with file pointers.

        Returns:
            Statistics about the compaction, including the stage applied.
        """
        original_tokens = self.total_tokens
        turns_masked = 0
        stage = self.active_stage
        cutoff = max(0, len(self._turns) - self.window_size)

        # Stage 1: Mask tool outputs outside window
        if stage >= MaskingStage.STAGE_1:
            for i in range(cutoff):
                turn = self._turns[i]
                if turn.role == "tool" and not turn.masked:
                    turn.content = f"[masked tool output from {turn.step_name}]"
                    turn.token_count = 10
                    turn.masked = True
                    turns_masked += 1

        # Stage 2: Compress assistant summaries outside window
        if stage >= MaskingStage.STAGE_2:
            for i in range(cutoff):
                turn = self._turns[i]
                if turn.role == "assistant" and not turn.masked:
                    turn.content = f"[compressed summary from {turn.step_name}]"
                    turn.token_count = 10
                    turn.masked = True
                    turns_masked += 1

        # Stage 3: Replace large text with file pointers
        if stage >= MaskingStage.STAGE_3:
            for i in range(cutoff):
                turn = self._turns[i]
                if not turn.masked and len(turn.content) >= _FILE_POINTER_MIN_CHARS:
                    turn.content = f"[content saved to file; ref: {turn.step_name}]"
                    turn.token_count = 10
                    turn.masked = True
                    turns_masked += 1

        if turns_masked > 0:
            self._compaction_pending = False
            self._turns_since_compaction = 0

        result = CompactionResult(
            original_tokens=original_tokens,
            compacted_tokens=self.total_tokens,
            turns_masked=turns_masked,
            turns_total=len(self._turns),
            stage_applied=stage,
        )
        logger.info(
            "context_compacted",
            original_tokens=result.original_tokens,
            compacted_tokens=result.compacted_tokens,
            turns_masked=result.turns_masked,
            stage=stage.name,
        )
        return result

    def get_context_window(self) -> list[dict[str, Any]]:
        """Return turns formatted for LLM consumption.

        Returns:
            List of dicts with ``role`` and ``content`` keys.
        """
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def format_for_api(
        self,
        system_prompt: str,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Assemble the full API payload with cache-stable ordering.

        Orders content as: system prompt -> tool definitions -> conversation
        history. System content and tool definitions are placed first for
        cache prefix stability.

        Args:
            system_prompt: The system prompt text.
            tool_definitions: Optional list of tool schemas.

        Returns:
            Dict with ``system``, ``tools``, and ``messages`` keys.
        """
        system_block = [
            {
                "type": "text",
                "text": system_prompt,
            }
        ]

        tools: list[dict[str, Any]] = list(tool_definitions) if tool_definitions else []

        messages = self.get_context_window()

        return {
            "system": system_block,
            "tools": tools,
            "messages": messages,
        }

    def window_report(self) -> dict[str, Any]:
        """Return a diagnostic report of the context window state.

        Useful for debugging context usage and masking behavior.

        Returns:
            Dict with turn_count, total_tokens, max_tokens,
            utilization_percent, active_stage, masked_count,
            and unmasked_count.
        """
        masked = sum(1 for t in self._turns if t.masked)
        return {
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization_percent": round(self.utilization_percent, 1),
            "active_stage": self.active_stage.name,
            "masked_count": masked,
            "unmasked_count": self.turn_count - masked,
            "window_size": self.window_size,
        }

    def clear(self) -> None:
        """Remove all turns and reset compaction state."""
        self._turns.clear()
        self._compaction_pending = False
        self._turns_since_compaction = 0
