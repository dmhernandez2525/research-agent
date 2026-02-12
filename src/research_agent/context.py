"""Observation masking and context compaction.

Implements a rolling window of raw conversation turns, masking older tool
outputs to stay within context limits while preserving recent detail.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


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
        """Mask older tool outputs to reduce context size.

        Keeps the most recent ``window_size`` turns unmasked. For older
        turns with ``role="tool"``, replaces content with a brief summary
        placeholder.

        Returns:
            Statistics about the compaction.
        """
        original_tokens = self.total_tokens
        turns_masked = 0
        cutoff = max(0, len(self._turns) - self.window_size)

        for i in range(cutoff):
            turn = self._turns[i]
            if turn.role == "tool" and not turn.masked:
                turn.content = f"[masked tool output from {turn.step_name}]"
                turn.token_count = 10  # approximate
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
        )
        logger.info(
            "context_compacted",
            original_tokens=result.original_tokens,
            compacted_tokens=result.compacted_tokens,
            turns_masked=result.turns_masked,
        )
        return result

    def get_context_window(self) -> list[dict[str, Any]]:
        """Return turns formatted for LLM consumption.

        Returns:
            List of dicts with ``role`` and ``content`` keys.
        """
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        """Remove all turns."""
        self._turns.clear()
