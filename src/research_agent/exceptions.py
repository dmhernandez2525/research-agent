"""Centralized exception hierarchy for the research-agent package.

All domain-specific exceptions inherit from ``ResearchAgentError`` so
callers can catch the entire family with a single ``except`` clause.
"""

from __future__ import annotations


class ResearchAgentError(Exception):
    """Base exception for all research-agent errors."""


# ---------------------------------------------------------------------------
# Checkpoint errors
# ---------------------------------------------------------------------------


class CheckpointError(ResearchAgentError):
    """Base exception for checkpoint operations."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint fails integrity verification."""


# ---------------------------------------------------------------------------
# Budget errors
# ---------------------------------------------------------------------------


class BudgetExhaustedError(ResearchAgentError):
    """Raised when the research run's cost budget is fully consumed."""


# ---------------------------------------------------------------------------
# Model routing errors
# ---------------------------------------------------------------------------


class ModelRoutingError(ResearchAgentError):
    """Raised when no model is available or all fallbacks fail."""


# ---------------------------------------------------------------------------
# Embedding errors
# ---------------------------------------------------------------------------


class EmbeddingError(ResearchAgentError):
    """Raised when an embedding operation fails."""


# ---------------------------------------------------------------------------
# Scraping errors
# ---------------------------------------------------------------------------


class ScrapingError(ResearchAgentError):
    """Raised when content scraping or extraction fails."""
