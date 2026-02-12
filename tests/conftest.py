"""Shared pytest fixtures for the research-agent test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# VCR.py configuration (cassette-based HTTP recording/replaying)
# ---------------------------------------------------------------------------


CASSETTE_DIR = Path(__file__).parent / "cassettes"

# Headers that must be stripped from cassette recordings.
FILTERED_HEADERS = [
    "authorization",
    "x-api-key",
    "api-key",
    "openai-api-key",
    "anthropic-api-key",
]


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    """Return VCR.py configuration that filters sensitive headers.

    Uses ``record_mode="none"`` in CI (replay only) and can be
    overridden locally with ``VCR_RECORD_MODE=new_episodes``.
    """
    import os

    record_mode = os.environ.get("VCR_RECORD_MODE", "none")

    return {
        "cassette_library_dir": str(CASSETTE_DIR),
        "filter_headers": FILTERED_HEADERS,
        "record_mode": record_mode,
        "decode_compressed_response": True,
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory for checkpoint storage."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Configuration fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config() -> dict[str, Any]:
    """Return a test-oriented Settings dict with safe defaults.

    Once the ``research_agent.config`` module exposes a ``Settings``
    pydantic-settings model, this fixture should instantiate it directly.
    For now we return a plain dict mirroring *config.yaml* with values
    suitable for fast, offline testing.
    """
    return {
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 0.0,
            "max_tokens": 256,
            "timeout": 5,
            "retries": 1,
        },
        "search": {
            "provider": "tavily",
            "max_results": 3,
            "search_depth": "basic",
        },
        "scraping": {
            "engine": "trafilatura",
            "timeout": 5,
            "max_concurrent": 2,
            "max_content_length": 10_000,
        },
        "embedding": {
            "provider": "sentence_transformers",
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "dimensions": 768,
        },
        "vector_store": {
            "persist_directory": "/tmp/test_chromadb",
            "collection_name": "test_research_docs",
        },
        "costs": {
            "max_cost_per_run": 0.10,
            "max_llm_calls_per_run": 5,
            "warn_at_percentage": 80,
        },
        "checkpoints": {
            "enabled": True,
            "directory": "/tmp/test_checkpoints",
            "save_interval": 1,
            "max_checkpoints": 3,
        },
        "report": {
            "output_dir": "/tmp/test_reports",
            "format": "markdown",
            "max_length": 500,
        },
        "logging": {
            "level": "DEBUG",
            "format": "console",
            "file": None,
        },
    }


# ---------------------------------------------------------------------------
# State fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_state() -> dict[str, Any]:
    """Return a minimal ResearchState dict for testing.

    Field names match the ``ResearchState`` TypedDict in
    ``research_agent.state``.
    """
    return {
        "query": "What are the latest advances in retrieval-augmented generation?",
        "step": "plan",
        "step_index": 0,
        "sub_questions": [],
        "search_results": [],
        "scraped_content": [],
        "summaries": [],
        "final_report": "",
        "sources": [],
        "error_log": [],
    }


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_response() -> str:
    """Return a deterministic mock LLM response string."""
    return (
        "Retrieval-augmented generation (RAG) combines a retriever with a "
        "generative model to produce grounded answers. Recent advances include "
        "adaptive retrieval, self-reflective RAG, and corrective RAG."
    )


@pytest.fixture()
def mock_llm(mock_llm_response: str) -> MagicMock:
    """Return a MagicMock that behaves like a LangChain BaseChatModel.

    The mock's ``invoke`` method returns an ``AIMessage``-like object
    whose ``.content`` attribute holds ``mock_llm_response``.
    """
    message = MagicMock()
    message.content = mock_llm_response

    llm = MagicMock()
    llm.invoke.return_value = message
    return llm
