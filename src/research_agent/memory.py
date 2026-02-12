"""Cross-session research memory using ChromaDB persistence.

Stores and retrieves key findings across research sessions, enabling
the agent to build on previous knowledge. Uses a dedicated ChromaDB
collection with metadata for staleness tracking.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from research_agent.embeddings import (
    EmbeddingDocument,
    ResearchEmbeddings,
    SimilarityResult,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_COLLECTION = "research_memory"
_DEFAULT_STALENESS_DAYS = 30
_DEFAULT_RELEVANCE_THRESHOLD = 0.80
_DEFAULT_MAX_RESULTS = 5


class MemoryEntry(BaseModel):
    """A stored memory entry with metadata."""

    content: str = Field(description="The key finding or knowledge.")
    query: str = Field(default="", description="The research query that produced this.")
    timestamp: str = Field(default="", description="ISO timestamp when stored.")
    score: float = Field(default=0.0, description="Relevance score from retrieval.")
    is_stale: bool = Field(default=False, description="Whether the entry is older than retention period.")


class ResearchMemory:
    """Cross-session memory backed by ChromaDB.

    Wraps ``ResearchEmbeddings`` with a dedicated collection for storing
    key findings. Supports relevance-based retrieval, staleness tracking,
    and configurable retention.

    Attributes:
        relevance_threshold: Minimum similarity score for retrieval.
        staleness_days: Number of days before entries are flagged as stale.
        max_results: Maximum number of results per query.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = _DEFAULT_COLLECTION,
        relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
        staleness_days: int = _DEFAULT_STALENESS_DAYS,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        """Initialize the research memory.

        Args:
            persist_directory: ChromaDB persistence directory.
            collection_name: Name of the memory collection.
            relevance_threshold: Minimum similarity for retrieval (0-1).
            staleness_days: Days before entries are flagged stale.
            max_results: Max results per query.
        """
        self.relevance_threshold = relevance_threshold
        self.staleness_days = staleness_days
        self.max_results = max_results
        self._embeddings = ResearchEmbeddings(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    def store(
        self,
        findings: list[str],
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store key findings in memory.

        Each finding is stored as a separate document with metadata
        including the source query, timestamp, and any additional metadata.

        Args:
            findings: List of key finding strings to store.
            query: The research query that produced these findings.
            metadata: Optional additional metadata per finding.

        Returns:
            Number of findings successfully stored.
        """
        if not findings:
            return 0

        now = datetime.now(tz=UTC).isoformat()
        docs: list[EmbeddingDocument] = []

        for i, finding in enumerate(findings):
            if not finding.strip():
                continue

            doc_id = f"mem-{int(time.time())}-{i}"
            doc_meta: dict[str, Any] = {
                "query": query,
                "stored_at": now,
                "type": "finding",
            }
            if metadata:
                doc_meta.update(metadata)

            docs.append(
                EmbeddingDocument(
                    id=doc_id,
                    content=finding.strip(),
                    metadata=doc_meta,
                )
            )

        if not docs:
            return 0

        added = self._embeddings.add_documents(docs)

        logger.info(
            "memory_stored",
            query=query,
            findings_count=added,
        )

        return added

    def recall(self, query: str) -> list[MemoryEntry]:
        """Retrieve relevant memories for a query.

        Returns entries with similarity above the relevance threshold,
        with staleness flags for entries older than the retention period.

        Args:
            query: The research query to search memories for.

        Returns:
            List of MemoryEntry objects sorted by relevance.
        """
        results: list[SimilarityResult] = self._embeddings.search(
            query=query,
            n_results=self.max_results,
        )

        now = datetime.now(tz=UTC)
        entries: list[MemoryEntry] = []

        for result in results:
            if result.score < self.relevance_threshold:
                continue

            stored_at = result.metadata.get("stored_at", "")
            is_stale = self._check_staleness(stored_at, now)

            entries.append(
                MemoryEntry(
                    content=result.content,
                    query=result.metadata.get("query", ""),
                    timestamp=stored_at,
                    score=result.score,
                    is_stale=is_stale,
                )
            )

        logger.info(
            "memory_recalled",
            query=query,
            results_count=len(entries),
            stale_count=sum(1 for e in entries if e.is_stale),
        )

        return entries

    def _check_staleness(self, stored_at: str, now: datetime) -> bool:
        """Check if an entry is older than the staleness period.

        Args:
            stored_at: ISO timestamp string of when the entry was stored.
            now: Current datetime for comparison.

        Returns:
            True if the entry is stale, False otherwise.
        """
        if not stored_at:
            return True

        try:
            stored_dt = datetime.fromisoformat(stored_at)
            age_days = (now - stored_dt).total_seconds() / 86400
            return age_days > self.staleness_days
        except (ValueError, TypeError):
            return True

    def format_context(self, entries: list[MemoryEntry]) -> str:
        """Format memory entries as context for the planner/searcher.

        Produces a text block suitable for including in LLM system prompts.
        Stale entries are noted as such.

        Args:
            entries: List of retrieved memory entries.

        Returns:
            Formatted context string, or empty string if no entries.
        """
        if not entries:
            return ""

        lines = ["Previous research findings:"]
        for entry in entries:
            staleness_note = " [stale]" if entry.is_stale else ""
            lines.append(f"- {entry.content}{staleness_note}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Return the number of entries in memory.

        Returns:
            Entry count.
        """
        return self._embeddings.count

    def clear(self) -> None:
        """Clear all memory entries."""
        self._embeddings.delete_collection()
        logger.info("memory_cleared")
