"""Unit tests for research_agent.memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from research_agent.embeddings import SimilarityResult
from research_agent.memory import MemoryEntry, ResearchMemory

# ---------------------------------------------------------------------------
# TestMemoryEntry
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    """MemoryEntry model defaults and construction."""

    def test_default_values(self) -> None:
        entry = MemoryEntry(content="Test finding")
        assert entry.content == "Test finding"
        assert entry.query == ""
        assert entry.score == 0.0
        assert entry.is_stale is False

    def test_custom_values(self) -> None:
        entry = MemoryEntry(
            content="Finding",
            query="test query",
            timestamp="2025-01-01T00:00:00+00:00",
            score=0.92,
            is_stale=True,
        )
        assert entry.score == 0.92
        assert entry.is_stale is True


# ---------------------------------------------------------------------------
# TestResearchMemoryInit
# ---------------------------------------------------------------------------


class TestResearchMemoryInit:
    """ResearchMemory initialization and configuration."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_default_values(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory()
        assert memory.relevance_threshold == 0.80
        assert memory.staleness_days == 30
        assert memory.max_results == 5

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_custom_values(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory(
            relevance_threshold=0.9,
            staleness_days=60,
            max_results=10,
        )
        assert memory.relevance_threshold == 0.9
        assert memory.staleness_days == 60
        assert memory.max_results == 10

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_creates_embeddings_with_collection(
        self, mock_embed_cls: MagicMock
    ) -> None:
        ResearchMemory(
            persist_directory="/tmp/test",
            collection_name="custom_memory",
        )
        mock_embed_cls.assert_called_once_with(
            collection_name="custom_memory",
            persist_directory="/tmp/test",
        )


# ---------------------------------------------------------------------------
# TestStore
# ---------------------------------------------------------------------------


class TestStore:
    """ResearchMemory.store() saves findings."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_stores_findings(self, mock_embed_cls: MagicMock) -> None:
        mock_embeddings = MagicMock()
        mock_embeddings.add_documents.return_value = 2
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        count = memory.store(
            findings=["Finding 1", "Finding 2"],
            query="test query",
        )
        assert count == 2
        mock_embeddings.add_documents.assert_called_once()

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_returns_zero_for_empty_findings(
        self, mock_embed_cls: MagicMock
    ) -> None:
        memory = ResearchMemory()
        count = memory.store(findings=[], query="test")
        assert count == 0

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_skips_blank_findings(self, mock_embed_cls: MagicMock) -> None:
        mock_embeddings = MagicMock()
        mock_embeddings.add_documents.return_value = 1
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        count = memory.store(
            findings=["Valid finding", "  ", ""],
            query="test",
        )
        assert count == 1
        # Should only have 1 document passed to add_documents
        docs = mock_embeddings.add_documents.call_args[0][0]
        assert len(docs) == 1

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_includes_metadata(self, mock_embed_cls: MagicMock) -> None:
        mock_embeddings = MagicMock()
        mock_embeddings.add_documents.return_value = 1
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        memory.store(
            findings=["A finding"],
            query="test query",
            metadata={"run_id": "run-123"},
        )
        docs = mock_embeddings.add_documents.call_args[0][0]
        assert docs[0].metadata["query"] == "test query"
        assert docs[0].metadata["run_id"] == "run-123"
        assert "stored_at" in docs[0].metadata


# ---------------------------------------------------------------------------
# TestRecall
# ---------------------------------------------------------------------------


class TestRecall:
    """ResearchMemory.recall() retrieves relevant memories."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_returns_entries_above_threshold(
        self, mock_embed_cls: MagicMock
    ) -> None:
        now = datetime.now(tz=UTC)
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [
            SimilarityResult(
                id="1",
                content="High relevance finding",
                score=0.92,
                metadata={"query": "q1", "stored_at": now.isoformat()},
            ),
            SimilarityResult(
                id="2",
                content="Low relevance",
                score=0.5,
                metadata={"query": "q2", "stored_at": now.isoformat()},
            ),
        ]
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory(relevance_threshold=0.8)
        entries = memory.recall("test query")
        assert len(entries) == 1
        assert entries[0].content == "High relevance finding"
        assert entries[0].score == 0.92

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_returns_empty_on_no_matches(
        self, mock_embed_cls: MagicMock
    ) -> None:
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        entries = memory.recall("obscure query")
        assert entries == []

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_flags_stale_entries(self, mock_embed_cls: MagicMock) -> None:
        old_date = (datetime.now(tz=UTC) - timedelta(days=60)).isoformat()
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [
            SimilarityResult(
                id="1",
                content="Old finding",
                score=0.95,
                metadata={"query": "q", "stored_at": old_date},
            ),
        ]
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory(staleness_days=30)
        entries = memory.recall("test")
        assert len(entries) == 1
        assert entries[0].is_stale is True

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_recent_entries_not_stale(
        self, mock_embed_cls: MagicMock
    ) -> None:
        now = datetime.now(tz=UTC).isoformat()
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [
            SimilarityResult(
                id="1",
                content="Fresh finding",
                score=0.9,
                metadata={"query": "q", "stored_at": now},
            ),
        ]
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory(staleness_days=30)
        entries = memory.recall("test")
        assert entries[0].is_stale is False


# ---------------------------------------------------------------------------
# TestCheckStaleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    """ResearchMemory._check_staleness() validates entry age."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_empty_timestamp_is_stale(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory()
        assert memory._check_staleness("", datetime.now(tz=UTC)) is True

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_invalid_timestamp_is_stale(
        self, mock_embed_cls: MagicMock
    ) -> None:
        memory = ResearchMemory()
        assert memory._check_staleness("not-a-date", datetime.now(tz=UTC)) is True

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_old_entry_is_stale(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory(staleness_days=30)
        now = datetime.now(tz=UTC)
        old = (now - timedelta(days=31)).isoformat()
        assert memory._check_staleness(old, now) is True

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_recent_entry_not_stale(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory(staleness_days=30)
        now = datetime.now(tz=UTC)
        recent = (now - timedelta(days=5)).isoformat()
        assert memory._check_staleness(recent, now) is False


# ---------------------------------------------------------------------------
# TestFormatContext
# ---------------------------------------------------------------------------


class TestFormatContext:
    """ResearchMemory.format_context() produces planner-ready text."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_formats_entries(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory()
        entries = [
            MemoryEntry(content="Finding A", score=0.9),
            MemoryEntry(content="Finding B", score=0.85),
        ]
        result = memory.format_context(entries)
        assert "Previous research findings:" in result
        assert "- Finding A" in result
        assert "- Finding B" in result

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_marks_stale_entries(self, mock_embed_cls: MagicMock) -> None:
        memory = ResearchMemory()
        entries = [
            MemoryEntry(content="Old finding", score=0.9, is_stale=True),
        ]
        result = memory.format_context(entries)
        assert "[stale]" in result

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_empty_entries_returns_empty(
        self, mock_embed_cls: MagicMock
    ) -> None:
        memory = ResearchMemory()
        result = memory.format_context([])
        assert result == ""


# ---------------------------------------------------------------------------
# TestCountAndClear
# ---------------------------------------------------------------------------


class TestCountAndClear:
    """Count and clear operations."""

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_count_delegates(self, mock_embed_cls: MagicMock) -> None:
        mock_embeddings = MagicMock()
        mock_embeddings.count = 42
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        assert memory.count == 42

    @patch("research_agent.memory.ResearchEmbeddings")
    def test_clear_deletes_collection(
        self, mock_embed_cls: MagicMock
    ) -> None:
        mock_embeddings = MagicMock()
        mock_embed_cls.return_value = mock_embeddings

        memory = ResearchMemory()
        memory.clear()
        mock_embeddings.delete_collection.assert_called_once()
