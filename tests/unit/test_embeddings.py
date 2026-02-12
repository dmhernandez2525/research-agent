"""Unit tests for research_agent.embeddings - models, init, and method behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from research_agent.embeddings import (
    DeduplicationResult,
    EmbeddingDocument,
    ResearchEmbeddings,
    SimilarityResult,
)
from research_agent.exceptions import EmbeddingError

# ---------------------------------------------------------------------------
# EmbeddingDocument model
# ---------------------------------------------------------------------------


class TestEmbeddingDocument:
    """EmbeddingDocument validates fields and provides defaults."""

    def test_valid_construction(self) -> None:
        doc = EmbeddingDocument(id="doc-1", content="Hello world")
        assert doc.id == "doc-1"
        assert doc.content == "Hello world"

    def test_default_metadata_is_empty_dict(self) -> None:
        doc = EmbeddingDocument(id="doc-1", content="text")
        assert doc.metadata == {}

    def test_metadata_accepted(self) -> None:
        doc = EmbeddingDocument(id="doc-1", content="text", metadata={"source": "web"})
        assert doc.metadata["source"] == "web"

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingDocument(content="text")  # type: ignore[call-arg]

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingDocument(id="doc-1")  # type: ignore[call-arg]

    def test_metadata_default_factory_isolation(self) -> None:
        """Each instance gets its own metadata dict."""
        doc1 = EmbeddingDocument(id="a", content="x")
        doc2 = EmbeddingDocument(id="b", content="y")
        doc1.metadata["key"] = "val"
        assert "key" not in doc2.metadata


# ---------------------------------------------------------------------------
# SimilarityResult model
# ---------------------------------------------------------------------------


class TestSimilarityResult:
    """SimilarityResult validates fields and provides defaults."""

    def test_valid_construction(self) -> None:
        result = SimilarityResult(id="doc-1", content="text", score=0.95)
        assert result.id == "doc-1"
        assert result.score == 0.95

    def test_defaults(self) -> None:
        result = SimilarityResult(id="doc-1")
        assert result.content == ""
        assert result.score == 0.0
        assert result.metadata == {}

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarityResult()  # type: ignore[call-arg]

    def test_metadata_accepted(self) -> None:
        result = SimilarityResult(id="doc-1", metadata={"chunk": 0})
        assert result.metadata["chunk"] == 0


# ---------------------------------------------------------------------------
# DeduplicationResult model
# ---------------------------------------------------------------------------


class TestDeduplicationResult:
    """DeduplicationResult validates fields, constraints, and defaults."""

    def test_defaults(self) -> None:
        result = DeduplicationResult()
        assert result.is_duplicate is False
        assert result.most_similar_id is None
        assert result.similarity_score == 0.0

    def test_duplicate_detected(self) -> None:
        result = DeduplicationResult(
            is_duplicate=True,
            most_similar_id="doc-42",
            similarity_score=0.92,
        )
        assert result.is_duplicate is True
        assert result.most_similar_id == "doc-42"
        assert result.similarity_score == 0.92

    def test_similarity_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DeduplicationResult(similarity_score=-0.1)

    def test_similarity_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            DeduplicationResult(similarity_score=1.5)

    def test_boundary_scores_accepted(self) -> None:
        low = DeduplicationResult(similarity_score=0.0)
        high = DeduplicationResult(similarity_score=1.0)
        assert low.similarity_score == 0.0
        assert high.similarity_score == 1.0


# ---------------------------------------------------------------------------
# ResearchEmbeddings initialization
# ---------------------------------------------------------------------------


class TestResearchEmbeddingsInit:
    """ResearchEmbeddings stores configuration and lazy-init fields."""

    def test_default_values(self) -> None:
        emb = ResearchEmbeddings()
        assert emb.collection_name == "research_docs"
        assert emb.persist_directory == "./data/chromadb"
        assert emb.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert emb.dimensions == 768
        assert emb.content_dedup_threshold == 0.85
        assert emb.exact_dedup_threshold == 0.95

    def test_custom_values(self) -> None:
        emb = ResearchEmbeddings(
            collection_name="my_collection",
            persist_directory="/tmp/test_db",
            model_name="custom-model",
            dimensions=384,
            content_dedup_threshold=0.80,
            exact_dedup_threshold=0.99,
        )
        assert emb.collection_name == "my_collection"
        assert emb.persist_directory == "/tmp/test_db"
        assert emb.model_name == "custom-model"
        assert emb.dimensions == 384
        assert emb.content_dedup_threshold == 0.80
        assert emb.exact_dedup_threshold == 0.99

    def test_lazy_fields_initially_none(self) -> None:
        emb = ResearchEmbeddings()
        assert emb._client is None
        assert emb._collection is None
        assert emb._model is None

    def test_persist_directory_converts_path_to_str(self) -> None:
        from pathlib import Path

        emb = ResearchEmbeddings(persist_directory=Path("/tmp/chromadb"))
        assert isinstance(emb.persist_directory, str)
        assert emb.persist_directory == "/tmp/chromadb"

    def test_class_constants(self) -> None:
        assert ResearchEmbeddings.DEFAULT_MODEL == "nomic-ai/nomic-embed-text-v1.5"
        assert ResearchEmbeddings.DEFAULT_CONTENT_DEDUP_THRESHOLD == 0.85
        assert ResearchEmbeddings.DEFAULT_EXACT_DEDUP_THRESHOLD == 0.95


# ---------------------------------------------------------------------------
# _get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    """ChromaDB client initialization."""

    def test_creates_persistent_client(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(persist_directory=tmp_path / "chromadb")
        client = emb._get_client()
        assert client is not None
        assert emb._client is client

    def test_returns_cached_client(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(persist_directory=tmp_path / "chromadb")
        client1 = emb._get_client()
        client2 = emb._get_client()
        assert client1 is client2

    def test_raises_if_chromadb_not_installed(self) -> None:
        import sys

        emb = ResearchEmbeddings()
        # Temporarily remove chromadb from sys.modules
        saved = sys.modules.get("chromadb")
        sys.modules["chromadb"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(EmbeddingError, match="chromadb is required"):
                emb._get_client()
        finally:
            if saved is not None:
                sys.modules["chromadb"] = saved
            else:
                sys.modules.pop("chromadb", None)


# ---------------------------------------------------------------------------
# _get_collection
# ---------------------------------------------------------------------------


class TestGetCollection:
    """ChromaDB collection initialization."""

    def test_creates_collection_with_cosine(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(
            persist_directory=tmp_path / "chromadb",
            collection_name="test_col",
        )
        collection = emb._get_collection()
        assert collection is not None
        assert emb._collection is collection

    def test_returns_cached_collection(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(persist_directory=tmp_path / "chromadb")
        col1 = emb._get_collection()
        col2 = emb._get_collection()
        assert col1 is col2


# ---------------------------------------------------------------------------
# _get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    """Sentence-transformers model initialization."""

    def test_raises_if_sentence_transformers_not_installed(self) -> None:
        import sys

        emb = ResearchEmbeddings()
        saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(EmbeddingError, match="sentence-transformers is required"):
                emb._get_model()
        finally:
            if saved is not None:
                sys.modules["sentence_transformers"] = saved
            else:
                sys.modules.pop("sentence_transformers", None)

    def test_returns_cached_model(self) -> None:
        emb = ResearchEmbeddings()
        mock_model = MagicMock()
        emb._model = mock_model
        result = emb._get_model()
        assert result is mock_model


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbed:
    """Embedding generation."""

    def test_empty_list_returns_empty(self) -> None:
        emb = ResearchEmbeddings()
        result = emb.embed([])
        assert result == []

    def test_embeds_texts_to_float_lists(self) -> None:
        emb = ResearchEmbeddings()
        mock_model = MagicMock()
        # Simulate model.encode returning numpy arrays
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        emb._model = mock_model

        result = emb.embed(["hello", "world"])
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)
        mock_model.encode.assert_called_once_with(
            ["hello", "world"], normalize_embeddings=True
        )

    def test_raises_embedding_error_on_failure(self) -> None:
        emb = ResearchEmbeddings()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        emb._model = mock_model

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            emb.embed(["text"])


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------


class TestAddDocuments:
    """Document addition with deduplication."""

    def test_empty_documents_returns_zero(self) -> None:
        emb = ResearchEmbeddings()
        assert emb.add_documents([]) == 0

    def test_adds_non_duplicate_documents(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        # Empty collection, so check_duplicate returns no duplicate
        mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
        }

        doc = EmbeddingDocument(id="doc-1", content="hello", metadata={"src": "web"})
        added = emb.add_documents([doc])

        assert added == 1
        mock_collection.add.assert_called_once()

    def test_skips_duplicate_documents(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        # High similarity => duplicate
        mock_collection.query.return_value = {
            "ids": [["existing-doc"]],
            "distances": [[0.05]],  # distance 0.05 => similarity 0.95
        }

        doc = EmbeddingDocument(id="doc-2", content="duplicate")
        added = emb.add_documents([doc])

        assert added == 0
        mock_collection.add.assert_not_called()

    def test_adds_document_without_metadata(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
        }

        doc = EmbeddingDocument(id="doc-1", content="hello")
        emb.add_documents([doc])

        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["metadatas"] is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    """Similarity search."""

    def test_empty_collection_returns_empty(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        emb._collection = mock_collection

        results = emb.search("query")
        assert results == []

    def test_returns_results_sorted_by_score(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [["doc-1", "doc-2"]],
            "documents": [["content 1", "content 2"]],
            "distances": [[0.3, 0.1]],  # doc-2 is closer
            "metadatas": [[{"src": "a"}, {"src": "b"}]],
        }

        results = emb.search("query", n_results=2)
        assert len(results) == 2
        assert results[0].id == "doc-2"  # Higher similarity (1 - 0.1 = 0.9)
        assert results[1].id == "doc-1"  # Lower similarity (1 - 0.3 = 0.7)
        assert results[0].score > results[1].score

    def test_passes_where_filter(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [["doc-1"]],
            "documents": [["content"]],
            "distances": [[0.2]],
            "metadatas": [[{"src": "web"}]],
        }

        emb.search("query", where={"src": "web"})
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"src": "web"}

    def test_clamps_n_results_to_collection_size(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [["doc-1", "doc-2"]],
            "documents": [["c1", "c2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{}, {}]],
        }

        emb.search("query", n_results=100)
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 2

    def test_handles_empty_query_results(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        results = emb.search("query")
        assert results == []


# ---------------------------------------------------------------------------
# check_duplicate
# ---------------------------------------------------------------------------


class TestCheckDuplicate:
    """Deduplication checking."""

    def test_empty_collection_returns_no_duplicate(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        emb._collection = mock_collection

        result = emb.check_duplicate("content")
        assert result.is_duplicate is False
        assert result.most_similar_id is None

    def test_detects_duplicate_above_threshold(self) -> None:
        emb = ResearchEmbeddings(content_dedup_threshold=0.85)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [["existing-doc"]],
            "distances": [[0.1]],  # similarity = 0.9 > 0.85
        }

        result = emb.check_duplicate("similar content")
        assert result.is_duplicate is True
        assert result.most_similar_id == "existing-doc"
        assert result.similarity_score == pytest.approx(0.9)

    def test_not_duplicate_below_threshold(self) -> None:
        emb = ResearchEmbeddings(content_dedup_threshold=0.85)
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [["other-doc"]],
            "distances": [[0.5]],  # similarity = 0.5 < 0.85
        }

        result = emb.check_duplicate("different content")
        assert result.is_duplicate is False
        assert result.most_similar_id == "other-doc"

    def test_handles_empty_query_results(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        emb._collection = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        emb._model = mock_model

        mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
        }

        result = emb.check_duplicate("content")
        assert result.is_duplicate is False


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


class TestDeleteCollection:
    """Collection deletion."""

    def test_deletes_and_clears_cache(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(persist_directory=tmp_path / "chromadb")
        # Pre-populate _collection
        emb._get_collection()
        assert emb._collection is not None

        emb.delete_collection()
        assert emb._collection is None

    def test_handles_delete_failure_gracefully(self) -> None:
        emb = ResearchEmbeddings()
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = RuntimeError("not found")
        emb._client = mock_client

        # Should not raise
        emb.delete_collection()


# ---------------------------------------------------------------------------
# count property
# ---------------------------------------------------------------------------


class TestCount:
    """Document count property."""

    def test_returns_collection_count(self, tmp_path: Any) -> None:
        emb = ResearchEmbeddings(persist_directory=tmp_path / "chromadb")
        assert emb.count == 0

    def test_returns_mock_count(self) -> None:
        emb = ResearchEmbeddings()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        emb._collection = mock_collection
        assert emb.count == 42
