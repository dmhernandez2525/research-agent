"""Unit tests for research_agent.embeddings - models, init, and stub verification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.embeddings import (
    DeduplicationResult,
    EmbeddingDocument,
    ResearchEmbeddings,
    SimilarityResult,
)

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
# Stub methods raise NotImplementedError
# ---------------------------------------------------------------------------


class TestStubMethods:
    """All unimplemented methods raise NotImplementedError."""

    @pytest.fixture()
    def embeddings(self) -> ResearchEmbeddings:
        return ResearchEmbeddings()

    def test_get_client_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="_get_client"):
            embeddings._get_client()

    def test_get_collection_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="_get_collection"):
            embeddings._get_collection()

    def test_get_model_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="_get_model"):
            embeddings._get_model()

    def test_embed_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="embed"):
            embeddings.embed(["hello"])

    def test_add_documents_raises(self, embeddings: ResearchEmbeddings) -> None:
        doc = EmbeddingDocument(id="d1", content="hello")
        with pytest.raises(NotImplementedError, match="add_documents"):
            embeddings.add_documents([doc])

    def test_search_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="search"):
            embeddings.search("query")

    def test_check_duplicate_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="check_duplicate"):
            embeddings.check_duplicate("some content")

    def test_delete_collection_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="delete_collection"):
            embeddings.delete_collection()

    def test_count_raises(self, embeddings: ResearchEmbeddings) -> None:
        with pytest.raises(NotImplementedError, match="count"):
            _ = embeddings.count
