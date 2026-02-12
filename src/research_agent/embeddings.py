"""nomic-embed-text + ChromaDB integration for semantic search and dedup.

Provides the ``ResearchEmbeddings`` class for embedding, storing, and
retrieving research content with deduplication support.

Dependencies (chromadb, sentence-transformers) are optional and
lazy-imported. Install with: ``pip install research-agent[local]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from research_agent.exceptions import EmbeddingError

if TYPE_CHECKING:
    from pathlib import Path

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EmbeddingDocument(BaseModel):
    """A document to embed and store in the vector store."""

    id: str = Field(description="Unique document identifier.")
    content: str = Field(description="Text content to embed.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata."
    )


class SimilarityResult(BaseModel):
    """A result from a similarity search."""

    id: str = Field(description="Document identifier.")
    content: str = Field(default="")
    score: float = Field(default=0.0, description="Cosine similarity score (0-1).")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeduplicationResult(BaseModel):
    """Result of a deduplication check."""

    is_duplicate: bool = Field(
        default=False, description="Whether the content is a near-duplicate."
    )
    most_similar_id: str | None = Field(
        default=None, description="ID of the most similar existing document."
    )
    similarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Similarity to the closest match."
    )


# ---------------------------------------------------------------------------
# Embeddings class
# ---------------------------------------------------------------------------


class ResearchEmbeddings:
    """Manages embeddings and vector storage for research content.

    Uses nomic-embed-text via sentence-transformers for embedding and
    ChromaDB for persistent vector storage.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Path to ChromaDB persistence directory.
        model_name: Sentence-transformers model identifier.
        dedup_threshold: Cosine similarity threshold for deduplication.
    """

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    DEFAULT_CONTENT_DEDUP_THRESHOLD = 0.85
    DEFAULT_EXACT_DEDUP_THRESHOLD = 0.95

    def __init__(
        self,
        collection_name: str = "research_docs",
        persist_directory: Path | str = "./data/chromadb",
        model_name: str = DEFAULT_MODEL,
        dimensions: int = 768,
        content_dedup_threshold: float = DEFAULT_CONTENT_DEDUP_THRESHOLD,
        exact_dedup_threshold: float = DEFAULT_EXACT_DEDUP_THRESHOLD,
    ) -> None:
        """Initialize the embeddings manager.

        Args:
            collection_name: ChromaDB collection name.
            persist_directory: Directory for ChromaDB persistence.
            model_name: Sentence-transformers model identifier.
            dimensions: Embedding vector dimensions.
            content_dedup_threshold: Cosine similarity for content-level dedup (0.85).
            exact_dedup_threshold: Cosine similarity for exact-match dedup (0.95).
        """
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.model_name = model_name
        self.dimensions = dimensions
        self.content_dedup_threshold = content_dedup_threshold
        self.exact_dedup_threshold = exact_dedup_threshold

        self._client: Any = None  # chromadb.ClientAPI (lazy-loaded)
        self._collection: Any = None  # chromadb.Collection (lazy-loaded)
        self._model: Any = None  # SentenceTransformer (lazy-loaded)

    def _get_client(self) -> Any:
        """Get or create the ChromaDB persistent client.

        Returns:
            ChromaDB client instance.

        Raises:
            EmbeddingError: If chromadb is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except ImportError as exc:
            raise EmbeddingError(
                "chromadb is required for embeddings. "
                "Install with: pip install research-agent[local]"
            ) from exc

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info("chromadb_client_created", path=self.persist_directory)
        return self._client

    def _get_collection(self) -> Any:
        """Get or create the ChromaDB collection with cosine similarity.

        Returns:
            ChromaDB collection instance.
        """
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chromadb_collection_ready",
            name=self.collection_name,
            count=self._collection.count(),
        )
        return self._collection

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers embedding model.

        Returns:
            A SentenceTransformer model instance.

        Raises:
            EmbeddingError: If sentence-transformers is not installed.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install research-agent[local]"
            ) from exc

        self._model = SentenceTransformer(
            self.model_name, trust_remote_code=True
        )
        logger.info("embedding_model_loaded", model=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            EmbeddingError: If the embedding model fails.
        """
        if not texts:
            return []

        model = self._get_model()
        try:
            embeddings = model.encode(texts, normalize_embeddings=True)
            return [vec.tolist() for vec in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    def add_documents(self, documents: list[EmbeddingDocument]) -> int:
        """Add documents to the vector store with deduplication.

        Checks each document against existing content before adding.
        Documents with similarity above ``content_dedup_threshold``
        are skipped as duplicates.

        Args:
            documents: Documents to add.

        Returns:
            Number of documents actually added (excluding duplicates).
        """
        if not documents:
            return 0

        collection = self._get_collection()
        added = 0

        for doc in documents:
            # Check for duplicates before adding
            dedup = self.check_duplicate(doc.content)
            if dedup.is_duplicate:
                logger.debug(
                    "document_skipped_duplicate",
                    doc_id=doc.id,
                    similar_to=dedup.most_similar_id,
                    score=dedup.similarity_score,
                )
                continue

            # Embed and add
            vectors = self.embed([doc.content])
            collection.add(
                ids=[doc.id],
                embeddings=vectors,
                documents=[doc.content],
                metadatas=[doc.metadata] if doc.metadata else None,
            )
            added += 1
            logger.debug("document_added", doc_id=doc.id)

        logger.info(
            "add_documents_complete",
            total=len(documents),
            added=added,
            skipped=len(documents) - added,
        )
        return added

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SimilarityResult]:
        """Perform similarity search against stored documents.

        Args:
            query: Query text.
            n_results: Maximum number of results.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of similarity results, ordered by descending similarity.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        # Clamp n_results to available documents
        actual_n = min(n_results, collection.count())

        query_embedding = self.embed([query])

        kwargs: dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = collection.query(**kwargs)

        results: list[SimilarityResult] = []
        if raw["ids"] and raw["ids"][0]:
            ids = raw["ids"][0]
            docs = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
            distances = raw["distances"][0] if raw.get("distances") else [1.0] * len(ids)
            metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)

            for i, doc_id in enumerate(ids):
                # ChromaDB cosine distance = 1 - similarity
                similarity = 1.0 - distances[i]
                results.append(
                    SimilarityResult(
                        id=doc_id,
                        content=docs[i] or "",
                        score=max(0.0, min(1.0, similarity)),
                        metadata=metadatas[i] or {},
                    )
                )

        # Sort by descending score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def check_duplicate(self, content: str) -> DeduplicationResult:
        """Check whether content is a near-duplicate of an existing document.

        Uses the content dedup threshold (0.85) for content-level
        similarity and the exact dedup threshold (0.95) for near-exact
        matches.

        Args:
            content: Text content to check.

        Returns:
            A ``DeduplicationResult`` indicating whether the content is
            a duplicate and the closest match.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            return DeduplicationResult()

        query_embedding = self.embed([content])
        raw = collection.query(
            query_embeddings=query_embedding,
            n_results=1,
            include=["distances"],
        )

        if not raw["ids"] or not raw["ids"][0]:
            return DeduplicationResult()

        distance = raw["distances"][0][0]
        similarity = 1.0 - distance
        similarity = max(0.0, min(1.0, similarity))
        most_similar_id = raw["ids"][0][0]

        is_duplicate = similarity >= self.content_dedup_threshold

        return DeduplicationResult(
            is_duplicate=is_duplicate,
            most_similar_id=most_similar_id,
            similarity_score=similarity,
        )

    def delete_collection(self) -> None:
        """Delete the entire ChromaDB collection."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info("collection_deleted", name=self.collection_name)
        except Exception as exc:
            logger.warning(
                "collection_delete_failed",
                name=self.collection_name,
                error=str(exc),
            )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection.

        Returns:
            Document count, or 0 if the collection does not exist.
        """
        collection = self._get_collection()
        result: int = collection.count()
        return result
