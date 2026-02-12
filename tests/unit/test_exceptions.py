"""Unit tests for research_agent.exceptions - centralized exception hierarchy."""

from __future__ import annotations

from research_agent.exceptions import (
    BudgetExhaustedError,
    CheckpointCorruptionError,
    CheckpointError,
    EmbeddingError,
    ModelRoutingError,
    ResearchAgentError,
    ScrapingError,
)


class TestResearchAgentError:
    """Base exception class tests."""

    def test_inherits_from_exception(self) -> None:
        assert issubclass(ResearchAgentError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        try:
            raise ResearchAgentError("test error")
        except ResearchAgentError as exc:
            assert str(exc) == "test error"

    def test_catches_all_subclasses(self) -> None:
        subclasses = [
            CheckpointError,
            CheckpointCorruptionError,
            BudgetExhaustedError,
            ModelRoutingError,
            EmbeddingError,
            ScrapingError,
        ]
        for cls in subclasses:
            try:
                raise cls("sub error")
            except ResearchAgentError:
                pass  # Expected: caught by base class


class TestCheckpointErrors:
    """Checkpoint exception hierarchy."""

    def test_checkpoint_error_inherits_base(self) -> None:
        assert issubclass(CheckpointError, ResearchAgentError)

    def test_checkpoint_error_is_exception(self) -> None:
        assert issubclass(CheckpointError, Exception)

    def test_corruption_error_inherits_checkpoint_error(self) -> None:
        assert issubclass(CheckpointCorruptionError, CheckpointError)

    def test_corruption_error_inherits_base(self) -> None:
        assert issubclass(CheckpointCorruptionError, ResearchAgentError)

    def test_checkpoint_error_message(self) -> None:
        err = CheckpointError("save failed")
        assert str(err) == "save failed"

    def test_corruption_error_caught_as_checkpoint_error(self) -> None:
        try:
            raise CheckpointCorruptionError("bad checksum")
        except CheckpointError as exc:
            assert "bad checksum" in str(exc)


class TestBudgetExhaustedError:
    """Budget exception tests."""

    def test_inherits_base(self) -> None:
        assert issubclass(BudgetExhaustedError, ResearchAgentError)

    def test_is_exception(self) -> None:
        assert issubclass(BudgetExhaustedError, Exception)

    def test_message(self) -> None:
        err = BudgetExhaustedError("budget exceeded")
        assert str(err) == "budget exceeded"


class TestModelRoutingError:
    """Model routing exception tests."""

    def test_inherits_base(self) -> None:
        assert issubclass(ModelRoutingError, ResearchAgentError)

    def test_is_exception(self) -> None:
        assert issubclass(ModelRoutingError, Exception)

    def test_message(self) -> None:
        err = ModelRoutingError("no models available")
        assert str(err) == "no models available"


class TestEmbeddingError:
    """Embedding exception tests."""

    def test_inherits_base(self) -> None:
        assert issubclass(EmbeddingError, ResearchAgentError)

    def test_is_exception(self) -> None:
        assert issubclass(EmbeddingError, Exception)

    def test_message(self) -> None:
        err = EmbeddingError("model not loaded")
        assert str(err) == "model not loaded"


class TestScrapingError:
    """Scraping exception tests."""

    def test_inherits_base(self) -> None:
        assert issubclass(ScrapingError, ResearchAgentError)

    def test_is_exception(self) -> None:
        assert issubclass(ScrapingError, Exception)

    def test_message(self) -> None:
        err = ScrapingError("connection timeout")
        assert str(err) == "connection timeout"
