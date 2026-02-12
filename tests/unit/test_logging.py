"""Unit tests for research_agent.logging - structured logging and provenance."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
import structlog

from research_agent.logging import (
    _VALID_LEVELS,
    configure_logging,
    generate_session_id,
    log_provenance,
    step_logging_context,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog state between tests."""
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    # Clear stdlib root logger handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# generate_session_id
# ---------------------------------------------------------------------------


class TestGenerateSessionId:
    """generate_session_id produces unique UUID4 strings."""

    def test_returns_string(self) -> None:
        sid = generate_session_id()
        assert isinstance(sid, str)

    def test_uuid_format(self) -> None:
        sid = generate_session_id()
        # UUID4 has 36 chars (32 hex + 4 hyphens)
        assert len(sid) == 36
        assert sid.count("-") == 4

    def test_unique_across_calls(self) -> None:
        ids = {generate_session_id() for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# configure_logging - level validation
# ---------------------------------------------------------------------------


class TestConfigureLoggingLevel:
    """configure_logging validates and applies log levels."""

    def test_valid_level_info(self) -> None:
        configure_logging(level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_valid_level_debug(self) -> None:
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_case_insensitive(self) -> None:
        configure_logging(level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            configure_logging(level="TRACE")

    def test_all_valid_levels_accepted(self) -> None:
        for lvl in _VALID_LEVELS:
            configure_logging(level=lvl)
            root = logging.getLogger()
            expected = getattr(logging, lvl)
            assert root.level == expected


# ---------------------------------------------------------------------------
# configure_logging - format
# ---------------------------------------------------------------------------


class TestConfigureLoggingFormat:
    """configure_logging selects the right renderer."""

    def test_console_format(self) -> None:
        configure_logging(fmt="console")
        # Should not raise; structlog is configured
        log = structlog.get_logger("test")
        assert log is not None

    def test_json_format(self) -> None:
        configure_logging(fmt="json")
        log = structlog.get_logger("test")
        assert log is not None


# ---------------------------------------------------------------------------
# configure_logging - session ID binding
# ---------------------------------------------------------------------------


class TestConfigureLoggingSessionId:
    """configure_logging binds session_id to context vars."""

    def test_session_id_bound(self) -> None:
        configure_logging(session_id="test-session-123")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("session_id") == "test-session-123"

    def test_no_session_id(self) -> None:
        configure_logging()
        ctx = structlog.contextvars.get_contextvars()
        assert "session_id" not in ctx


# ---------------------------------------------------------------------------
# configure_logging - file output
# ---------------------------------------------------------------------------


class TestConfigureLoggingFile:
    """configure_logging creates file handlers."""

    def test_file_handler_created(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        configure_logging(log_file=str(log_file))
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_file_receives_output(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log = structlog.get_logger("test_file")
        log.info("test_message", key="value")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "test_message" in content

    def test_no_file_handler_without_param(self) -> None:
        configure_logging()
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_stderr_handler_always_present(self) -> None:
        configure_logging()
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_reconfigure_clears_handlers(self, tmp_path: Path) -> None:
        log_file1 = tmp_path / "first.log"
        log_file2 = tmp_path / "second.log"
        configure_logging(log_file=str(log_file1))
        configure_logging(log_file=str(log_file2))
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        # Should only have the second file handler (first cleared)
        assert len(file_handlers) == 1

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        log_file = tmp_path / "path_obj.log"
        configure_logging(log_file=log_file)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1


# ---------------------------------------------------------------------------
# step_logging_context
# ---------------------------------------------------------------------------


class TestStepLoggingContext:
    """step_logging_context binds and unbinds step metadata."""

    def test_context_binds_step_name(self) -> None:
        configure_logging(level="DEBUG")
        with step_logging_context("plan", step_index=0) as log:
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("step_name") == "plan"
            assert ctx.get("step_index") == 0
            assert log is not None

    def test_context_unbinds_on_exit(self) -> None:
        configure_logging(level="DEBUG")
        with step_logging_context("plan", step_index=0):
            pass
        ctx = structlog.contextvars.get_contextvars()
        assert "step_name" not in ctx
        assert "step_index" not in ctx

    def test_extra_kwargs_bound(self) -> None:
        configure_logging(level="DEBUG")
        with step_logging_context("search", step_index=1, query="test"):
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("query") == "test"

    def test_extra_kwargs_unbound(self) -> None:
        configure_logging(level="DEBUG")
        with step_logging_context("search", step_index=1, query="test"):
            pass
        ctx = structlog.contextvars.get_contextvars()
        assert "query" not in ctx

    def test_exception_propagated(self) -> None:
        configure_logging(level="DEBUG")
        with pytest.raises(RuntimeError, match="test error"), step_logging_context("failing_step"):
            raise RuntimeError("test error")

    def test_context_cleaned_after_exception(self) -> None:
        configure_logging(level="DEBUG")
        with pytest.raises(RuntimeError), step_logging_context("failing_step", step_index=5):
            raise RuntimeError("boom")
        ctx = structlog.contextvars.get_contextvars()
        assert "step_name" not in ctx

    def test_yields_logger(self) -> None:
        configure_logging(level="DEBUG")
        with step_logging_context("test_step") as log:
            assert hasattr(log, "info")
            assert hasattr(log, "debug")
            assert hasattr(log, "warning")


# ---------------------------------------------------------------------------
# log_provenance
# ---------------------------------------------------------------------------


class TestLogProvenance:
    """log_provenance emits structured provenance entries."""

    def test_logs_provenance_entry(self, tmp_path: Path) -> None:
        log_file = tmp_path / "provenance.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log_provenance(
            source_url="https://example.com/article",
            action="scraped",
            step_name="scrape",
            details={"status_code": 200},
        )

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "provenance_entry" in content
        assert "https://example.com/article" in content

    def test_logs_without_details(self, tmp_path: Path) -> None:
        log_file = tmp_path / "provenance.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log_provenance(
            source_url="https://example.com",
            action="cited",
        )

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "cited" in content

    def test_logs_with_session_context(self, tmp_path: Path) -> None:
        log_file = tmp_path / "provenance.log"
        configure_logging(
            level="INFO", fmt="json",
            log_file=str(log_file),
            session_id="sess-abc",
        )

        log_provenance(
            source_url="https://example.com",
            action="embedded",
            step_name="embed",
        )

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        assert "sess-abc" in content


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """JSON format produces parseable log lines."""

    def test_json_parseable(self, tmp_path: Path) -> None:
        log_file = tmp_path / "json.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log = structlog.get_logger("json_test")
        log.info("test_event", data="hello")

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text().strip()
        # Should be valid JSON
        parsed = json.loads(content)
        assert parsed["event"] == "test_event"
        assert parsed["data"] == "hello"

    def test_json_includes_timestamp(self, tmp_path: Path) -> None:
        log_file = tmp_path / "json.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log = structlog.get_logger("ts_test")
        log.info("ts_event")

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text().strip()
        parsed = json.loads(content)
        assert "timestamp" in parsed

    def test_json_includes_log_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "json.log"
        configure_logging(level="INFO", fmt="json", log_file=str(log_file))

        log = structlog.get_logger("level_test")
        log.warning("warn_event")

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text().strip()
        parsed = json.loads(content)
        assert parsed["level"] == "warning"


# ---------------------------------------------------------------------------
# Integration: session binding + step context
# ---------------------------------------------------------------------------


class TestSessionStepIntegration:
    """Session ID persists through step contexts."""

    def test_session_id_in_step_context(self, tmp_path: Path) -> None:
        log_file = tmp_path / "integration.log"
        configure_logging(
            level="DEBUG", fmt="json",
            log_file=str(log_file),
            session_id="integration-sess",
        )

        with step_logging_context("plan", step_index=0):
            log = structlog.get_logger("integration")
            log.info("inside_step")

        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text()
        # The session ID should be present in the step context logs
        assert "integration-sess" in content
        assert "inside_step" in content
