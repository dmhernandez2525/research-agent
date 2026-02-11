"""structlog configuration and provenance chain logging.

Provides session ID generation, step-level logging context managers,
and structured log configuration for console and JSON output.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------


def generate_session_id() -> str:
    """Generate a unique session identifier.

    Returns:
        A UUID4 string for the current research session.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# structlog configuration
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    fmt: str = "console",
    log_file: str | None = None,
    session_id: str | None = None,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Output format -- ``"console"`` for human-readable or
            ``"json"`` for machine-parseable.
        log_file: Optional file path for log output (in addition to stderr).
        session_id: Optional session ID to bind to all log entries.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if session_id:
        shared_processors.insert(
            0,
            structlog.processors.EventRenamer("event"),
        )

    renderer: structlog.types.Processor
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if session_id:
        structlog.contextvars.bind_contextvars(session_id=session_id)


# ---------------------------------------------------------------------------
# Step logging context manager
# ---------------------------------------------------------------------------


@contextmanager
def step_logging_context(
    step_name: str,
    step_index: int = 0,
    **extra: Any,
) -> Iterator[structlog.stdlib.BoundLogger]:
    """Context manager that binds step-level metadata to structlog.

    Automatically logs step start and completion, and binds the step
    name and index to all log entries within the context.

    Args:
        step_name: Name of the graph node / step.
        step_index: Ordinal index of the step in the pipeline.
        **extra: Additional key-value pairs to bind.

    Yields:
        A bound structlog logger with step context.

    Example::

        with step_logging_context("plan", step_index=0) as log:
            log.info("decomposing_query", query=query)
    """
    structlog.contextvars.bind_contextvars(
        step_name=step_name,
        step_index=step_index,
        **extra,
    )

    log: structlog.stdlib.BoundLogger = structlog.get_logger(step_name)
    log.info("step_start", step_name=step_name, step_index=step_index)

    try:
        yield log
    except Exception:
        log.exception("step_error", step_name=step_name, step_index=step_index)
        raise
    finally:
        log.info("step_end", step_name=step_name, step_index=step_index)
        structlog.contextvars.unbind_contextvars(
            "step_name", "step_index", *extra.keys()
        )


# ---------------------------------------------------------------------------
# Provenance logger
# ---------------------------------------------------------------------------


def log_provenance(
    source_url: str,
    action: str,
    step_name: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    """Log a provenance chain entry for audit / reproducibility.

    Args:
        source_url: The URL or resource identifier.
        action: What was done (e.g. "scraped", "embedded", "cited").
        step_name: The graph node that performed the action.
        details: Optional additional details.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger("provenance")
    logger.info(
        "provenance_entry",
        source_url=source_url,
        action=action,
        step_name=step_name,
        **(details or {}),
    )
