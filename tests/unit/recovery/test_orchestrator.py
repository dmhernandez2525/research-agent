"""Unit tests for recovery orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.recovery import RecoveryOrchestrator, RetryPolicy


@pytest.mark.asyncio
async def test_retry_exhaustion_creates_dead_letter() -> None:
    attempts = {"count": 0}

    async def always_fail(_: dict[str, Any]) -> dict[str, Any]:
        attempts["count"] += 1
        raise RuntimeError("boom")

    orchestrator = RecoveryOrchestrator(
        default_policy=RetryPolicy(
            attempts=2,
            backoff_initial_seconds=0.001,
            backoff_max_seconds=0.001,
        ),
    )
    wrapped = orchestrator.wrap("search", always_fail)

    result = await wrapped({"report_metadata": {"session": "x"}})

    assert attempts["count"] == 2
    assert len(result["error_log"]) == 1
    assert result["error_log"][0]["recoverable"] is False
    metrics = result["report_metadata"]["recovery_metrics"]
    assert metrics["retry_exhausted"] == 1
    assert metrics["dead_letter_count"] == 1
    assert len(result["report_metadata"]["dead_letter_queue"]) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_skips_node_when_open() -> None:
    attempts = {"count": 0}

    async def always_fail(_: dict[str, Any]) -> dict[str, Any]:
        attempts["count"] += 1
        raise RuntimeError("unhealthy")

    orchestrator = RecoveryOrchestrator(
        default_policy=RetryPolicy(attempts=1, backoff_initial_seconds=0.001),
        circuit_breaker_threshold=2,
        circuit_breaker_cooldown_seconds=300,
    )
    wrapped = orchestrator.wrap("scrape", always_fail)

    await wrapped({})
    await wrapped({})
    skipped = await wrapped({})

    assert attempts["count"] == 2
    assert "circuit breaker is open" in skipped["error_log"][0]["message"]
    metrics = skipped["report_metadata"]["recovery_metrics"]
    assert metrics["circuit_breaker_opened"] == 1
    assert metrics["circuit_breaker_skips"] == 1


@pytest.mark.asyncio
async def test_retries_can_recover_without_dead_letter() -> None:
    attempts = {"count": 0}

    async def flaky(_: dict[str, Any]) -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("transient")
        return {"step": "plan"}

    orchestrator = RecoveryOrchestrator(
        default_policy=RetryPolicy(
            attempts=3,
            backoff_initial_seconds=0.001,
            backoff_max_seconds=0.001,
        )
    )
    wrapped = orchestrator.wrap("plan", flaky)

    result = await wrapped({"report_metadata": {"existing": True}})

    assert attempts["count"] == 2
    assert result["step"] == "plan"
    assert result["report_metadata"]["existing"] is True
    metrics = result["report_metadata"]["recovery_metrics"]
    assert metrics["retries_attempted"] == 1
    assert metrics["recovered_failures"] == 1
    assert metrics["dead_letter_count"] == 0
