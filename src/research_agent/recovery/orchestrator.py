"""Retry/circuit-breaker orchestration for graph node execution."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog

from research_agent.recovery.models import DeadLetterEntry, RecoveryMetrics, RetryPolicy
from research_agent.state import ResearchState

if TYPE_CHECKING:
    from research_agent.config import RecoverySettings

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

NodeCallable = Callable[[ResearchState], Awaitable[dict[str, Any]]]


class RecoveryOrchestrator:
    """Centralized node recovery with retry, circuit breaker, and DLQ."""

    def __init__(
        self,
        default_policy: RetryPolicy,
        node_policies: dict[str, RetryPolicy] | None = None,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_seconds: int = 120,
        dead_letter_max_entries: int = 200,
    ) -> None:
        self._default_policy = default_policy
        self._node_policies = node_policies or {}
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_cooldown_seconds = circuit_breaker_cooldown_seconds
        self._dead_letters: deque[DeadLetterEntry] = deque(
            maxlen=dead_letter_max_entries
        )
        self._failure_counts: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}
        self.metrics = RecoveryMetrics()

    @classmethod
    def from_settings(cls, settings: RecoverySettings) -> RecoveryOrchestrator:
        """Build orchestrator from config settings."""
        node_policy_settings = (
            settings.node_policies if isinstance(settings.node_policies, dict) else {}
        )
        policies = {
            node: RetryPolicy(
                attempts=cls._coerce_int(policy.attempts, default=3),
                backoff_initial_seconds=cls._coerce_float(
                    policy.backoff_initial_seconds, default=0.5
                ),
                backoff_max_seconds=cls._coerce_float(
                    policy.backoff_max_seconds, default=8.0
                ),
            )
            for node, policy in node_policy_settings.items()
        }
        return cls(
            default_policy=RetryPolicy(
                attempts=cls._coerce_int(settings.default_policy.attempts, default=3),
                backoff_initial_seconds=cls._coerce_float(
                    settings.default_policy.backoff_initial_seconds,
                    default=0.5,
                ),
                backoff_max_seconds=cls._coerce_float(
                    settings.default_policy.backoff_max_seconds,
                    default=8.0,
                ),
            ),
            node_policies=policies,
            circuit_breaker_threshold=cls._coerce_int(
                settings.circuit_breaker_threshold,
                default=3,
            ),
            circuit_breaker_cooldown_seconds=cls._coerce_int(
                settings.circuit_breaker_cooldown_seconds,
                default=120,
            ),
            dead_letter_max_entries=cls._coerce_int(
                settings.dead_letter_max_entries,
                default=200,
            ),
        )

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if isinstance(value, int):
            return value
        return default

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        return default

    def wrap(self, node_name: str, node_fn: NodeCallable) -> NodeCallable:
        """Wrap a node with retry, circuit breaker, and DLQ handling."""

        async def _wrapped(state: ResearchState) -> dict[str, Any]:
            if self._is_circuit_open(node_name):
                self.metrics.circuit_breaker_skips += 1
                message = f"Node '{node_name}' skipped because circuit breaker is open."
                self._enqueue_dead_letter(
                    node_name=node_name,
                    exc=RuntimeError(message),
                    attempts=1,
                    reason="circuit_open",
                )
                return self._augment_result(
                    state,
                    {},
                    error_entries=[
                        {
                            "step": node_name,
                            "message": message,
                            "recoverable": False,
                        }
                    ],
                )

            policy = self._policy_for(node_name)
            attempt = 0
            last_exc: Exception | None = None

            while attempt < policy.attempts:
                attempt += 1
                try:
                    result = await node_fn(state)
                    self._failure_counts[node_name] = 0
                    self._circuit_open_until.pop(node_name, None)
                    if attempt > 1:
                        self.metrics.recovered_failures += 1
                    return self._augment_result(state, result)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "node_execution_failed",
                        node=node_name,
                        attempt=attempt,
                        max_attempts=policy.attempts,
                        error=str(exc),
                    )
                    if not self._should_retry(exc) or attempt >= policy.attempts:
                        break
                    self.metrics.retries_attempted += 1
                    await asyncio.sleep(self._backoff_seconds(policy, attempt))

            self.metrics.retry_exhausted += 1
            self._failure_counts[node_name] = self._failure_counts.get(node_name, 0) + 1

            if self._failure_counts[node_name] >= self._circuit_breaker_threshold:
                self.metrics.circuit_breaker_opened += 1
                self._circuit_open_until[node_name] = (
                    time.monotonic() + self._circuit_breaker_cooldown_seconds
                )

            final_exc = last_exc or RuntimeError("Unknown node failure")
            self._enqueue_dead_letter(
                node_name=node_name, exc=final_exc, attempts=attempt
            )
            return self._augment_result(
                state,
                {},
                error_entries=[
                    {
                        "step": node_name,
                        "message": str(final_exc),
                        "recoverable": False,
                    }
                ],
            )

        return _wrapped

    def _policy_for(self, node_name: str) -> RetryPolicy:
        return self._node_policies.get(node_name, self._default_policy)

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        return not isinstance(
            exc, (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
        )

    @staticmethod
    def _backoff_seconds(policy: RetryPolicy, attempt: int) -> float:
        initial: float = policy.backoff_initial_seconds
        max_backoff: float = policy.backoff_max_seconds
        raw: float = initial * float(2 ** max(attempt - 1, 0))
        if raw < max_backoff:
            return raw
        return max_backoff

    def _is_circuit_open(self, node_name: str) -> bool:
        open_until = self._circuit_open_until.get(node_name)
        if open_until is None:
            return False
        if time.monotonic() >= open_until:
            self._circuit_open_until.pop(node_name, None)
            self._failure_counts[node_name] = 0
            return False
        return True

    def _enqueue_dead_letter(
        self,
        node_name: str,
        exc: Exception,
        attempts: int,
        reason: str = "retry_exhausted",
    ) -> None:
        entry = DeadLetterEntry(
            node=node_name,
            error_type=exc.__class__.__name__,
            message=str(exc),
            attempts=max(attempts, 1),
            reason=reason,
        )
        self._dead_letters.append(entry)
        self.metrics.dead_letter_count = len(self._dead_letters)
        logger.error(
            "node_dead_lettered",
            node=node_name,
            error_type=entry.error_type,
            reason=reason,
            attempts=attempts,
        )

    def _augment_result(
        self,
        state: ResearchState,
        result: dict[str, Any],
        error_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        augmented = dict(result)

        state_meta = state.get("report_metadata", {})
        merged_meta = dict(state_meta) if isinstance(state_meta, dict) else {}

        result_meta = augmented.get("report_metadata")
        if isinstance(result_meta, dict):
            merged_meta.update(result_meta)

        merged_meta["recovery_metrics"] = self.metrics.model_dump()
        merged_meta["dead_letter_queue"] = [
            item.model_dump() for item in self._dead_letters
        ]
        augmented["report_metadata"] = merged_meta

        if error_entries:
            existing = augmented.get("error_log", [])
            existing_list = list(existing) if isinstance(existing, list) else []
            existing_list.extend(error_entries)
            augmented["error_log"] = existing_list

        return augmented
