"""Resilience patterns for Sarvam API calls: circuit breaker and rate tracking.

The retry logic is already in client.py via ``_retry_async()``.  This module
adds a per-service circuit breaker so that a cascade of failures in one API
(e.g. TTS) doesn't block calls to other APIs (e.g. LLM).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-service circuit breaker.

    Tracks consecutive failures for a named service.  When failures exceed
    *failure_threshold*, the circuit opens and all subsequent calls fail-fast
    with ``CircuitOpenError`` until *recovery_timeout* seconds have elapsed.
    After that the circuit enters half-open state and allows up to
    *half_open_max_calls* probe calls.  If any succeeds the circuit closes;
    if one fails it re-opens.

    Usage::

        cb = CircuitBreaker(name="tts")
        cb.check()          # raises CircuitOpenError if open
        try:
            result = await do_tts_call()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 2

    state: CircuitState = CircuitState.CLOSED
    _consecutive_failures: int = 0
    _last_failure_time: float = 0.0
    _half_open_calls: int = 0

    def check(self) -> None:
        """Raise :class:`CircuitOpenError` if the circuit is open."""
        if self.state == CircuitState.CLOSED:
            return
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit %s: OPEN -> HALF_OPEN (recovery probe)", self.name)
            else:
                raise CircuitOpenError(self.name)
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError(self.name)
            self._half_open_calls += 1

    def record_success(self) -> None:
        """Record a successful API call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit %s: HALF_OPEN -> CLOSED (probe succeeded)", self.name)
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed API call."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit %s: HALF_OPEN -> OPEN (probe failed)", self.name)
        elif self._consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "Circuit %s: CLOSED -> OPEN (%d consecutive failures)",
                self.name,
                self._consecutive_failures,
            )


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and calls should be skipped."""

    def __init__(self, service: str) -> None:
        self.service = service
        super().__init__(f"Circuit breaker open for service: {service}")


@dataclass
class ServiceCircuitBreakers:
    """Collection of per-service circuit breakers for all Sarvam APIs."""

    llm: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="llm"))
    stt: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="stt"))
    tts: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="tts"))
    translate: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="translate"))
    transliterate: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(name="transliterate")
    )
    language_id: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="language_id"))
    vision: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="vision"))

    def get(self, service: str) -> CircuitBreaker:
        """Return the circuit breaker for *service*, or a no-op default."""
        return getattr(self, service, CircuitBreaker(name=service))

    def status(self) -> dict[str, str]:
        """Return a dict of service -> circuit state for monitoring."""
        services = (
            "llm",
            "stt",
            "tts",
            "translate",
            "transliterate",
            "language_id",
            "vision",
        )
        return {name: getattr(self, name).state.value for name in services}
