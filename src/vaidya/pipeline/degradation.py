"""Graceful degradation framework for Vaidya services.

Tracks consecutive failures per service and computes the current
degradation level.  The orchestrator checks the level before dispatching
agents so it can proactively skip unavailable services rather than
waiting for timeouts.
"""

from __future__ import annotations

import logging
from enum import IntEnum

logger = logging.getLogger(__name__)


class DegradationLevel(IntEnum):
    """Ordered severity levels -- higher means more features unavailable."""

    FULL = 0  # All services operational
    NO_REVIEWER = 1  # Reviewer unavailable, eligibility-only with caveats
    REDUCED_LANGUAGES = 2  # Translation down, Hindi+English only
    SCRIPTED = 3  # LLM unavailable, pre-scripted responses
    SMS_ONLY = 4  # Voice pipeline down, redirect to SMS


# Map service names to the degradation level that triggers when they fail
_SERVICE_LEVELS: dict[str, DegradationLevel] = {
    "reviewer": DegradationLevel.NO_REVIEWER,
    "translator": DegradationLevel.REDUCED_LANGUAGES,
    "llm": DegradationLevel.SCRIPTED,
    "voice": DegradationLevel.SMS_ONLY,
}


class DegradationManager:
    """Tracks service failures and computes the current degradation level.

    Usage::

        dm = DegradationManager()
        dm.record_failure("reviewer")
        dm.record_failure("reviewer")
        dm.record_failure("reviewer")  # 3 consecutive → degrades
        assert dm.level >= DegradationLevel.NO_REVIEWER

        dm.record_success("reviewer")  # counter reset
        assert dm.level == DegradationLevel.FULL
    """

    def __init__(self, threshold: int = 3) -> None:
        self._failures: dict[str, int] = {}
        self._threshold = threshold

    def record_failure(self, service: str) -> None:
        """Increment the consecutive failure count for *service*."""
        self._failures[service] = self._failures.get(service, 0) + 1
        count = self._failures[service]
        if count == self._threshold:
            logger.warning(
                "Service degraded after %d consecutive failures",
                count,
                extra={"service": service},
            )

    def record_success(self, service: str) -> None:
        """Reset the failure counter for *service* (it recovered)."""
        if self._failures.get(service, 0) > 0:
            logger.info(
                "Service recovered",
                extra={"service": service, "prior_failures": self._failures[service]},
            )
            self._failures[service] = 0

    @property
    def level(self) -> DegradationLevel:
        """Compute the current degradation level from all tracked services.

        Returns the *highest* (most severe) level among services that have
        exceeded the failure threshold.
        """
        max_level = DegradationLevel.FULL
        for service, count in self._failures.items():
            if count >= self._threshold:
                svc_level = _SERVICE_LEVELS.get(service, DegradationLevel.FULL)
                if svc_level > max_level:
                    max_level = svc_level
        return max_level

    def is_service_available(self, service: str) -> bool:
        """Return ``True`` if *service* has not exceeded the failure threshold."""
        return self._failures.get(service, 0) < self._threshold

    @property
    def failed_services(self) -> list[str]:
        """List of services currently exceeding the failure threshold."""
        return [s for s, c in self._failures.items() if c >= self._threshold]
