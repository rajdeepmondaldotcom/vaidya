"""Escalating silence handler per PRD Section 3.2."""

from __future__ import annotations

from vaidya.agents.constants import (
    SILENCE_CONNECTION_LOSS,
    SILENCE_END_CALL,
    SILENCE_REASSURE,
    SILENCE_REPHRASE,
)
from vaidya.i18n import get_msg


class SilenceHandler:
    """Escalating silence handling per PRD Section 3.2.

    Thresholds:
      - 0-3s  : natural pause, no action
      - 5s    : reassuring prompt
      - 10s   : repeat question in simpler words
      - 15s   : connection-loss message, offer callback
      - 20s+  : end call, trigger callback
    """

    _THRESHOLDS = (SILENCE_CONNECTION_LOSS, SILENCE_REPHRASE, SILENCE_REASSURE)
    END_CALL_THRESHOLD: float = SILENCE_END_CALL

    def get_silence_response(
        self,
        silence_seconds: float,
        language: str,
    ) -> str | None:
        """Return the appropriate prompt for the given silence duration.

        Returns ``None`` when no action is needed (< 5s) or when the call
        should be ended (>= 20s) -- the caller must check
        ``should_end_call()`` separately for the termination case.
        """
        if silence_seconds >= self.END_CALL_THRESHOLD:
            return None

        for threshold in self._THRESHOLDS:
            if silence_seconds >= threshold:
                return get_msg("orchestrator", f"silence_{threshold}s", language)

        return None

    def should_end_call(self, silence_seconds: float) -> bool:
        """Return True when silence has exceeded the end-call threshold."""
        return silence_seconds >= self.END_CALL_THRESHOLD
