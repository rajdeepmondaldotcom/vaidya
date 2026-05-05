"""Escalating silence handler per PRD Section 3.2."""

from __future__ import annotations

from vaidya.agents.constants import (
    SILENCE_CONNECTION_LOSS,
    SILENCE_END_CALL,
    SILENCE_REASSURE,
    SILENCE_REPHRASE,
    SILENCE_STEPS,
)
from vaidya.i18n import get_msg


class SilenceHandler:
    """Escalating silence handling per PRD Section 3.2.

    Two tables of thresholds:

    Simulation / text channel (legacy, 5/10/15/20):
      - 0-3s  : natural pause, no action
      - 5s    : reassuring prompt
      - 10s   : repeat question in simpler words
      - 15s   : connection-loss message, offer callback
      - 20s+  : end call, trigger callback

    Voice channel (real phone calls, 6/12/20):
      - 0-5s  : natural pause
      - 6s    : gentle nudge
      - 12s   : reprompt (prefix + last question)
      - 20s+  : terminal closure, hang up
    """

    _THRESHOLDS = (SILENCE_CONNECTION_LOSS, SILENCE_REPHRASE, SILENCE_REASSURE)
    END_CALL_THRESHOLD: float = SILENCE_END_CALL

    def get_silence_response(
        self,
        silence_seconds: float,
        language: str,
    ) -> str | None:
        """Legacy text/simulation escalation. Returns None below 5s or at/after
        the end-call threshold (the caller must check ``should_end_call``)."""
        if silence_seconds >= self.END_CALL_THRESHOLD:
            return None

        for threshold in self._THRESHOLDS:
            if silence_seconds >= threshold:
                return get_msg("orchestrator", f"silence_{threshold}s", language)

        return None

    def should_end_call(self, silence_seconds: float) -> bool:
        """Return True when silence has exceeded the end-call threshold."""
        return silence_seconds >= self.END_CALL_THRESHOLD

    def get_voice_step(
        self,
        elapsed_seconds: float,
    ) -> tuple[float, str, bool] | None:
        """Return the (threshold, i18n_key, is_terminal) step whose threshold
        exactly matches ``elapsed_seconds``, or None if no step fires.

        Used by the voice-edge silence watcher, which loops through
        ``SILENCE_STEPS`` and calls this as each threshold elapses.
        """
        for threshold, key, terminal in SILENCE_STEPS:
            if abs(elapsed_seconds - threshold) < 1e-6:
                return threshold, key, terminal
        return None
