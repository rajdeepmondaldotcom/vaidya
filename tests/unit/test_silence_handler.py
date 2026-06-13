"""Unit tests for SilenceHandler and the voice-edge SILENCE_STEPS table."""

from __future__ import annotations

import pytest

from vaidya.agents.constants import SILENCE_STEPS
from vaidya.agents.silence import SilenceHandler

# ---------------------------------------------------------------------------
# SILENCE_STEPS
# ---------------------------------------------------------------------------


class TestSilenceSteps:
    def test_has_three_steps(self):
        assert len(SILENCE_STEPS) == 3

    def test_thresholds_are_10_20_32(self):
        thresholds = [s[0] for s in SILENCE_STEPS]
        assert thresholds == [10.0, 20.0, 32.0]

    def test_keys_match_i18n(self):
        keys = [s[1] for s in SILENCE_STEPS]
        assert keys == ["silence_nudge", "silence_reprompt_prefix", "silence_closure"]

    def test_only_20s_is_terminal(self):
        terminals = [s[2] for s in SILENCE_STEPS]
        assert terminals == [False, False, True]

    def test_thresholds_are_ascending(self):
        thresholds = [s[0] for s in SILENCE_STEPS]
        assert thresholds == sorted(thresholds)


# ---------------------------------------------------------------------------
# SilenceHandler.get_voice_step
# ---------------------------------------------------------------------------


class TestGetVoiceStep:
    def setup_method(self):
        self.handler = SilenceHandler()

    def test_returns_nudge_at_10s(self):
        step = self.handler.get_voice_step(10.0)
        assert step is not None
        threshold, key, terminal = step
        assert threshold == 10.0
        assert key == "silence_nudge"
        assert terminal is False

    def test_returns_reprompt_at_20s(self):
        step = self.handler.get_voice_step(20.0)
        assert step is not None
        threshold, key, terminal = step
        assert threshold == 20.0
        assert key == "silence_reprompt_prefix"
        assert terminal is False

    def test_returns_closure_at_32s_terminal(self):
        step = self.handler.get_voice_step(32.0)
        assert step is not None
        threshold, key, terminal = step
        assert threshold == 32.0
        assert key == "silence_closure"
        assert terminal is True

    @pytest.mark.parametrize("elapsed", [0.0, 3.0, 5.9, 7.0, 11.0, 13.0, 19.0, 25.0])
    def test_returns_none_off_threshold(self, elapsed: float):
        assert self.handler.get_voice_step(elapsed) is None


# ---------------------------------------------------------------------------
# SilenceHandler.get_silence_response (legacy text/simulation path)
# ---------------------------------------------------------------------------


class TestLegacyGetSilenceResponse:
    """The legacy text-channel path is still used by simulation tests."""

    def setup_method(self):
        self.handler = SilenceHandler()

    def test_returns_none_below_5s(self):
        assert self.handler.get_silence_response(3.0, "hi-IN") is None

    def test_returns_string_at_5s(self):
        msg = self.handler.get_silence_response(5.0, "hi-IN")
        assert msg is not None
        assert len(msg) > 0

    def test_returns_none_at_end_threshold(self):
        assert (
            self.handler.get_silence_response(SilenceHandler.END_CALL_THRESHOLD, "hi-IN") is None
        )

    def test_should_end_call_at_20s(self):
        assert self.handler.should_end_call(32.0) is True
        assert self.handler.should_end_call(19.0) is False
