"""Unit tests for Pipecat voice pipeline construction helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.telephony import agent_processor as ap
from vaidya.telephony.agent_processor import (
    UTTERANCE_DEBOUNCE_SECONDS,
    UTTERANCE_FLUSH_MIN_CONFIDENCE,
    UTTERANCE_FLUSH_SECONDS,
    VaidyaAgentProcessor,
    _looks_complete,
)
from vaidya.telephony.pipeline import (
    _TELEPHONY_SAMPLE_RATE,
    PIPECAT_AVAILABLE,
    _build_stt_service,
    _build_tts_service,
    _build_twilio_serializer,
    _build_websocket_params,
)

_requires_pipecat = pytest.mark.skipif(not PIPECAT_AVAILABLE, reason="pipecat-ai is not installed")


@_requires_pipecat
class TestVoicePipelineHelpers:
    def test_websocket_params_do_not_include_unsupported_vad_analyzer(self):
        serializer = _build_twilio_serializer(
            stream_sid="MZ123",
            twilio_call_sid="CA123",
            twilio_account_sid="",
            twilio_auth_token="",
        )
        params = _build_websocket_params(serializer)

        assert not hasattr(params, "vad_analyzer")
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True
        assert params.audio_in_sample_rate == _TELEPHONY_SAMPLE_RATE
        assert params.audio_out_sample_rate == _TELEPHONY_SAMPLE_RATE

    def test_stt_service_uses_sarvam_vad_signals_and_telephony_audio(self):
        stt = _build_stt_service("test-key")

        assert stt._init_sample_rate == _TELEPHONY_SAMPLE_RATE
        # "wav" is required: sarvamai streaming AudioData only accepts
        # encoding="audio/wav"; PCM codec labels are 16 kHz-only upstream.
        assert stt._input_audio_codec == "wav"
        assert stt._mode == "codemix"
        assert stt._settings.model == "saaras:v3"
        assert stt._settings.language is None
        assert stt._settings.vad_signals is True
        assert stt._settings.high_vad_sensitivity is True
        assert stt._settings.interrupt_min_speech_frames == 3

    def test_tts_service_uses_human_pacing_for_telephony(self):
        tts = _build_tts_service("test-key", "priya", "hi-IN")

        assert tts._init_sample_rate == _TELEPHONY_SAMPLE_RATE
        assert tts._settings.model == "bulbul:v3"
        assert tts._settings.voice == "priya"
        assert tts._settings.language == "hi-IN"
        assert tts._settings.pace == 0.94
        assert tts._settings.temperature == 0.55
        assert tts._settings.min_buffer_size == 35
        assert tts._settings.max_chunk_length == 130

    def test_twilio_serializer_disables_auto_hangup_without_credentials(self):
        serializer = _build_twilio_serializer(
            stream_sid="MZ123",
            twilio_call_sid="CA123",
            twilio_account_sid="",
            twilio_auth_token="",
        )

        assert serializer._params.auto_hang_up is False
        assert serializer._params.sample_rate == _TELEPHONY_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Adaptive utterance debounce (voice-edge latency tuning)
#
# These exercise pure helpers and the processor's fragment-buffering timer.
# The processor imports cleanly without pipecat-ai (the module ships frame
# stubs), so — unlike the pipeline-helper tests above — they run regardless
# of whether pipecat is installed.
# ---------------------------------------------------------------------------


class TestUtteranceCompletionHelper:
    """``_looks_complete`` detects sentence-final punctuation across scripts."""

    @pytest.mark.parametrize(
        "text",
        [
            "I have five people.",
            "How many in your family?",
            "मेरे परिवार में पाँच लोग हैं।",  # Devanagari danda
            "5 জন আছে।",  # Bengali danda
            "சரி.",  # Tamil + period
            "done!",
            "trailing whitespace ok.   ",  # trailing space is ignored
        ],
    )
    def test_complete_fragments(self, text):
        assert _looks_complete(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "how many people",  # mid-sentence, no terminator
            "मेरे परिवार में",  # mid-sentence Devanagari
            "five thousand rupees per",  # cut off mid-clause
            "",
            "   ",
        ],
    )
    def test_incomplete_fragments(self, text):
        assert _looks_complete(text) is False


class TestAdaptiveDebounceWindow:
    """``_debounce_window`` picks a short window only on a strong end signal."""

    def test_constants_invariants(self):
        # The early-flush window must be strictly shorter than the base quiet
        # window (that's the whole point), and the base window must be well
        # under the old 2.0s flat floor so single-fragment turns start sooner.
        assert 0 < UTTERANCE_FLUSH_SECONDS < UTTERANCE_DEBOUNCE_SECONDS
        assert UTTERANCE_DEBOUNCE_SECONDS < 2.0
        assert 0.0 < UTTERANCE_FLUSH_MIN_CONFIDENCE <= 1.0

    def test_complete_high_confidence_uses_short_window(self):
        window = VaidyaAgentProcessor._debounce_window("I have five people.", 0.95)
        assert window == UTTERANCE_FLUSH_SECONDS

    def test_complete_but_low_confidence_uses_full_window(self):
        # A sentence-final fragment we don't trust (low STT confidence) is
        # likely a misfire; wait the full window so corrections can merge.
        window = VaidyaAgentProcessor._debounce_window(
            "I have five people.", UTTERANCE_FLUSH_MIN_CONFIDENCE - 0.1
        )
        assert window == UTTERANCE_DEBOUNCE_SECONDS

    def test_partial_fragment_uses_full_window(self):
        # Mid-sentence fragment, even at high confidence, must wait the full
        # window so the rest of the slow sentence can still merge.
        window = VaidyaAgentProcessor._debounce_window("I have five", 0.99)
        assert window == UTTERANCE_DEBOUNCE_SECONDS

    def test_confidence_at_threshold_uses_short_window(self):
        window = VaidyaAgentProcessor._debounce_window(
            "I have five people.", UTTERANCE_FLUSH_MIN_CONFIDENCE
        )
        assert window == UTTERANCE_FLUSH_SECONDS


def _make_debounce_processor() -> tuple[VaidyaAgentProcessor, MagicMock]:
    """Build a processor with a mocked manager and real-task spawning.

    Works without pipecat-ai: the agent_processor module provides frame stubs,
    and we feed ``SimpleNamespace`` transcription frames (only ``text`` /
    ``language`` / ``confidence`` are read off the frame).
    """
    mgr = MagicMock()
    mgr.handle_turn = AsyncMock(return_value="bot reply")
    mgr.switch_language = AsyncMock(return_value=True)
    mgr.get_context = AsyncMock(return_value=None)

    proc = VaidyaAgentProcessor(
        conversation_manager=mgr,
        call_id="debounce-call",
        language="hi-IN",
    )
    # Spawn real asyncio tasks (the base FrameProcessor stub already does this,
    # but be explicit and accept the optional name kwarg the real one takes).
    proc.create_task = lambda coro, name=None: asyncio.create_task(coro)  # type: ignore[assignment]

    async def _noop_push(frame, direction=None):
        return None

    proc.push_frame = _noop_push  # type: ignore[assignment]
    return proc, mgr


def _frame(text: str, *, language: str | None = "hi-IN", confidence: float = 0.95):
    return SimpleNamespace(text=text, language=language, confidence=confidence)


async def _drain_turn(proc: VaidyaAgentProcessor) -> None:
    """Wait out the (patched-tiny) debounce and the spawned turn task."""
    for _ in range(200):
        if proc._debounce_task is None and not proc._turn_tasks:
            return
        if proc._turn_tasks:
            await asyncio.gather(*list(proc._turn_tasks))
        else:
            await asyncio.sleep(0.005)


class TestAdaptiveDebounceBehavior:
    """The fragment buffer resets on each fragment and flushes only on quiet."""

    @pytest.fixture(autouse=True)
    def _tiny_windows(self, monkeypatch):
        # Shrink both windows so the timer-driven tests don't sleep in real
        # time, preserving their ordering (flush < base) so behaviour holds.
        monkeypatch.setattr(ap, "UTTERANCE_DEBOUNCE_SECONDS", 0.06)
        monkeypatch.setattr(ap, "UTTERANCE_FLUSH_SECONDS", 0.02)

    @pytest.mark.asyncio
    async def test_single_complete_fragment_flushes_and_runs_turn(self):
        proc, mgr = _make_debounce_processor()

        await proc._on_transcription(_frame("Mujhe scheme chahiye."))
        await _drain_turn(proc)

        mgr.handle_turn.assert_awaited_once()
        assert mgr.handle_turn.await_args.args[1] == "Mujhe scheme chahiye."

    @pytest.mark.asyncio
    async def test_new_fragment_resets_the_timer(self, monkeypatch):
        """A fragment arriving before the window elapses cancels the pending
        flush and re-arms a fresh one — so the quiet is measured from the
        LAST fragment, and slow multi-fragment speech never splits."""
        proc, mgr = _make_debounce_processor()

        # First (partial) fragment arms a debounce task.
        await proc._on_transcription(_frame("Mere parivaar mein", confidence=0.95))
        first_task = proc._debounce_task
        assert first_task is not None

        # Second fragment arrives immediately: the first task must be cancelled
        # and a brand-new one armed (reset), with both fragments still buffered.
        await proc._on_transcription(_frame("paanch log hain.", confidence=0.95))
        assert proc._debounce_task is not None
        assert proc._debounce_task is not first_task
        # Give the event loop a tick so the cancelled task observes cancellation.
        await asyncio.sleep(0)
        assert first_task.cancelled()
        assert proc._pending_fragments == ["Mere parivaar mein", "paanch log hain."]

        # After quiet, both fragments merge into one turn (not two half-turns).
        await _drain_turn(proc)
        mgr.handle_turn.assert_awaited_once()
        assert mgr.handle_turn.await_args.args[1] == "Mere parivaar mein paanch log hain."

    @pytest.mark.asyncio
    async def test_multiple_fragments_merge_into_single_turn(self):
        proc, mgr = _make_debounce_processor()

        # Three slow fragments of one sentence, the last sentence-final.
        await proc._on_transcription(_frame("Mera", confidence=0.9))
        await asyncio.sleep(0)
        await proc._on_transcription(_frame("naam", confidence=0.9))
        await asyncio.sleep(0)
        await proc._on_transcription(_frame("Rajdeep hai.", confidence=0.9))
        await _drain_turn(proc)

        mgr.handle_turn.assert_awaited_once()
        assert mgr.handle_turn.await_args.args[1] == "Mera naam Rajdeep hai."

    @pytest.mark.asyncio
    async def test_partial_fragment_waits_then_flushes_alone(self):
        """A single mid-sentence fragment with no follow-up still flushes after
        the (full) quiet window — we never strand a turn."""
        proc, mgr = _make_debounce_processor()

        await proc._on_transcription(_frame("kuch problem hai", confidence=0.95))
        await _drain_turn(proc)

        mgr.handle_turn.assert_awaited_once()
        assert mgr.handle_turn.await_args.args[1] == "kuch problem hai"
