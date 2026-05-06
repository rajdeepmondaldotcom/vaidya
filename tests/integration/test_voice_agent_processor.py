"""Integration tests for VaidyaAgentProcessor voice-edge behavior.

These tests exercise:
- Language auto-detect + TTSUpdateSettingsFrame push on first transcription.
- Normal transcription routing to ConversationManager.handle_turn.
- Idle-loop silence escalation (nudge/reprompt/closure + EndTaskFrame upstream).

The idle-loop test monkey-patches ``SILENCE_STEPS`` with sub-second thresholds
so the tests complete quickly without real-time sleeps.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.telephony import agent_processor as ap_module
from vaidya.telephony.agent_processor import (
    PIPECAT_AVAILABLE,
    VaidyaAgentProcessor,
    _normalize_lang_code,
)

pytestmark = pytest.mark.skipif(not PIPECAT_AVAILABLE, reason="pipecat-ai is not installed")


def _make_processor(language: str = "hi-IN") -> tuple[VaidyaAgentProcessor, MagicMock, list]:
    """Build a processor with a mocked conversation manager and a frame-capture list.

    Replaces ``push_frame`` on the instance so we can observe frames without
    needing a real Pipecat pipeline.
    """
    mgr = MagicMock()
    mgr.handle_turn = AsyncMock(return_value="bot response text")
    mgr.switch_language = AsyncMock(return_value=True)
    mgr.handle_silence = AsyncMock(return_value=("silence-spoken-text", False))
    mgr.get_context = AsyncMock(return_value=None)

    proc = VaidyaAgentProcessor(
        conversation_manager=mgr,
        call_id="test-call",
        language=language,
    )

    captured: list = []

    async def _capture(frame, direction=None):
        captured.append((frame, direction))

    proc.push_frame = _capture  # type: ignore[assignment]
    return proc, mgr, captured


def _tts_setting(frame, key: str):
    if frame.delta is not None:
        return getattr(frame.delta, key)
    return frame.settings[key]


# ---------------------------------------------------------------------------
# Language auto-switch
# ---------------------------------------------------------------------------


class TestLanguageAutoSwitch:
    @pytest.mark.asyncio
    async def test_first_transcription_switches_language(self):
        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")

        frame = TranscriptionFrame(
            text="Vanakkam, naan Chennai-il irukkiren",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="ta-IN",
        )
        await proc._on_transcription(frame)

        mgr.switch_language.assert_awaited_once_with("test-call", "ta-IN")
        # First captured frame should be the TTS settings update
        tts_frames = [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)]
        assert len(tts_frames) == 1
        assert _tts_setting(tts_frames[0], "language") == "ta-IN"
        assert _tts_setting(tts_frames[0], "voice") == "kavitha"
        # Processor's own language state was updated
        assert proc._language == "ta-IN"

    @pytest.mark.asyncio
    async def test_language_name_overrides_stt_language_tag(self):
        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")

        frame = TranscriptionFrame(
            text="Tamil",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="en-IN",
        )
        await proc._on_transcription(frame)

        mgr.switch_language.assert_awaited_once_with("test-call", "ta-IN")
        tts_frames = [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)]
        assert len(tts_frames) == 1
        assert _tts_setting(tts_frames[0], "language") == "ta-IN"
        assert _tts_setting(tts_frames[0], "voice") == "kavitha"

    @pytest.mark.asyncio
    async def test_language_name_does_not_override_after_welcome(self):
        from types import SimpleNamespace

        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")
        mgr.get_context = AsyncMock(
            return_value=SimpleNamespace(
                language="hi-IN",
                metadata={"awaiting_language": False},
            )
        )

        frame = TranscriptionFrame(
            text="Tamil Nadu",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="hi-IN",
        )
        await proc._on_transcription(frame)

        mgr.switch_language.assert_not_awaited()
        assert [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)] == []
        assert proc._language == "hi-IN"

    @pytest.mark.asyncio
    async def test_same_language_does_not_push_settings_frame(self):
        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")

        frame = TranscriptionFrame(
            text="Namaste", user_id="u1", timestamp="2026-04-17T00:00:00Z", language="hi-IN"
        )
        await proc._on_transcription(frame)

        mgr.switch_language.assert_not_awaited()
        tts_frames = [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)]
        assert tts_frames == []

    @pytest.mark.asyncio
    async def test_unsupported_language_does_not_switch(self):
        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")
        # manager returns False for unsupported
        mgr.switch_language = AsyncMock(return_value=False)

        frame = TranscriptionFrame(
            text="Bonjour", user_id="u1", timestamp="2026-04-17T00:00:00Z", language="fr-FR"
        )
        await proc._on_transcription(frame)

        tts_frames = [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)]
        assert tts_frames == []
        # Language state unchanged
        assert proc._language == "hi-IN"

    @pytest.mark.asyncio
    async def test_no_language_on_frame_does_not_lock(self):
        """If STT returns no language (garbled cough / short utterance),
        we must NOT lock — a later, cleaner transcription should get to
        drive the language switch."""
        from pipecat.frames.frames import TranscriptionFrame

        proc, mgr, _ = _make_processor(language="hi-IN")
        assert proc._language_locked is False

        # First transcription with no language info
        frame_noisy = TranscriptionFrame(
            text="ahh umm", user_id="u1", timestamp="2026-04-17T00:00:00Z", language=None
        )
        await proc._on_transcription(frame_noisy)
        # Not locked yet
        assert proc._language_locked is False
        mgr.switch_language.assert_not_awaited()

        # Second transcription with clear Tamil detection
        frame_clear = TranscriptionFrame(
            text="Naan Chennai-il irukkiren",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="ta-IN",
        )
        await proc._on_transcription(frame_clear)
        # Now locked, and switch happened
        assert proc._language_locked is True
        mgr.switch_language.assert_awaited_once_with("test-call", "ta-IN")

    @pytest.mark.asyncio
    async def test_unsupported_language_on_frame_still_locks(self):
        """Urdu/French/etc. isn't in our supported voice languages. Lock
        into the current default (hi-IN) and proceed — don't keep trying
        to switch on every subsequent turn."""
        from pipecat.frames.frames import TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")
        frame = TranscriptionFrame(
            text="Bonjour", user_id="u1", timestamp="2026-04-17T00:00:00Z", language="fr-FR"
        )
        await proc._on_transcription(frame)

        # Locked
        assert proc._language_locked is True
        # But no switch / no TTS settings frame
        mgr.switch_language.assert_not_awaited()
        assert [f for f, _ in captured if isinstance(f, TTSUpdateSettingsFrame)] == []
        # Stays in default language
        assert proc._language == "hi-IN"

    def test_normalizes_pipecat_language_enum_and_odia_alias(self):
        from pipecat.transcriptions.language import Language

        assert _normalize_lang_code(Language.TA_IN) == "ta-IN"
        assert _normalize_lang_code(Language.OR_IN) == "od-IN"

    def test_normalize_rejects_unsupported_language(self):
        assert _normalize_lang_code("fr-FR") is None


# ---------------------------------------------------------------------------
# Normal transcription routing
# ---------------------------------------------------------------------------


class TestTranscriptionRouting:
    @pytest.mark.asyncio
    async def test_transcription_calls_handle_turn_and_pushes_text(self):
        from pipecat.frames.frames import TextFrame, TranscriptionFrame

        proc, mgr, captured = _make_processor()

        frame = TranscriptionFrame(
            text="Mujhe scheme chahiye",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="hi-IN",
        )
        await proc._on_transcription(frame)

        mgr.handle_turn.assert_awaited_once()
        call_args = mgr.handle_turn.await_args
        # Positional: call_id, user_text, language
        assert call_args.args[0] == "test-call"
        assert call_args.args[1] == "Mujhe scheme chahiye"
        # channel kwarg must be "voice"
        assert call_args.kwargs["channel"] == "voice"

        text_frames = [f for f, _ in captured if isinstance(f, TextFrame)]
        assert len(text_frames) == 1
        assert text_frames[0].text == "bot response text"

    @pytest.mark.asyncio
    async def test_empty_transcription_is_ignored(self):
        from pipecat.frames.frames import TranscriptionFrame

        proc, mgr, captured = _make_processor()

        frame = TranscriptionFrame(
            text="   ",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="hi-IN",
        )
        await proc._on_transcription(frame)
        mgr.handle_turn.assert_not_awaited()
        assert captured == []

    @pytest.mark.asyncio
    async def test_handle_turn_failure_pushes_fallback(self):
        from pipecat.frames.frames import TextFrame, TranscriptionFrame

        proc, mgr, captured = _make_processor(language="hi-IN")
        mgr.handle_turn = AsyncMock(side_effect=RuntimeError("LLM down"))

        frame = TranscriptionFrame(
            text="aaj kuch problem hai",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language="hi-IN",
        )
        await proc._on_transcription(frame)

        text_frames = [f for f, _ in captured if isinstance(f, TextFrame)]
        assert len(text_frames) == 1
        # Hindi fallback
        assert "Maaf" in text_frames[0].text

    @pytest.mark.asyncio
    async def test_syncs_language_changed_by_manager_before_tts(self):
        from types import SimpleNamespace

        from pipecat.frames.frames import TextFrame, TranscriptionFrame, TTSUpdateSettingsFrame

        proc, mgr, captured = _make_processor(language="hi-IN")
        mgr.get_context = AsyncMock(return_value=SimpleNamespace(language="ta-IN"))

        frame = TranscriptionFrame(
            text="Tamil",
            user_id="u1",
            timestamp="2026-04-17T00:00:00Z",
            language=None,
        )
        await proc._on_transcription(frame)

        tts_indices = [
            idx for idx, (f, _) in enumerate(captured) if isinstance(f, TTSUpdateSettingsFrame)
        ]
        text_indices = [idx for idx, (f, _) in enumerate(captured) if isinstance(f, TextFrame)]
        assert tts_indices
        assert text_indices
        assert max(tts_indices) < min(text_indices)
        assert proc._language == "ta-IN"


# ---------------------------------------------------------------------------
# Idle loop (silence watcher)
# ---------------------------------------------------------------------------


class TestIdleLoop:
    @pytest.mark.asyncio
    async def test_idle_loop_fires_all_steps_and_ends_task(self, monkeypatch):
        """Tiny thresholds → 3 silence frames + 1 EndTaskFrame upstream."""
        from pipecat.frames.frames import EndTaskFrame, TextFrame
        from pipecat.processors.frame_processor import FrameDirection

        # Short thresholds so the test finishes fast
        monkeypatch.setattr(
            ap_module,
            "SILENCE_STEPS",
            [
                (0.02, "silence_nudge", False),
                (0.04, "silence_reprompt_prefix", False),
                (0.06, "silence_closure", True),
            ],
        )

        proc, mgr, captured = _make_processor()

        # Sequence of (spoken, terminal) responses — mirrors the real ConversationManager
        mgr.handle_silence = AsyncMock(
            side_effect=[
                ("nudge text", False),
                ("reprompt text", False),
                ("closure text", True),
            ]
        )

        import asyncio

        proc._wake = asyncio.Event()
        # Drive the idle loop directly
        await proc._idle_loop()

        # All three silence steps fired
        assert mgr.handle_silence.await_count == 3

        # Three TextFrames pushed downstream
        text_frames = [f for f, d in captured if isinstance(f, TextFrame)]
        assert [t.text for t in text_frames] == ["nudge text", "reprompt text", "closure text"]

        # EndTaskFrame pushed upstream
        end_frames = [(f, d) for f, d in captured if isinstance(f, EndTaskFrame)]
        assert len(end_frames) == 1
        _, direction = end_frames[0]
        assert direction == FrameDirection.UPSTREAM

    @pytest.mark.asyncio
    async def test_idle_loop_stops_when_user_wakes(self, monkeypatch):
        """If _wake is set, the loop exits cleanly on the next iteration."""
        import asyncio

        from pipecat.frames.frames import EndTaskFrame, TextFrame

        monkeypatch.setattr(
            ap_module,
            "SILENCE_STEPS",
            [
                (0.05, "silence_nudge", False),
                (0.10, "silence_reprompt_prefix", False),
                (0.15, "silence_closure", True),
            ],
        )

        proc, mgr, captured = _make_processor()
        proc._wake = asyncio.Event()
        # Caller "speaks" before any threshold fires
        proc._wake.set()

        await proc._idle_loop()

        # No silence frames emitted, no EndTask
        assert mgr.handle_silence.await_count == 0
        assert [f for f, _ in captured if isinstance(f, TextFrame)] == []
        assert [f for f, _ in captured if isinstance(f, EndTaskFrame)] == []
