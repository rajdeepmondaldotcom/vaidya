"""Unit tests for the agent processor's TTS-cache integration helpers.

These exercise the pure helpers and the cached-audio emission decision without
requiring pipecat (the module imports via stubs when pipecat is absent).
"""

from __future__ import annotations

import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock

from vaidya.telephony import agent_processor as ap
from vaidya.telephony.agent_processor import (
    VaidyaAgentProcessor,
    _pcm_from_tts_bytes,
    _resolve_tts_cache,
)


def _wav(rate: int = 8000, channels: int = 1, width: int = 2, frames: int = 100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01\x02" * frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _pcm_from_tts_bytes
# ---------------------------------------------------------------------------


def test_pcm_decodes_base64_wav_at_telephony_format() -> None:
    pcm = _pcm_from_tts_bytes(base64.b64encode(_wav(8000)))
    assert pcm == b"\x01\x02" * 100  # WAV header stripped, raw PCM returned


def test_pcm_decodes_raw_wav_bytes() -> None:
    # Already-raw WAV bytes (not base64) still decode via the fallback path.
    pcm = _pcm_from_tts_bytes(_wav(8000))
    assert pcm == b"\x01\x02" * 100


def test_pcm_rejects_wrong_sample_rate() -> None:
    assert _pcm_from_tts_bytes(base64.b64encode(_wav(16000))) is None


def test_pcm_rejects_stereo() -> None:
    assert _pcm_from_tts_bytes(base64.b64encode(_wav(8000, channels=2))) is None


def test_pcm_rejects_8bit_samples() -> None:
    assert _pcm_from_tts_bytes(base64.b64encode(_wav(8000, width=1))) is None


def test_pcm_rejects_garbage_and_none() -> None:
    assert _pcm_from_tts_bytes(b"not-audio-bytes") is None
    assert _pcm_from_tts_bytes(None) is None
    assert _pcm_from_tts_bytes(b"") is None
    assert _pcm_from_tts_bytes("a string") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_tts_cache
# ---------------------------------------------------------------------------


def test_resolve_cache_from_manager_client() -> None:
    client = MagicMock()
    client.tts_cache = "CACHE"
    mgr = MagicMock()
    mgr._sarvam_client = client
    assert _resolve_tts_cache(mgr) == "CACHE"


def test_resolve_cache_none_when_no_client() -> None:
    mgr = MagicMock()
    mgr._sarvam_client = None
    assert _resolve_tts_cache(mgr) is None


def test_resolve_cache_none_when_manager_lacks_attr() -> None:
    assert _resolve_tts_cache(object()) is None


# ---------------------------------------------------------------------------
# _emit_bot_text / _try_emit_cached_audio decision
# ---------------------------------------------------------------------------


def _processor_with_cache(cache: object) -> VaidyaAgentProcessor:
    mgr = MagicMock()
    mgr._sarvam_client = MagicMock(tts_cache=cache)
    proc = VaidyaAgentProcessor(conversation_manager=mgr, call_id="c1", language="hi-IN")
    # push_frame is a stub on the pipecat-less FrameProcessor; track calls.
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]
    proc._send_playback_mark = AsyncMock()  # type: ignore[method-assign]
    return proc


async def test_emit_uses_text_path_when_audio_frames_unavailable(monkeypatch) -> None:
    """With audio frames unavailable, the cache is never consulted (text path)."""
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", False)
    cache = AsyncMock()
    proc = _processor_with_cache(cache)

    await proc._emit_bot_text("Namaste", send_mark=False)

    cache.get_or_render.assert_not_called()
    # Text-path envelope: start/text/end frames pushed.
    assert proc.push_frame.await_count == 3


async def test_try_emit_returns_false_without_cache(monkeypatch) -> None:
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    proc = _processor_with_cache(None)  # no cache wired
    assert await proc._try_emit_cached_audio("Namaste", send_mark=False) is False


async def test_try_emit_falls_back_when_render_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    cache = AsyncMock()
    cache.get_or_render = AsyncMock(return_value=None)
    proc = _processor_with_cache(cache)

    assert await proc._try_emit_cached_audio("Namaste", send_mark=False) is False
    cache.get_or_render.assert_awaited_once()
    # No audio frames pushed on a fall-back.
    proc.push_frame.assert_not_called()


async def test_try_emit_falls_back_on_format_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    cache = AsyncMock()
    # 16 kHz audio does not match telephony format → decode returns None.
    cache.get_or_render = AsyncMock(return_value=base64.b64encode(_wav(16000)))
    proc = _processor_with_cache(cache)

    assert await proc._try_emit_cached_audio("Namaste", send_mark=False) is False
    proc.push_frame.assert_not_called()


async def test_try_emit_falls_back_when_render_raises(monkeypatch) -> None:
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    cache = AsyncMock()
    cache.get_or_render = AsyncMock(side_effect=RuntimeError("boom"))
    proc = _processor_with_cache(cache)

    # Must swallow the error and fall back, not propagate.
    assert await proc._try_emit_cached_audio("Namaste", send_mark=False) is False
    proc.push_frame.assert_not_called()


async def test_try_emit_skips_empty_text(monkeypatch) -> None:
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    cache = AsyncMock()
    proc = _processor_with_cache(cache)
    assert await proc._try_emit_cached_audio("", send_mark=False) is False
    cache.get_or_render.assert_not_called()


class _StubFrame:
    """Minimal stand-in for a pipecat audio frame (pipecat not installed)."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _enable_audio_frames(monkeypatch) -> None:
    """Simulate pipecat audio frames being importable for the emission path."""
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    monkeypatch.setattr(ap, "TTSStartedFrame", _StubFrame)
    monkeypatch.setattr(ap, "TTSStoppedFrame", _StubFrame)
    monkeypatch.setattr(ap, "TTSAudioRawFrame", _StubFrame)


async def test_cache_key_uses_call_tts_pace_not_default(monkeypatch) -> None:
    """The render is keyed on the call's tts_pace so it matches the live render."""
    _enable_audio_frames(monkeypatch)
    cache = AsyncMock()
    cache.get_or_render = AsyncMock(return_value=base64.b64encode(_wav(8000)))
    mgr = MagicMock()
    mgr._sarvam_client = MagicMock(tts_cache=cache)
    proc = VaidyaAgentProcessor(
        conversation_manager=mgr,
        call_id="c1",
        language="hi-IN",
        tts_pace=0.92,
        tts_speaker="anushka",
    )
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]
    proc._send_playback_mark = AsyncMock()  # type: ignore[method-assign]

    emitted = await proc._try_emit_cached_audio("Namaste", send_mark=False)

    assert emitted is True
    _, kwargs = cache.get_or_render.call_args
    assert kwargs["pace"] == 0.92
    assert kwargs["speaker"] == "anushka"
    assert kwargs["sample_rate"] == ap._TELEPHONY_SAMPLE_RATE
    assert kwargs["model"] == ap._TTS_CACHE_MODEL
    # TTS envelope: started / audio / stopped — and no bot-speaking frames.
    assert proc.push_frame.await_count == 3


async def test_emit_cached_audio_sends_mark_when_enabled(monkeypatch) -> None:
    """On a cache hit with send_mark, the playback mark is requested (text path skipped)."""
    _enable_audio_frames(monkeypatch)
    cache = AsyncMock()
    cache.get_or_render = AsyncMock(return_value=base64.b64encode(_wav(8000)))
    proc = _processor_with_cache(cache)
    proc._playback_marks_enabled = True

    await proc._emit_bot_text("Namaste", send_mark=True)

    proc._send_playback_mark.assert_awaited_once()
    # Audio path used (3 frames), not the 3-frame text envelope — distinguish
    # by confirming the cache was consulted and a TTSAudioRawFrame went out.
    cache.get_or_render.assert_awaited_once()
    pushed_types = [type(c.args[0]).__name__ for c in proc.push_frame.call_args_list]
    assert pushed_types.count("_StubFrame") == 3
