"""Unit tests for the agent processor's streaming-TTS leading-fragment split.

These verify that a long reply is split into a short, speakable leading
fragment (spoken first via ``TTSSpeakFrame`` for fast time-to-first-audio)
plus a remainder routed through the proven ``LLMFullResponse*`` envelope whose
end frame still forces the final flush. The cached fast-path, playback marks,
and silence-timer interactions must not regress.

Like the sibling ``test_agent_processor_tts_cache`` module, these exercise the
processor through the pipecat-less stub frames the module installs when
``pipecat-ai`` is absent, so they run with no external deps.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from vaidya.telephony import agent_processor as ap
from vaidya.telephony.agent_processor import (
    _STREAM_MIN_SPLIT_LENGTH,
    VaidyaAgentProcessor,
    _split_leading_fragment,
)

# ---------------------------------------------------------------------------
# _split_leading_fragment — pure boundary logic
# ---------------------------------------------------------------------------


def test_split_short_reply_returns_no_head() -> None:
    """A reply below the min split length is left whole (today's behaviour)."""
    text = "Aap kitne log hain?"
    assert len(text) < _STREAM_MIN_SPLIT_LENGTH
    assert _split_leading_fragment(text) == ("", text)


def test_split_empty_and_whitespace() -> None:
    assert _split_leading_fragment("") == ("", "")
    assert _split_leading_fragment("   ") == ("", "")


def test_split_prefers_sentence_boundary() -> None:
    """A long reply with an early sentence end splits there, head first."""
    text = "Hello there. How many people live in your home? Please tell me your income."
    head, tail = _split_leading_fragment(text)
    assert head == "Hello there. How many people live in your home?"
    assert tail == "Please tell me your income."
    # Boundary punctuation stays attached to the head.
    assert head.endswith("?")


def test_split_falls_back_to_clause_boundary_english() -> None:
    """With no early sentence end, the first clause comma is the head boundary."""
    text = (
        "Based on the details about your family and income, "
        "we can tell you about a few government schemes that may help you."
    )
    head, tail = _split_leading_fragment(text)
    assert head == "Based on the details about your family and income,"
    assert tail == "we can tell you about a few government schemes that may help you."


def test_split_clause_boundary_devanagari() -> None:
    """Indic clause comma is a valid head boundary (Hindi)."""
    text = "आपके परिवार की जानकारी के आधार पर, हम आपको कुछ सरकारी योजनाओं के बारे में बता सकते हैं।"
    head, tail = _split_leading_fragment(text)
    assert head == "आपके परिवार की जानकारी के आधार पर,"
    assert tail == "हम आपको कुछ सरकारी योजनाओं के बारे में बता सकते हैं।"


def test_split_sentence_boundary_devanagari_danda() -> None:
    """A Devanagari danda is treated as a sentence boundary."""
    text = "आपके परिवार की जानकारी के आधार पर, हम आपको योजनाओं के बारे में बता सकते हैं। कृपया बताइए।"
    head, tail = _split_leading_fragment(text)
    assert head.endswith("।")
    assert tail == "कृपया बताइए।"


def test_split_clause_boundary_tamil() -> None:
    """Clause comma split works for Tamil (multi-byte graphemes intact)."""
    text = "உங்கள் குடும்பத்தைப் பற்றிய தகவலின் அடிப்படையில், சில அரசு திட்டங்களைப் பற்றி நாங்கள் சொல்ல முடியும்."
    head, tail = _split_leading_fragment(text)
    assert head == "உங்கள் குடும்பத்தைப் பற்றிய தகவலின் அடிப்படையில்,"
    assert tail == "சில அரசு திட்டங்களைப் பற்றி நாங்கள் சொல்ல முடியும்."


def test_split_no_boundary_returns_no_head() -> None:
    """A long reply with no sentence/clause boundary is not split."""
    text = (
        "this is a fairly long reply that just keeps going without any "
        "punctuation marks whatsoever so there is nowhere clean to split it"
    )
    assert _split_leading_fragment(text) == ("", text)


def test_split_boundary_only_at_end_returns_no_head() -> None:
    """A single sentence whose only ender is the final word yields no head.

    Splitting at the very last word leaves no remainder to stream behind it,
    so there is no time-to-first-audio win and we keep the reply whole.
    """
    text = "this whole thing is one clause with the only period at the very end of it all done."
    assert _split_leading_fragment(text) == ("", text)


def test_split_does_not_break_off_tiny_sliver() -> None:
    """An early one-word clause is not split off as a clipped sliver."""
    text = "हाँ, इसके आधार पर हम आपको कई सरकारी योजनाओं के बारे में विस्तार से बता सकते हैं अभी।"
    head, _tail = _split_leading_fragment(text)
    # The leading "हाँ," (4 chars) is below the head minimum, so no head here
    # (the only sentence end is the final danda with no remainder).
    assert head == ""


def test_split_rejoin_invariant_across_inputs() -> None:
    """head + ' ' + tail must reproduce the (stripped) input exactly."""
    samples = [
        "Hello there. How many people live in your home? Please tell me your income.",
        "Based on the details about your family and income, we can tell you about schemes.",
        "आपके परिवार की जानकारी के आधार पर, हम आपको कुछ सरकारी योजनाओं के बारे में बता सकते हैं।",
        "உங்கள் குடும்பத்தைப் பற்றிய தகவலின் அடிப்படையில், சில அரசு திட்டங்களைப் பற்றி சொல்வோம்.",
        "এই তথ্যের ভিত্তিতে, আমরা আপনাকে কয়েকটি সরকারি প্রকল্প সম্পর্কে বলতে পারি যা কাজে আসবে।",
        "this is a fairly long reply with no punctuation at all so nothing splits here ok then",
        "short reply",
    ]
    for text in samples:
        head, tail = _split_leading_fragment(text)
        rejoined = f"{head} {tail}".strip() if head else tail
        assert rejoined == text.strip(), f"rejoin mismatch for {text!r}"
        if head:
            # Never split mid-word: the head is a whitespace-bounded prefix.
            assert text.strip().startswith(head)
            after = text.strip()[len(head) :]
            assert after == "" or after[0] == " ", f"mid-word split: {after[:8]!r}"
            assert tail, "non-empty head must leave a non-empty tail"


# ---------------------------------------------------------------------------
# _emit_bot_text — frame emission order / flush preservation
# ---------------------------------------------------------------------------


def _processor(cache: object = None) -> VaidyaAgentProcessor:
    mgr = MagicMock()
    mgr._sarvam_client = MagicMock(tts_cache=cache)
    proc = VaidyaAgentProcessor(conversation_manager=mgr, call_id="c1", language="hi-IN")
    proc.push_frame = AsyncMock()  # type: ignore[method-assign]
    proc._send_playback_mark = AsyncMock()  # type: ignore[method-assign]
    return proc


# A reply long enough to split, with a clear clause boundary.
_LONG_REPLY = (
    "Based on the details about your family and income, "
    "we can tell you about a few government schemes that may help you."
)


def _pushed_frames(proc: VaidyaAgentProcessor) -> list[object]:
    return [c.args[0] for c in proc.push_frame.call_args_list]


async def test_emit_speaks_head_first_then_envelope() -> None:
    """A long reply emits TTSSpeakFrame(head) BEFORE the envelope/tail."""
    proc = _processor()

    # send_mark=True is the real-reply path that streams a leading fragment;
    # _send_playback_mark is mocked so no extra frames are pushed.
    await proc._emit_bot_text(_LONG_REPLY, send_mark=True)

    frames = _pushed_frames(proc)
    # First frame out is the spoken head — audio starts before the full text.
    assert isinstance(frames[0], ap.TTSSpeakFrame)
    head, tail = _split_leading_fragment(_LONG_REPLY)
    assert frames[0].text == head
    # Followed by the proven envelope wrapping the remainder.
    assert isinstance(frames[1], ap.LLMFullResponseStartFrame)
    assert isinstance(frames[2], ap.TextFrame)
    assert frames[2].text == tail
    assert isinstance(frames[3], ap.LLMFullResponseEndFrame)
    assert len(frames) == 4


async def test_emit_head_precedes_full_text() -> None:
    """The head is strictly shorter than the full reply and emitted first."""
    proc = _processor()

    await proc._emit_bot_text(_LONG_REPLY, send_mark=True)

    frames = _pushed_frames(proc)
    head_frame = frames[0]
    assert isinstance(head_frame, ap.TTSSpeakFrame)
    # The first thing synthesized is a proper prefix of the whole reply, so the
    # caller hears it before the remainder has been pushed.
    assert _LONG_REPLY.startswith(head_frame.text)
    assert len(head_frame.text) < len(_LONG_REPLY)


async def test_emit_final_frame_still_flushes() -> None:
    """The last frame pushed is the end frame (preserves the flush)."""
    proc = _processor()

    await proc._emit_bot_text(_LONG_REPLY, send_mark=True)

    frames = _pushed_frames(proc)
    assert isinstance(frames[-1], ap.LLMFullResponseEndFrame)


async def test_emit_head_and_tail_recombine_to_full_text() -> None:
    """No words are dropped: head + tail equals the spoken text."""
    proc = _processor()

    await proc._emit_bot_text(_LONG_REPLY, send_mark=True)

    frames = _pushed_frames(proc)
    head = frames[0].text  # TTSSpeakFrame
    tail = frames[2].text  # TextFrame inside envelope
    assert f"{head} {tail}" == _LONG_REPLY


async def test_emit_short_reply_keeps_single_envelope() -> None:
    """A short reply emits no head: exactly Start/Text/End (today's behaviour)."""
    proc = _processor()
    short = "Aap kitne log hain ghar mein?"
    assert len(short) < _STREAM_MIN_SPLIT_LENGTH

    await proc._emit_bot_text(short, send_mark=False)

    frames = _pushed_frames(proc)
    assert not any(isinstance(f, ap.TTSSpeakFrame) for f in frames)
    assert isinstance(frames[0], ap.LLMFullResponseStartFrame)
    assert isinstance(frames[1], ap.TextFrame)
    assert frames[1].text == short
    assert isinstance(frames[2], ap.LLMFullResponseEndFrame)
    assert len(frames) == 3


async def test_emit_keepalive_never_speaks_head() -> None:
    """send_mark=False (keepalive) never splits a head, even when long.

    A keepalive interjection must not surface an extra BotStoppedSpeaking that
    could start the idle watcher before the reply has finished playing.
    """
    proc = _processor()

    await proc._emit_bot_text(_LONG_REPLY, send_mark=False)

    frames = _pushed_frames(proc)
    assert not any(isinstance(f, ap.TTSSpeakFrame) for f in frames)
    # Whole reply rides the single envelope unchanged.
    assert isinstance(frames[0], ap.LLMFullResponseStartFrame)
    assert frames[1].text == _LONG_REPLY
    assert isinstance(frames[2], ap.LLMFullResponseEndFrame)
    assert len(frames) == 3


async def test_emit_sends_single_mark_after_streamed_reply() -> None:
    """With send_mark, exactly one playback mark is requested, after the end frame."""
    proc = _processor()
    proc._playback_marks_enabled = True

    await proc._emit_bot_text(_LONG_REPLY, send_mark=True)

    # One mark for the whole reply (head + tail), so the idle watcher restarts
    # only after Twilio confirms both have played.
    proc._send_playback_mark.assert_awaited_once()
    frames = _pushed_frames(proc)
    # The head was streamed first, the end frame still closes the envelope.
    assert isinstance(frames[0], ap.TTSSpeakFrame)
    assert isinstance(frames[-1], ap.LLMFullResponseEndFrame)


async def test_emit_stream_false_forces_single_envelope() -> None:
    """stream=False (terminal closure) never splits, even for a long reply.

    The terminal closure pushes EndTaskFrame right after, so it must play as a
    single uninterrupted audio context that the teardown cannot truncate.
    """
    proc = _processor()

    await proc._emit_bot_text(_LONG_REPLY, send_mark=True, stream=False)

    frames = _pushed_frames(proc)
    assert not any(isinstance(f, ap.TTSSpeakFrame) for f in frames)
    assert isinstance(frames[0], ap.LLMFullResponseStartFrame)
    assert frames[1].text == _LONG_REPLY
    assert isinstance(frames[2], ap.LLMFullResponseEndFrame)
    assert len(frames) == 3


async def test_emit_long_reply_without_boundary_still_flushes() -> None:
    """A long reply with no sentence/clause boundary uses the envelope + flush.

    No head can be split, so the whole reply rides the envelope and the end
    frame still flushes it (no regression vs. today).
    """
    proc = _processor()
    no_boundary = (
        "this is a fairly long reply that just keeps going without any "
        "punctuation marks whatsoever so there is nowhere clean to split"
    )
    assert len(no_boundary) >= _STREAM_MIN_SPLIT_LENGTH

    await proc._emit_bot_text(no_boundary, send_mark=True)

    frames = _pushed_frames(proc)
    assert not any(isinstance(f, ap.TTSSpeakFrame) for f in frames)
    assert isinstance(frames[0], ap.LLMFullResponseStartFrame)
    assert frames[1].text == no_boundary
    assert isinstance(frames[-1], ap.LLMFullResponseEndFrame)


async def test_emit_cache_hit_skips_both_text_and_head(monkeypatch) -> None:
    """On a cache hit, neither the head nor the envelope is emitted."""
    monkeypatch.setattr(ap, "TTS_AUDIO_FRAMES_AVAILABLE", True)
    monkeypatch.setattr(ap, "TTSStartedFrame", MagicMock())
    monkeypatch.setattr(ap, "TTSStoppedFrame", MagicMock())
    monkeypatch.setattr(ap, "TTSAudioRawFrame", MagicMock())

    proc = _processor()
    # Make the cached-audio fast-path succeed.
    proc._try_emit_cached_audio = AsyncMock(return_value=True)  # type: ignore[method-assign]

    await proc._emit_bot_text(_LONG_REPLY, send_mark=False)

    # Fast-path returned True → no text/head frames pushed at all.
    proc.push_frame.assert_not_called()
