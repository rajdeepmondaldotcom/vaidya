"""Custom Pipecat processor that bridges STT output to Vaidya's ConversationManager.

Instead of routing transcribed speech through a generic LLM, this processor
feeds it into Vaidya's multi-agent orchestrator and pushes the response text
downstream to TTS.

It also handles real-world voice UX:

- **Silence watching**: starts an idle timer whenever the bot stops speaking
  and cancels it when the user starts speaking or a transcription arrives.
  Escalates through ``SILENCE_STEPS`` (6s nudge, 12s reprompt, 20s closure)
  and, on the terminal step, pushes :class:`EndTaskFrame` upstream so the
  pipeline hangs the call up cleanly via the Twilio serializer.
- **Language auto-switch**: on the first transcription with a detected
  language, if it differs from the session default, persists the new
  language into the session and pushes a :class:`TTSUpdateSettingsFrame`
  so subsequent replies speak in the caller's language.

Requires ``pipecat-ai`` -- guarded import so the rest of the app works without it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import wave
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, cast

from vaidya.agents.constants import SILENCE_STEPS
from vaidya.i18n import get_msg
from vaidya.telephony.twilio_serializer import (
    TwilioPlaybackMarkFrame,
    TwilioPlaybackMarkRequestFrame,
)
from vaidya.voice.language import (
    TTS_SPEAKERS,
    detect_language_from_text,
    is_filler_utterance,
    is_voice_language,
    normalize_language,
)
from vaidya.voice.prosody import format_for_tts

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        EndTaskFrame,
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        TranscriptionFrame,
        TTSSpeakFrame,
        TTSUpdateSettingsFrame,
        UserStartedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.sarvam.tts import SarvamTTSSettings

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

# Audio-output frames are imported separately: emitting pre-rendered cached
# audio (instead of pushing text to the downstream TTS service) requires these
# frame types, and their availability/signature is more version-sensitive than
# the core frames above. If they can't be imported, the cache still renders
# bytes but emission falls back to the proven TextFrame path.
try:
    from pipecat.frames.frames import (  # type: ignore[attr-defined]
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )

    TTS_AUDIO_FRAMES_AVAILABLE = True
except ImportError:
    TTS_AUDIO_FRAMES_AVAILABLE = False
    # Defensive Any-typed placeholders so the cached-audio path fails safe to
    # the text path (via the TTS_AUDIO_FRAMES_AVAILABLE guard) rather than
    # NameError if the flag and these names ever diverge.
    TTSAudioRawFrame: Any = None
    TTSStartedFrame: Any = None
    TTSStoppedFrame: Any = None

    # Stubs so the module can be imported without pipecat.
    class FrameProcessor:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def process_frame(self, frame: object, direction: object) -> None:
            pass

        async def push_frame(self, frame: object, direction: object | None = None) -> None:
            pass

        def create_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
            return asyncio.create_task(coro)

        async def cancel_task(self, task: asyncio.Task[None] | None) -> None:
            if task:
                task.cancel()

        async def cleanup(self) -> None:
            pass

    class Frame:  # type: ignore[no-redef]
        pass

    class TextFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = "") -> None:
            self.text = text

    class TTSSpeakFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = "") -> None:
            self.text = text

    class TranscriptionFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = "", language: str | None = None) -> None:
            self.text = text
            self.language = language

    class BotStoppedSpeakingFrame(Frame):  # type: ignore[no-redef]
        pass

    class UserStartedSpeakingFrame(Frame):  # type: ignore[no-redef]
        pass

    class EndTaskFrame(Frame):  # type: ignore[no-redef]
        pass

    class LLMFullResponseStartFrame(Frame):  # type: ignore[no-redef]
        pass

    class LLMFullResponseEndFrame(Frame):  # type: ignore[no-redef]
        pass

    class TTSUpdateSettingsFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, settings: dict[str, Any] | None = None, delta: object = None) -> None:
            self.settings = settings or {}
            self.delta = delta

    class SarvamTTSSettings:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FrameDirection:  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"


if TYPE_CHECKING:
    from vaidya.models.conversation import ConversationContext
    from vaidya.pipeline.conversation import ConversationManager


# Turn-processing keepalive: speak a short ack only if a turn is taking
# longer than a normal intake turn, then a progress note on an interval. The
# ack delay is set just ABOVE a normal intake turn so quick turns just answer
# (no "one second" filler); only slow turns trip it. With the lean extraction
# prompt an intake question now resolves in ~5-8s, so 8.0s keeps EVERY intake
# turn silent (snappier -- no repetitive "one moment" before each answer) while
# the multi-call eligibility crunch (~20-30s) still reassures. The crunch also
# has its own "finding the right plans for you" filler, so the slightly later
# keepalive there is invisible.
PROCESSING_ACK_DELAY_SECONDS = 8.0
# 8s, not 15s: at 15s a slow Sarvam crunch left 13-17s of dead air between
# reassurances (observed on a real call), past the ~9s point silence starts to
# feel like a dropped line. 8s keeps the gap under that bar. The keepalive is
# cancelled the instant the real reply is ready, so a fast crunch still never
# hears more than the opening "finding the right plans" filler.
PROCESSING_PROGRESS_INTERVAL_SECONDS = 8.0
# A few unobtrusive reassurances over the eligibility crunch — NOT a stream.
# At 12 every long turn buried the actual reply under a wall of "still working"
# notes; 6 at an 8s cadence covers ~55s (the worst-case slow-Sarvam crunch)
# without the chatter, and the reply cancels the loop early on a normal crunch.
PROCESSING_PROGRESS_MAX_NOTES = 6

# Sarvam's VAD splits natural pauses mid-sentence ("আমার পরিবারে" +
# "5 জন আছে।" arrive as two transcripts). Launching a turn per fragment
# makes the agent answer half-sentences and re-ask questions. Buffer
# fragments and start the turn only after a window of transcript *quiet*.
#
# The window is ADAPTIVE rather than a flat floor on every turn:
#
# - The timer RESETS whenever a new fragment arrives (``_on_transcription``
#   cancels and re-arms it), so multi-fragment slow speech still merges —
#   only true quiet, measured from the LAST fragment, ever flushes.
# - A non-terminal fragment (no sentence-final punctuation yet, i.e. the
#   speaker is probably mid-sentence) waits the conservative
#   ``UTTERANCE_DEBOUNCE_SECONDS`` window. This is the floor that has to
#   tolerate Sarvam's real inter-fragment gaps inside one slow sentence, so
#   it stays well above them — only modestly below the old 2.0s flat floor.
# - When the latest fragment carries a strong end-of-utterance signal — it
#   ends on sentence-final punctuation AND STT confidence is high — we flush
#   after the much shorter ``UTTERANCE_FLUSH_SECONDS`` window. That is the
#   common case (Saaras tags a finished utterance with terminal punctuation)
#   and the big latency win: a single complete sentence starts ~1.5s sooner
#   than the old 2.0s floor, with no risk of splitting because a fragment
#   that already *ended* a sentence has no mid-sentence remainder to merge.
# Measured against live Sarvam streaming STT: a single spoken answer arrives as
# 2-4 transcript fragments spread over 2-3s, with inter-fragment gaps up to ~2.1s
# (e.g. "Ami West Bengal-e thaki." [+2.1s] "Hooghly jelar ekta grame"). The window
# re-arms on every fragment, so it only has to exceed the largest *inter-fragment*
# gap to merge the whole utterance. 2.5s clears the observed gaps; below it the
# turn launches on a partial answer and the bot re-asks / desyncs.
UTTERANCE_DEBOUNCE_SECONDS = 2.5
# No fast-flush shortcut: Saaras tags *fragments* with sentence-final punctuation
# (not just the final one), so "ends on a period" is NOT a reliable end-of-turn
# signal — it was splitting multi-fragment answers after 0.4s. Keep the flush
# window equal to the full debounce so every utterance gets the same quiet window.
UTTERANCE_FLUSH_SECONDS = 2.5
# Below this STT confidence we don't trust the end-of-utterance shortcut and
# fall back to the full quiet window (sarvam reports per-transcript confidence
# in 0..1; mid/low-confidence fragments are often mid-thought partials). When
# the streaming frame carries no confidence field, ``_on_transcription``
# defaults to 1.0, so a terminal-punctuation fragment still takes the fast
# path on the strength of the punctuation signal alone.
UTTERANCE_FLUSH_MIN_CONFIDENCE = 0.6

_DEDUPE_STRIP_CHARS = ".,!?।|॥'\"-—:; \t\n"

# Telephony audio target: 8 kHz, mono, 16-bit PCM. Cached prompts whose decoded
# WAV matches this exactly can be emitted as raw audio frames; anything else
# falls back to the text path so we never push a mismatched waveform to Twilio.
_TELEPHONY_SAMPLE_RATE = 8000
_TELEPHONY_CHANNELS = 1
_TELEPHONY_SAMPLE_WIDTH = 2
_TTS_CACHE_MODEL = "bulbul:v3"

# Sentence-ending punctuation across the 11 voice scripts (Latin + Indic),
# matching pipecat's own SENTENCE_ENDING_PUNCTUATION for the languages we
# support. A fragment that ends on one of these is a complete utterance the
# TTS can synthesize and speak on its own.
_SENTENCE_ENDERS = frozenset(".?!;…।॥।。？！；．")
# Clause separators: weaker boundaries used to break an over-long leading
# sentence so the FIRST spoken fragment stays short (fast time-to-first-audio)
# without ever splitting mid-word. Comma, colon, semicolon, dashes, and the
# full-width / Indic / Arabic comma variants.
_CLAUSE_SEPARATORS = frozenset(",:;—–-،，、；：")
# Lower bound (chars / code points) for the leading fragment. Large enough
# that the head is a natural clause (~6+ words) rather than a clipped two-word
# sliver, yet small enough to catch common Indic clause boundaries (e.g. a
# Hindi/Tamil "..., " around 30 chars) so audio starts almost immediately.
# Only applies once the whole reply is already past _STREAM_MIN_SPLIT_LENGTH,
# so the remaining tail is always substantial too.
_STREAM_HEAD_MIN_CHARS = 28
# If the leading sentence/clause is longer than this we still speak it as the
# head (better to start a slightly longer fragment now than buffer everything);
# this only bounds how far we scan for the first boundary.
_STREAM_HEAD_MAX_CHARS = 140
# Below this total length there is nothing to gain from splitting a head off:
# the whole reply is short enough that the end-frame flush is already prompt.
_STREAM_MIN_SPLIT_LENGTH = 60


def _split_leading_fragment(text: str) -> tuple[str, str]:
    """Split *text* into a short, speakable leading fragment and the remainder.

    Returns ``(head, tail)``. The ``head`` is meant to be spoken IMMEDIATELY
    (via a TTS speak frame) so the caller hears audio without waiting for the
    whole reply to synthesize; ``tail`` is everything after it and flows
    through the normal sentence-aggregating TTS envelope.

    Boundary selection (longest-first preference for a natural unit):

    1. Prefer the first **sentence** boundary (``. ? ! । ॥`` …) at or past
       :data:`_STREAM_HEAD_MIN_CHARS`.
    2. Otherwise fall back to the first **clause** boundary (comma/colon/dash,
       incl. Indic/Arabic/full-width variants) at or past the minimum.

    Robustness guarantees:

    - Boundaries are only taken at whitespace-delimited word ends, so a run of
      non-whitespace characters — and therefore any multi-byte grapheme in
      Devanagari, Tamil, Bengali, Odia, etc. — is never cut in half.
    - Boundary punctuation stays attached to the ``head`` so prosody is intact.
    - Concatenating ``head`` and ``tail`` with a single space reproduces the
      input (already single-space normalized by :func:`format_for_tts`), so the
      total synthesized audio is unchanged.

    When there is no clean early boundary (or the reply is short), returns
    ``("", text)`` so the caller keeps today's single-envelope behaviour with
    no regression.
    """
    text = text.strip()
    # Short replies: the end-frame flush already starts them promptly.
    if len(text) < _STREAM_MIN_SPLIT_LENGTH:
        return "", text

    words = text.split()
    sentence_split: int | None = None
    clause_split: int | None = None
    length = 0
    for idx, word in enumerate(words):
        length += len(word) + (1 if length else 0)
        if length < _STREAM_HEAD_MIN_CHARS:
            continue
        last = word[-1]
        if sentence_split is None and last in _SENTENCE_ENDERS:
            sentence_split = idx + 1
            break  # a sentence boundary is the best head; stop scanning
        if clause_split is None and last in _CLAUSE_SEPARATORS:
            clause_split = idx + 1
        if length >= _STREAM_HEAD_MAX_CHARS:
            break  # don't scan unboundedly for a head

    # Prefer a sentence boundary, but only if it leaves a real remainder to
    # stream behind it. If the only sentence end is the final word, fall back
    # to the clause boundary we found earlier (also requiring a remainder).
    # A split at the very last word yields no streaming win, so we skip it.
    last_index = len(words)
    for split_at in (sentence_split, clause_split):
        if split_at is not None and split_at < last_index:
            return " ".join(words[:split_at]), " ".join(words[split_at:])
    return "", text


def _normalize_for_dedupe(text: str) -> str:
    """Normalize an utterance for duplicate detection across STT retries."""
    cleaned = "".join(ch for ch in text.lower() if ch not in _DEDUPE_STRIP_CHARS)
    return cleaned


def _looks_complete(text: str) -> bool:
    """True when *text* ends on sentence-final punctuation across our scripts.

    Sarvam tags a transcript fragment with terminal punctuation (``.`` ``?``
    ``।`` …) when its VAD believes the utterance ended, so a fragment ending
    this way — paired with high STT confidence — is a strong end-of-utterance
    signal we can flush on quickly instead of waiting out the full quiet window.
    Trailing whitespace is ignored; an empty/whitespace fragment is never
    "complete".
    """
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_ENDERS


def _pcm_from_tts_bytes(audio: bytes | None) -> bytes | None:
    """Decode Sarvam REST TTS output into raw 8 kHz mono 16-bit PCM, or None.

    ``SarvamClient.tts`` returns the SDK's ``audios[0]`` — a base64-encoded WAV
    string. To emit it through the telephony transport (which is configured
    with ``add_wav_header=False``) we must base64-decode, strip the WAV
    container, and confirm the format matches telephony exactly. Any deviation
    (wrong rate/channels/width, non-WAV, undecodable) returns None so the caller
    falls back to the text path rather than playing a mismatched waveform.
    """
    if not audio:
        return None

    raw: bytes
    if isinstance(audio, (bytes, bytearray)):
        raw = bytes(audio)
    else:
        return None

    # The SDK hands back base64 text (as bytes or str). Decode it to the WAV
    # container; if it's already raw WAV bytes, the decode will fail and we
    # treat the original bytes as the container.
    wav_bytes: bytes
    try:
        wav_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        wav_bytes = raw

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if (
                wav_file.getframerate() != _TELEPHONY_SAMPLE_RATE
                or wav_file.getnchannels() != _TELEPHONY_CHANNELS
                or wav_file.getsampwidth() != _TELEPHONY_SAMPLE_WIDTH
            ):
                return None
            return wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error, OSError):
        return None


def _resolve_tts_cache(conversation_manager: object) -> Any:
    """Find the shared TTS cache the app lifespan attached to the Sarvam client.

    The pipeline constructs :class:`VaidyaAgentProcessor` without a ``tts_cache``
    kwarg, so the single process-wide cache is reached through the conversation
    manager's Sarvam client (``app.state`` wiring puts it there). Returns the
    cache or ``None`` if it isn't wired (e.g. text-only deployments / tests).
    """
    client = getattr(conversation_manager, "_sarvam_client", None)
    return getattr(client, "tts_cache", None)


class VaidyaAgentProcessor(FrameProcessor):
    """Bridges Pipecat STT transcription to Vaidya's multi-agent pipeline.

    Also runs a silence watcher and a one-shot language auto-switcher at
    the voice edge of the pipeline.
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        call_id: str,
        language: str,
        playback_marks_enabled: bool = True,
        tts_cache: Any = None,
        tts_pace: float = 0.94,
        tts_speaker: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mgr = conversation_manager
        self._call_id = call_id
        self._language = language
        self._wake: asyncio.Event = asyncio.Event()
        self._idle_task: asyncio.Task[None] | None = None
        # Silence escalation passed so far (seconds). Persists across the watch
        # restarts that each nudge's playback-mark triggers, so successive silent
        # stretches ESCALATE (nudge -> reprompt -> graceful close) instead of
        # re-firing the first nudge forever. Reset to 0 whenever the caller speaks.
        self._silence_elapsed: float = 0.0
        self._language_locked: bool = False
        self._playback_marks_enabled = playback_marks_enabled
        self._pending_playback_mark: str | None = None
        self._mark_counter = 0
        # In-flight turn bookkeeping (turns run as spawned tasks so the
        # frame pipeline stays responsive during 10-60s agent work).
        self._inflight_text: str | None = None
        self._turn_tasks: set[asyncio.Task[None]] = set()
        self._keepalive_task: asyncio.Task[None] | None = None
        # Fragment buffer for utterance debouncing.
        self._pending_fragments: list[str] = []
        self._pending_language: object | None = None
        self._pending_confidence: float = 1.0
        self._debounce_task: asyncio.Task[None] | None = None
        # TTS audio cache for fixed/templated prompts. The pipeline builds this
        # processor without the kwarg today, so fall back to the shared cache
        # attached to the conversation manager's Sarvam client (wired in the app
        # lifespan). The cache key must use the SAME pace the downstream Sarvam
        # TTS service was built with, or cached audio would differ from a live
        # render — that is the call's tts_pace, NOT a per-utterance profile.
        self._tts_cache = (
            tts_cache if tts_cache is not None else _resolve_tts_cache(conversation_manager)
        )
        self._tts_pace = tts_pace
        self._tts_speaker = tts_speaker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Route frames: handle silence signals, auto-switch language, route transcriptions."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TwilioPlaybackMarkFrame):
            self._on_playback_mark(frame)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            # Start the idle/silence watch only when a real turn is NOT being
            # processed. Keepalive interjections ("one moment") emit without a
            # playback mark; without this guard their BotStoppedSpeaking starts a
            # watch mid-processing whose nudge then glues onto the reply. The real
            # reply restarts the watch cleanly via its own playback mark.
            if not self._pending_playback_mark and self._inflight_text is None:
                self._start_idle_watch()
        elif isinstance(frame, UserStartedSpeakingFrame | TranscriptionFrame):
            self._cancel_idle_watch(interrupted=True)

        if isinstance(frame, TranscriptionFrame):
            await self._on_transcription(frame)
            return

        # Pass through everything else unchanged
        await self.push_frame(frame, direction)

    async def _on_transcription(self, frame: TranscriptionFrame) -> None:
        """Buffer the transcript fragment and (re)start the debounce timer.

        Sarvam's VAD splits slow natural speech into fragments; the turn
        starts only after a window of transcript quiet, with the fragments
        merged into one utterance. The window is adaptive (see
        :meth:`_debounce_window`): a finished-looking, high-confidence
        fragment flushes fast, anything else waits the full quiet window so
        a trailing fragment can still merge. Re-arming on every fragment
        means the quiet is always measured from the LAST fragment, so
        multi-fragment slow speech never splits.

        Turns then run as spawned tasks: agent work takes 10-60s and blocking
        ``process_frame`` for that long would stall every queued frame.
        """
        user_text = (frame.text or "").strip()
        if not user_text:
            return

        logger.debug(
            "STT transcription received",
            extra={"call_id": self._call_id, "text_length": len(user_text)},
        )

        # Callers repeat themselves into dead air while a slow turn is
        # processing. Re-running the same utterance stacks 30s turns and
        # produces duplicate replies — drop exact repeats of the turn
        # that is already in flight.
        normalized = _normalize_for_dedupe(user_text)
        if self._inflight_text is not None and normalized == self._inflight_text:
            logger.info(
                "Dropping duplicate utterance while turn in flight",
                extra={"call_id": self._call_id},
            )
            return

        self._pending_fragments.append(user_text)
        language = getattr(frame, "language", None)
        if language is not None:
            self._pending_language = language
        confidence = float(getattr(frame, "confidence", 1.0) or 1.0)
        self._pending_confidence = confidence

        # Choose the quiet window from THIS fragment (the latest one): a
        # complete-looking, high-confidence fragment flushes fast; otherwise
        # we wait the full window so a trailing fragment can still merge.
        window = self._debounce_window(user_text, confidence)

        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = self.create_task(self._flush_utterance_after_debounce(window))

    @staticmethod
    def _debounce_window(latest_fragment: str, confidence: float) -> float:
        """Pick the transcript-quiet window before flushing this utterance.

        Returns :data:`UTTERANCE_FLUSH_SECONDS` when the latest fragment is a
        strong end-of-utterance signal (ends on sentence-final punctuation and
        STT confidence is high), else :data:`UTTERANCE_DEBOUNCE_SECONDS`. The
        shorter window still leaves room for an immediately-following fragment
        to merge, while a mid-thought partial waits out the full window.
        """
        if confidence >= UTTERANCE_FLUSH_MIN_CONFIDENCE and _looks_complete(latest_fragment):
            return UTTERANCE_FLUSH_SECONDS
        return UTTERANCE_DEBOUNCE_SECONDS

    async def _flush_utterance_after_debounce(self, window: float) -> None:
        """After ``window`` seconds of transcript quiet, merge and launch the turn."""
        try:
            await asyncio.sleep(window)
        except asyncio.CancelledError:
            return  # superseded by a newer fragment

        self._debounce_task = None
        fragments, self._pending_fragments = self._pending_fragments, []
        stt_language, self._pending_language = self._pending_language, None
        user_text = " ".join(fragments).strip()
        if not user_text:
            return

        if not self._language_locked:
            # Only lock once we have a credible language signal. Short
            # mumbled utterances with no STT language tag leave us
            # unlocked so the next (cleaner) transcription can pick.
            if is_filler_utterance(user_text):
                # "Okay"/"Haan"/"Hello" carry no language signal. Release
                # the welcome gate in the current language so the call
                # advances, but stay unlocked so the caller's first real
                # sentence picks the language.
                await self._mgr.switch_language(self._call_id, self._language)
            else:
                self._language_locked = await self._maybe_switch_language(
                    await self._language_signal_for_turn(user_text, stt_language)
                )

        self._inflight_text = _normalize_for_dedupe(user_text)
        self._start_processing_keepalive()
        task = self.create_task(self._run_turn(user_text, stt_confidence=self._pending_confidence))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _run_turn(self, user_text: str, stt_confidence: float) -> None:
        """Run one user turn through the multi-agent pipeline and speak it."""
        try:
            response = await self._mgr.handle_turn(
                self._call_id,
                user_text,
                self._language,
                stt_confidence=stt_confidence,
                channel="voice",
            )
            self._stop_processing_keepalive()
            await self._sync_language_from_context()
            await self._emit_bot_text(response)
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(
                "Voice agent processing failed: %s",
                error_type,
                extra={"call_id": self._call_id, "error": str(exc)[:200]},
                exc_info=True,
            )
            self._stop_processing_keepalive()
            fallback = get_msg("conversation", "error", self._language)
            # Speak the short error message as one uninterrupted utterance (no
            # leading-fragment split) so it stays intact and clear.
            await self._emit_bot_text(fallback, profile="repair", stream=False)
        finally:
            self._inflight_text = None
            self._stop_processing_keepalive()

    # ------------------------------------------------------------------
    # Processing keepalive ("heard you, working on it")
    # ------------------------------------------------------------------

    def _start_processing_keepalive(self) -> None:
        self._stop_processing_keepalive()
        self._keepalive_task = self.create_task(self._processing_keepalive())

    def _stop_processing_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is not None:
            task.cancel()

    async def _processing_keepalive(self) -> None:
        """Acknowledge quickly, then keep the line alive while agents work."""
        try:
            await asyncio.sleep(PROCESSING_ACK_DELAY_SECONDS)
            await self._emit_bot_text(
                get_msg("conversation", "processing_ack", self._language),
                send_mark=False,
            )
            for _ in range(PROCESSING_PROGRESS_MAX_NOTES):
                await asyncio.sleep(PROCESSING_PROGRESS_INTERVAL_SECONDS)
                await self._emit_bot_text(
                    get_msg("conversation", "still_working", self._language),
                    send_mark=False,
                )
        except asyncio.CancelledError:
            pass

    async def _get_context(self) -> ConversationContext | None:
        getter = getattr(self._mgr, "get_context", None)
        if getter is None:
            return None
        try:
            context = await getter(self._call_id)
            return None if context is None else cast("ConversationContext", context)
        except (AttributeError, TypeError):
            return None

    async def _language_signal_for_turn(self, user_text: str, stt_language: object) -> str | None:
        """Prefer explicit language-name utterances over STT language tags.

        During the opening prompt, callers often say a language name in a
        different language ("Tamil", "English", "Hindi"). The STT language
        tag for that word can be English or Hindi, but the user's intent is
        the named language. Lexical detection must therefore win over STT,
        but only while the session is actually waiting for a language.
        """
        context = await self._get_context()
        metadata = getattr(context, "metadata", {}) if context is not None else {}
        if context is None or metadata.get("awaiting_language"):
            detected = detect_language_from_text(user_text)
            if detected is not None:
                return detected.value
        if stt_language is None:
            return None
        value = getattr(stt_language, "value", stt_language)
        return str(value)

    async def _sync_language_from_context(self) -> None:
        """Reflect orchestrator-side language changes into the downstream TTS."""
        context = await self._get_context()
        if context is None or context.language == self._language:
            return

        normalized = _normalize_lang_code(context.language)
        if not normalized:
            return

        self._language = normalized
        voice = _speaker_for_language(normalized)
        await self.push_frame(
            TTSUpdateSettingsFrame(
                delta=SarvamTTSSettings(voice=voice, language=normalized),
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _maybe_switch_language(self, detected: str | None) -> bool:
        """Handle STT-detected language on the first utterance.

        Returns ``True`` once we have a credible signal (so the caller
        should stop trying to auto-detect) and ``False`` when the signal
        is too weak to decide (no language field on the STT frame).

        Behaviour:

        - No detected language → don't lock, try again on next transcription.
        - Detected language unsupported (e.g. ``fr-FR``) → lock, keep the
          current default language; the orchestrator will proceed in it.
        - Detected language == current session language → lock, no TTS
          settings change needed.
        - Detected language is a supported voice language different from
          current → call :meth:`ConversationManager.switch_language`, push
          a :class:`TTSUpdateSettingsFrame` so the next TTS utterance uses
          the caller's voice/language, and lock.
        """
        if not detected:
            return False

        normalized = _normalize_lang_code(detected)
        if not normalized:
            logger.info(
                "Unsupported language detected; continuing in default",
                extra={
                    "call_id": self._call_id,
                    "detected": detected,
                    "falling_back_to": self._language,
                },
            )
            return True

        if normalized == self._language:
            # Same as the session default still counts as the caller's
            # choice — let the manager confirm it so the welcome language
            # gate releases instead of re-prompting for a language name.
            await self._mgr.switch_language(self._call_id, normalized)
            return True

        switched = await self._mgr.switch_language(self._call_id, normalized)
        if not switched:
            return True

        self._language = normalized
        voice = _speaker_for_language(normalized)
        logger.info(
            "Switching TTS voice/language",
            extra={"call_id": self._call_id, "language": normalized, "voice": voice},
        )
        await self.push_frame(
            TTSUpdateSettingsFrame(
                delta=SarvamTTSSettings(voice=voice, language=normalized),
            ),
            FrameDirection.DOWNSTREAM,
        )
        return True

    async def _emit_bot_text(
        self,
        text: str,
        profile: str = "default",
        send_mark: bool = True,
        stream: bool = True,
    ) -> None:
        """Push text to TTS and, for Twilio, request a playback-complete mark.

        The text is wrapped in an ``LLMFullResponseStartFrame`` /
        ``LLMFullResponseEndFrame`` envelope: the TTS service only flushes
        its sentence aggregation AND tells Sarvam to flush its server-side
        text buffer on the end frame. A bare ``TextFrame`` leaves the last
        sentence (and anything under ``min_buffer_size`` chars) stuck until
        a later turn's text pushes it out — replies glue together and
        arrive minutes late.

        ``send_mark=False`` is for keepalive interjections ("one moment")
        that must not restart the idle watcher when their playback acks.

        Fixed/templated prompts (greeting, intake questions, processing filler,
        silence nudges, closure) are byte-identical every call, so we try the
        TTS cache first: on a hit (or first-call render) the audio plays from
        memory instead of round-tripping to Sarvam's TTS service. Any
        uncertainty — no cache, cache miss returning None, unavailable audio
        frames, or a decoded waveform that doesn't match telephony format —
        falls back to the proven TextFrame path below, so emission is never
        worse than today.

        Time-to-first-audio: for a long reply we split off a short leading
        fragment and speak it FIRST via a ``TTSSpeakFrame``, which the
        downstream Sarvam service synthesizes immediately instead of buffering
        the whole paragraph behind its sentence-aggregation / ``min_buffer_size``
        gate. The remainder still flows through the ``LLMFullResponseStart`` /
        ``LLMFullResponseEnd`` envelope so its final sentence is flushed by the
        end frame. Short replies (and the cached fast-path) are unchanged.

        ``stream=False`` forces the whole reply through a single audio context
        (no leading-fragment split). The terminal closure uses it: it pushes
        an ``EndTaskFrame`` immediately after, and a single uninterrupted
        context is least likely to be truncated by the teardown than two
        contexts with a gap between them.
        """
        spoken = format_for_tts(text, profile=profile)

        if await self._try_emit_cached_audio(spoken, send_mark=send_mark):
            return

        # Only stream a leading fragment for real replies (send_mark=True) that
        # are not about to be torn down (stream=True). Keepalive interjections
        # ("one moment") pass send_mark=False precisely so their playback must
        # NOT touch the idle watcher; speaking a head fragment would surface an
        # extra BotStoppedSpeaking with no pending mark to suppress it. They are
        # also already short, so there is nothing to gain. Keep their exact
        # single-envelope behaviour.
        head, tail = _split_leading_fragment(spoken) if (send_mark and stream) else ("", spoken)
        if head:
            # Speak the leading fragment right away. TTSSpeakFrame bypasses the
            # downstream sentence aggregator and synthesizes its own context
            # immediately, so the caller hears audio while `tail` is still being
            # synthesized. The output transport derives Bot{Started,Stopped}-
            # SpeakingFrame from the audio it plays, so we do not push those
            # ourselves (mirrors the cached-audio path). The pending playback
            # mark (set below) suppresses the head's BotStoppedSpeaking from
            # starting the idle watch before the tail has played.
            await self.push_frame(TTSSpeakFrame(text=head))

        # Remainder (or the whole reply when there was no clean early split)
        # goes through the proven envelope: the end frame forces the TTS to
        # flush its sentence aggregation AND Sarvam's server-side buffer, so the
        # last sentence is never left stuck.
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=tail))
        await self.push_frame(LLMFullResponseEndFrame())
        if send_mark and self._playback_marks_enabled:
            await self._send_playback_mark()

    async def _try_emit_cached_audio(self, spoken: str, *, send_mark: bool) -> bool:
        """Emit pre-rendered cached audio for *spoken*; return False to fall back.

        Returns ``True`` only when cached audio was successfully pushed as raw
        telephony PCM frames. Returns ``False`` (so :meth:`_emit_bot_text` uses
        the TextFrame path) when the cache is absent, audio frames are
        unavailable, the render fails/returns None, or the decoded waveform
        doesn't match the telephony format.
        """
        if self._tts_cache is None or not TTS_AUDIO_FRAMES_AVAILABLE or not spoken:
            return False

        speaker = self._tts_speaker or _speaker_for_language(self._language)
        try:
            audio = await self._tts_cache.get_or_render(
                spoken,
                self._language,
                pace=self._tts_pace,
                speaker=speaker,
                model=_TTS_CACHE_MODEL,
                sample_rate=_TELEPHONY_SAMPLE_RATE,
            )
        except Exception as exc:  # noqa: BLE001 - cache failure must fall back, not crash
            logger.warning(
                "TTS cache render failed; using text path",
                extra={"call_id": self._call_id, "error": str(exc)[:200]},
            )
            return False

        pcm = _pcm_from_tts_bytes(audio)
        if not pcm:
            return False

        # Emit only the TTS envelope around the audio. The output transport
        # derives Bot{Started,Stopped}SpeakingFrame from the audio it plays, so
        # pushing those ourselves would double-signal and race the idle watcher
        # / interruption gate (the normal TextFrame path never pushes them).
        await self.push_frame(TTSStartedFrame())
        await self.push_frame(
            TTSAudioRawFrame(
                audio=pcm,
                sample_rate=_TELEPHONY_SAMPLE_RATE,
                num_channels=_TELEPHONY_CHANNELS,
            )
        )
        await self.push_frame(TTSStoppedFrame())
        if send_mark and self._playback_marks_enabled:
            await self._send_playback_mark()
        return True

    async def _send_playback_mark(self) -> None:
        self._mark_counter += 1
        mark_name = f"vaidya-bot-{self._mark_counter}"
        self._pending_playback_mark = mark_name
        await self.push_frame(TwilioPlaybackMarkRequestFrame(mark_name=mark_name))

    def _on_playback_mark(self, frame: TwilioPlaybackMarkFrame) -> None:
        """Start silence timing only after Twilio confirms playback completion."""
        if not self._pending_playback_mark:
            return
        if frame.mark_name != self._pending_playback_mark:
            return
        self._pending_playback_mark = None
        self._start_idle_watch()

    # ------------------------------------------------------------------
    # Idle timer (silence watcher)
    # ------------------------------------------------------------------

    def _start_idle_watch(self) -> None:
        self._cancel_idle_watch()
        self._wake = asyncio.Event()
        self._idle_task = self.create_task(self._idle_loop())

    def _cancel_idle_watch(self, *, interrupted: bool = False) -> None:
        if interrupted:
            self._pending_playback_mark = None
            # The caller spoke (or a new turn began): reset the silence escalation
            # so the next silent stretch starts again from a gentle nudge.
            self._silence_elapsed = 0.0
        if self._idle_task is not None:
            self._wake.set()
            self._idle_task = None

    async def cleanup(self) -> None:
        """Stop watchers and in-flight turn tasks when Pipecat tears down."""
        task = self._idle_task
        self._pending_playback_mark = None
        if task is not None:
            self._wake.set()
            self._idle_task = None
            await self.cancel_task(task)
        self._stop_processing_keepalive()
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            self._debounce_task = None
        for turn_task in list(self._turn_tasks):
            await self.cancel_task(turn_task)
        self._turn_tasks.clear()
        await super().cleanup()  # type: ignore[no-untyped-call]

    async def _idle_loop(self) -> None:
        """Wait through the silence steps, ESCALATING across watch restarts.

        ``self._silence_elapsed`` persists the thresholds already fired — each
        nudge's playback mark restarts this loop — so a continuously-silent caller
        progresses nudge -> reprompt -> graceful closure instead of re-hearing the
        first nudge forever. It resets to 0 the instant the caller speaks
        (``_cancel_idle_watch(interrupted=True)``).
        """
        for threshold, _key, terminal in await self._silence_steps():
            if threshold <= self._silence_elapsed:
                continue  # already escalated past this step in an earlier stretch
            wait_for = threshold - self._silence_elapsed
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=wait_for)
                return  # user spoke -> silence broken
            except TimeoutError:
                pass
            self._silence_elapsed = threshold

            if self._inflight_text is not None:
                # The "silence" is ours, not the caller's: a turn is being
                # processed and the keepalive owns the line. Nudging with
                # "I'm listening, speak" mid-processing makes callers
                # repeat themselves and stack slow turns.
                return

            try:
                spoken, is_terminal = await self._mgr.handle_silence(self._call_id, threshold)
            except Exception:
                logger.error(
                    "handle_silence failed",
                    extra={"call_id": self._call_id, "elapsed": threshold},
                    exc_info=True,
                )
                return

            if not spoken:
                continue

            ending = is_terminal or terminal
            # The terminal closure is followed immediately by EndTaskFrame, so
            # speak it as a single uninterrupted audio context (stream=False):
            # a leading-fragment split would add a second context the teardown
            # could truncate. Non-terminal nudges stream normally.
            await self._emit_bot_text(spoken, profile="repair", stream=not ending)

            if ending:
                # Give the TTS a moment to flush, then end the task.
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                return
            if self._playback_marks_enabled:
                return

    async def _silence_steps(self) -> list[tuple[float, str, bool]]:
        try:
            steps = await self._mgr.voice_silence_steps(self._call_id)
        except (AttributeError, TypeError):
            return SILENCE_STEPS
        return steps or SILENCE_STEPS


def _speaker_for_language(language: str) -> str:
    """Return the configured TTS speaker for a normalized voice language."""
    return TTS_SPEAKERS.get(normalize_language(language), "priya")


def _normalize_lang_code(raw: object) -> str | None:
    """Map a raw STT language tag to a supported BCP-47 voice language code.

    Accepts plain strings and Pipecat's ``Language`` enum values. Sarvam/
    Pipecat use ``or-IN`` for Odia in some paths; Vaidya's public language
    code is ``od-IN``.
    """
    if not raw:
        return None
    value = getattr(raw, "value", raw)
    candidate = str(value).strip()
    if not candidate:
        return None
    if not is_voice_language(candidate):
        return None
    return normalize_language(candidate).value
