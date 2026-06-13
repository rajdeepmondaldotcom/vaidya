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
import logging
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
        TTSUpdateSettingsFrame,
        UserStartedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.sarvam.tts import SarvamTTSSettings

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

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
# longer than a normal intake turn (now ~3-5s: one fast LLM call + two
# translations), then a progress note on an interval. The ack delay is set
# ABOVE the intake-turn time so quick turns just answer (no "one second"
# filler); only the multi-call eligibility crunch (~15-25s) trips it.
PROCESSING_ACK_DELAY_SECONDS = 7.0
PROCESSING_PROGRESS_INTERVAL_SECONDS = 14.0
# Must cover the longest agent phase (eligibility+reviewer).
PROCESSING_PROGRESS_MAX_NOTES = 12

# Sarvam's VAD splits natural pauses mid-sentence ("আমার পরিবারে" +
# "5 জন আছে।" arrive as two transcripts). Launching a turn per fragment
# makes the agent answer half-sentences and re-ask questions. Buffer
# fragments and start the turn after this much transcript quiet.
# Fragment transcripts arrive up to ~2s apart on slow speech, so the
# window must comfortably exceed that.
UTTERANCE_DEBOUNCE_SECONDS = 2.0

_DEDUPE_STRIP_CHARS = ".,!?।|॥'\"-—:; \t\n"


def _normalize_for_dedupe(text: str) -> str:
    """Normalize an utterance for duplicate detection across STT retries."""
    cleaned = "".join(ch for ch in text.lower() if ch not in _DEDUPE_STRIP_CHARS)
    return cleaned


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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mgr = conversation_manager
        self._call_id = call_id
        self._language = language
        self._wake: asyncio.Event = asyncio.Event()
        self._idle_task: asyncio.Task[None] | None = None
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Route frames: handle silence signals, auto-switch language, route transcriptions."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TwilioPlaybackMarkFrame):
            self._on_playback_mark(frame)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            if not self._pending_playback_mark:
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
        starts only after ``UTTERANCE_DEBOUNCE_SECONDS`` of transcript
        quiet, with the fragments merged into one utterance. Turns then run
        as spawned tasks: agent work takes 10-60s and blocking
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
        self._pending_confidence = float(getattr(frame, "confidence", 1.0) or 1.0)

        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = self.create_task(self._flush_utterance_after_debounce())

    async def _flush_utterance_after_debounce(self) -> None:
        """After transcript quiet, merge fragments and launch the turn."""
        try:
            await asyncio.sleep(UTTERANCE_DEBOUNCE_SECONDS)
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
            await self._emit_bot_text(fallback, profile="repair")
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
        """
        spoken = format_for_tts(text, profile=profile)
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=spoken))
        await self.push_frame(LLMFullResponseEndFrame())
        if send_mark and self._playback_marks_enabled:
            await self._send_playback_mark()

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
        """Wait through SILENCE_STEPS thresholds, escalating on each timeout."""
        elapsed = 0.0
        for threshold, _key, terminal in await self._silence_steps():
            wait_for = threshold - elapsed
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=wait_for)
                return  # user spoke -> silence broken
            except TimeoutError:
                pass
            elapsed = threshold

            if self._inflight_text is not None:
                # The "silence" is ours, not the caller's: a turn is being
                # processed and the keepalive owns the line. Nudging with
                # "I'm listening, speak" mid-processing makes callers
                # repeat themselves and stack slow turns.
                return

            try:
                spoken, is_terminal = await self._mgr.handle_silence(self._call_id, elapsed)
            except Exception:
                logger.error(
                    "handle_silence failed",
                    extra={"call_id": self._call_id, "elapsed": elapsed},
                    exc_info=True,
                )
                return

            if not spoken:
                continue

            await self._emit_bot_text(spoken, profile="repair")

            if is_terminal or terminal:
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
