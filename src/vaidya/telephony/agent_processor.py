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
from typing import TYPE_CHECKING

from vaidya.agents.constants import SILENCE_STEPS
from vaidya.i18n import get_msg
from vaidya.voice.language import (
    TTS_SPEAKERS,
    detect_language_from_text,
    is_voice_language,
    normalize_language,
)

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        EndTaskFrame,
        Frame,
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
        def __init__(self, **kwargs):
            pass

        async def process_frame(self, frame, direction):
            pass

        async def push_frame(self, frame, direction=None):
            pass

        def create_task(self, coro):
            return asyncio.create_task(coro)

        async def cancel_task(self, task):
            if task:
                task.cancel()

        async def cleanup(self):
            pass

    class Frame:  # type: ignore[no-redef]
        pass

    class TextFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = ""):
            self.text = text

    class TranscriptionFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = "", language: str | None = None):
            self.text = text
            self.language = language

    class BotStoppedSpeakingFrame(Frame):  # type: ignore[no-redef]
        pass

    class UserStartedSpeakingFrame(Frame):  # type: ignore[no-redef]
        pass

    class EndTaskFrame(Frame):  # type: ignore[no-redef]
        pass

    class TTSUpdateSettingsFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, settings: dict | None = None, delta=None):
            self.settings = settings or {}
            self.delta = delta

    class SarvamTTSSettings:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FrameDirection:  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"


if TYPE_CHECKING:
    from vaidya.pipeline.conversation import ConversationManager


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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._mgr = conversation_manager
        self._call_id = call_id
        self._language = language
        self._wake: asyncio.Event = asyncio.Event()
        self._idle_task: asyncio.Task | None = None
        self._language_locked: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Route frames: handle silence signals, auto-switch language, route transcriptions."""
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._start_idle_watch()
        elif isinstance(frame, UserStartedSpeakingFrame | TranscriptionFrame):
            self._cancel_idle_watch()

        if isinstance(frame, TranscriptionFrame):
            await self._on_transcription(frame)
            return

        # Pass through everything else unchanged
        await self.push_frame(frame, direction)

    async def _on_transcription(self, frame: TranscriptionFrame) -> None:
        """Detect language on the first utterance, then handle the turn."""
        user_text = (frame.text or "").strip()
        if not user_text:
            return

        logger.debug(
            "STT transcription received",
            extra={"call_id": self._call_id, "text_length": len(user_text)},
        )

        if not self._language_locked:
            # Only lock once we have a credible language signal. Short
            # mumbled utterances with no ``frame.language`` leave us
            # unlocked so the next (cleaner) transcription can pick.
            self._language_locked = await self._maybe_switch_language(
                await self._language_signal_for_turn(user_text, getattr(frame, "language", None))
            )

        try:
            response = await self._mgr.handle_turn(
                self._call_id,
                user_text,
                self._language,
                channel="voice",
            )
            await self._sync_language_from_context()
            await self.push_frame(TextFrame(text=response))
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(
                "Voice agent processing failed: %s",
                error_type,
                extra={"call_id": self._call_id, "error": str(exc)[:200]},
                exc_info=True,
            )
            fallback = get_msg("conversation", "error", self._language)
            await self.push_frame(TextFrame(text=fallback))

    async def _get_context(self):
        getter = getattr(self._mgr, "get_context", None)
        if getter is None:
            return None
        try:
            return await getter(self._call_id)
        except (AttributeError, TypeError):
            return None

    async def _language_signal_for_turn(self, user_text: str, stt_language: object) -> object:
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
        return stt_language

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

    # ------------------------------------------------------------------
    # Idle timer (silence watcher)
    # ------------------------------------------------------------------

    def _start_idle_watch(self) -> None:
        self._cancel_idle_watch()
        self._wake = asyncio.Event()
        self._idle_task = self.create_task(self._idle_loop())

    def _cancel_idle_watch(self) -> None:
        if self._idle_task is not None:
            self._wake.set()
            self._idle_task = None

    async def cleanup(self) -> None:
        """Stop the idle watcher when Pipecat tears down the processor."""
        task = self._idle_task
        if task is not None:
            self._wake.set()
            self._idle_task = None
            await self.cancel_task(task)
        await super().cleanup()

    async def _idle_loop(self) -> None:
        """Wait through SILENCE_STEPS thresholds, escalating on each timeout."""
        elapsed = 0.0
        for threshold, _key, terminal in SILENCE_STEPS:
            wait_for = threshold - elapsed
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=wait_for)
                return  # user spoke -> silence broken
            except TimeoutError:
                pass
            elapsed = threshold

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

            await self.push_frame(TextFrame(text=spoken))

            if is_terminal or terminal:
                # Give the TTS a moment to flush, then end the task.
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                return


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
