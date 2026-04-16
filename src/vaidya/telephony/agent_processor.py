"""Custom Pipecat processor that bridges STT output to Vaidya's ConversationManager.

Instead of routing transcribed speech through a generic LLM, this processor
feeds it into Vaidya's multi-agent orchestrator and pushes the response text
downstream to TTS.

Requires ``pipecat-ai`` — guarded import so the rest of the app works without it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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

    class Frame:  # type: ignore[no-redef]
        pass

    class TextFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = ""):
            self.text = text

    class TranscriptionFrame(Frame):  # type: ignore[no-redef]
        def __init__(self, text: str = ""):
            self.text = text

    class FrameDirection:  # type: ignore[no-redef]
        DOWNSTREAM = "downstream"


if TYPE_CHECKING:
    from vaidya.pipeline.conversation import ConversationManager

_FALLBACK_MESSAGES = {
    "hi-IN": "Maaf kijiye, kuch galat ho gaya. Kripya dobara boliye.",
    "ta-IN": "Manaikkavum, oru pizhvu undayatu. Thayavu seithu meeNum sollungal.",
    "bn-IN": "Dukkhito, kichu bhul hoyeche. Onugrah kore abar bolun.",
    "en-IN": "Sorry, something went wrong. Please try again.",
}


class VaidyaAgentProcessor(FrameProcessor):
    """Bridges Pipecat STT transcription to Vaidya's multi-agent pipeline.

    When a :class:`TranscriptionFrame` arrives (i.e., the user finished
    speaking), the processor:

    1. Extracts the transcribed text.
    2. Calls :meth:`ConversationManager.handle_turn` to route through
       the orchestrator state machine.
    3. Pushes the agent's text response downstream as a :class:`TextFrame`
       so the TTS service can synthesize it.

    Non-transcription frames are passed through unchanged.
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Route transcription frames through Vaidya; pass others through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            if not user_text or not user_text.strip():
                return

            logger.debug(
                "STT transcription received",
                extra={"call_id": self._call_id, "text_length": len(user_text)},
            )

            try:
                response = await self._mgr.handle_turn(
                    self._call_id,
                    user_text,
                    self._language,
                )
                # Push agent response downstream to TTS
                await self.push_frame(TextFrame(text=response))
            except Exception as exc:
                error_type = type(exc).__name__
                logger.error(
                    "Voice agent processing failed: %s",
                    error_type,
                    extra={"call_id": self._call_id, "error": str(exc)[:200]},
                    exc_info=True,
                )
                fallback = _FALLBACK_MESSAGES.get(self._language, _FALLBACK_MESSAGES["en-IN"])
                await self.push_frame(TextFrame(text=fallback))
        else:
            # Pass through all non-transcription frames unchanged
            await self.push_frame(frame, direction)
