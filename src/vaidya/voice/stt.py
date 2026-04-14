"""Speech-to-Text wrapper for Sarvam Saaras v3.

Wraps :meth:`SarvamClient.stt` with language hints and mode selection.
Saaras v3 supports 23 Indian languages, multiple modes (transcribe,
translate, verbatim, translit, codemix), and returns transcripts with
detected language metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import STT_MODEL

logger = logging.getLogger(__name__)


class STTClient:
    """Thin wrapper around :class:`SarvamClient` for speech-to-text."""

    def __init__(self, client: SarvamClient) -> None:
        self._client = client

    async def transcribe(
        self,
        audio_file: Any,
        language: str = "hi-IN",
        mode: str = "transcribe",
    ) -> tuple[str, str, float]:
        """Transcribe audio using Saaras v3.

        Parameters
        ----------
        audio_file:
            Audio file object accepted by the Sarvam SDK.
        language:
            BCP-47 language hint (e.g. ``"hi-IN"``).
        mode:
            Saaras mode: ``transcribe``, ``translate``, ``verbatim``,
            ``translit``, or ``codemix``.

        Returns
        -------
        tuple[str, str, float]
            ``(transcript, detected_language, confidence)``
        """
        # SarvamClient.stt() returns (transcript, language_code, probability)
        transcript, detected_lang, confidence = await self._client.stt(
            audio_file=audio_file,
            model=STT_MODEL,
            mode=mode,
            language=language,
        )
        logger.debug(
            "STT transcription completed",
            extra={
                "detected_language": detected_lang,
                "confidence": f"{confidence:.2f}",
                "mode": mode,
                "transcript_length": len(transcript),
            },
        )
        return transcript, detected_lang, confidence

    async def transcribe_codemix(
        self,
        audio_file: Any,
        language: str = "hi-IN",
    ) -> tuple[str, str, float]:
        """Transcribe with code-mixing support."""
        return await self.transcribe(audio_file, language, mode="codemix")

    @staticmethod
    def is_available() -> bool:
        """Return whether STT is available."""
        return True
