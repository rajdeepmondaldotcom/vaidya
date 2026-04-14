"""Text-to-Speech wrapper for Sarvam Bulbul v3.

Wraps :meth:`SarvamClient.tts` with language normalisation and speaker
selection.  Bulbul v3 supports 2,500 chars, 45 speakers, temperature
control for expressiveness, and pace control for speech speed.
"""

from __future__ import annotations

import logging

from vaidya.sarvam.client import SarvamClient
from vaidya.voice.language import TTS_SPEAKERS, normalize_language

logger = logging.getLogger(__name__)


class TTSClient:
    """Thin wrapper around :class:`SarvamClient` for text-to-speech."""

    def __init__(self, client: SarvamClient) -> None:
        self._client = client

    async def synthesize(
        self,
        text: str,
        language: str = "hi-IN",
        pace: float = 1.0,
        temperature: float = 0.6,
        speech_sample_rate: int = 8000,
    ) -> bytes | None:
        """Convert *text* to speech audio bytes.

        Parameters
        ----------
        text:
            The text to speak.  Truncated to 2500 chars by the underlying
            client (Bulbul v3 limit).
        language:
            BCP-47 language tag (e.g. ``"hi-IN"``).
        pace:
            Speech pace multiplier (default 1.0).
        speech_sample_rate:
            Audio sample rate in Hz (default 8000).

        Returns
        -------
        bytes | None
            Audio bytes on success, ``None`` on failure.
        """
        lang = normalize_language(language)
        speaker = TTS_SPEAKERS.get(lang, "anushka")

        try:
            audio = await self._client.tts(
                text,
                lang.value,
                speaker,
                temperature=temperature,
                pace=pace,
                speech_sample_rate=speech_sample_rate,
            )
            if audio:
                logger.debug(
                    "TTS synthesis succeeded",
                    extra={"language": lang.value, "text_length": len(text)},
                )
            return audio
        except Exception as exc:
            logger.error(
                "TTS synthesis failed",
                extra={"language": lang.value, "error": str(exc)},
            )
            return None
