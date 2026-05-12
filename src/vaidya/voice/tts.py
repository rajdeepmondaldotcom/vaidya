"""Text-to-Speech wrapper for Sarvam Bulbul v3.

Wraps :meth:`SarvamClient.tts` with language normalisation, speaker
selection, and automatic chunking for long text (>2500 chars).
Bulbul v3 supports 2,500 chars, 30+ speakers, temperature control
for expressiveness, and pace control for speech speed.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from vaidya.voice.language import TTS_SPEAKERS, normalize_language
from vaidya.voice.prosody import format_for_tts, pace_for_profile, temperature_for_profile

if TYPE_CHECKING:
    from vaidya.sarvam.client import SarvamClient

# Bulbul v3 character limit -- kept in sync with sarvam.models.TTS_MAX_CHARS_V3.
# Cannot import from sarvam.models directly because of a circular import chain:
# sarvam.models -> voice.language -> voice.__init__ -> voice.tts -> sarvam.models.
_TTS_MAX_CHARS: int = 2500

logger = logging.getLogger(__name__)

# Sentence boundary pattern for chunking (Hindi/English/Indic punctuation)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[।.?!])\s+")


def _chunk_text(text: str, max_chars: int = _TTS_MAX_CHARS) -> list[str]:
    """Split *text* into chunks of at most *max_chars* at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence[:max_chars] if len(sentence) > max_chars else sentence
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


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
        metadata: dict[str, Any] | None = None,
    ) -> bytes | None:
        """Convert *text* to speech audio bytes, chunking if needed."""
        profile = str(metadata.get("tts_profile", "default")) if metadata else "default"
        if metadata and "tts_speech_rate_factor" in metadata:
            pace = float(metadata["tts_speech_rate_factor"])
        elif metadata and "tts_profile" in metadata:
            pace = pace_for_profile(profile, pace)
        if metadata and "tts_profile" in metadata:
            temperature = temperature_for_profile(profile, temperature)

        lang = normalize_language(language)
        speaker = TTS_SPEAKERS.get(lang, "priya")
        text = format_for_tts(text, profile=profile)
        chunks = _chunk_text(text)

        if len(chunks) > 1:
            logger.info("TTS chunking: %d chars -> %d chunks", len(text), len(chunks))

        audio_parts = await self._synthesize_chunks(
            chunks,
            lang.value,
            speaker,
            temperature,
            pace,
            speech_sample_rate,
        )

        if not audio_parts:
            return None

        logger.debug(
            "TTS synthesis succeeded",
            extra={"language": lang.value, "text_length": len(text), "chunks": len(chunks)},
        )
        return b"".join(audio_parts)

    async def _synthesize_chunks(
        self,
        chunks: list[str],
        lang_code: str,
        speaker: str,
        temperature: float,
        pace: float,
        speech_sample_rate: int,
    ) -> list[bytes]:
        """Synthesize each chunk, skipping failures."""
        audio_parts: list[bytes] = []
        for chunk in chunks:
            try:
                audio = await self._client.tts(
                    chunk,
                    lang_code,
                    speaker,
                    temperature=temperature,
                    pace=pace,
                    speech_sample_rate=speech_sample_rate,
                )
                if audio:
                    audio_parts.append(audio)
            except Exception as exc:
                logger.error(
                    "TTS synthesis failed",
                    extra={"language": lang_code, "error": str(exc)},
                )
        return audio_parts
