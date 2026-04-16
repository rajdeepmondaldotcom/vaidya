"""Tests for TTSClient: speaker selection, synthesis, error handling.

Covers:
- synthesize with each supported language maps to correct speaker via TTS_SPEAKERS
- synthesize calls SarvamClient.tts with correct params
- synthesize returns bytes on success
- synthesize returns None when SarvamClient.tts raises
- synthesize returns None when client returns None
- Default language fallback behaviour
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.voice.language import TTS_SPEAKERS, Language
from vaidya.voice.tts import TTSClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    tts_return: bytes | None = b"\xff\xfe\x00\x01",
    tts_side_effect: Exception | None = None,
) -> tuple[TTSClient, MagicMock]:
    """Return (TTSClient, mock_sarvam) with a canned tts() response."""
    mock_sarvam = MagicMock()
    if tts_side_effect:
        mock_sarvam.tts = AsyncMock(side_effect=tts_side_effect)
    else:
        mock_sarvam.tts = AsyncMock(return_value=tts_return)
    return TTSClient(mock_sarvam), mock_sarvam


# ---------------------------------------------------------------------------
# TestTTSSpeakerSelection
# ---------------------------------------------------------------------------


class TestTTSSpeakerSelection:
    """Ensure each language maps to the correct speaker from TTS_SPEAKERS."""

    @pytest.mark.asyncio
    async def test_hindi_uses_priya(self) -> None:
        client, mock = _make_client()

        await client.synthesize("namaste", language="hi-IN")

        assert mock.tts.call_args.args[2] == "priya"

    @pytest.mark.asyncio
    async def test_tamil_uses_kavitha(self) -> None:
        client, mock = _make_client()

        await client.synthesize("vanakkam", language="ta-IN")

        assert mock.tts.call_args.args[2] == "kavitha"

    @pytest.mark.asyncio
    async def test_bengali_uses_rupali(self) -> None:
        client, mock = _make_client()

        await client.synthesize("namaskar", language="bn-IN")

        assert mock.tts.call_args.args[2] == "rupali"

    @pytest.mark.asyncio
    async def test_english_uses_amelia(self) -> None:
        client, mock = _make_client()

        await client.synthesize("hello", language="en-IN")

        assert mock.tts.call_args.args[2] == "amelia"

    @pytest.mark.asyncio
    async def test_all_tts_speakers_covered(self) -> None:
        """Every Language in TTS_SPEAKERS is testable and maps to a non-empty speaker."""
        for lang, speaker in TTS_SPEAKERS.items():
            assert isinstance(lang, Language)
            assert isinstance(speaker, str)
            assert len(speaker) > 0


# ---------------------------------------------------------------------------
# TestTTSSynthesizeParams
# ---------------------------------------------------------------------------


class TestTTSSynthesizeParams:
    """Verify SarvamClient.tts is called with the correct positional and keyword args."""

    @pytest.mark.asyncio
    async def test_passes_normalized_language_value(self) -> None:
        """The language argument to SarvamClient.tts should be Language.value (e.g. 'hi-IN')."""
        client, mock = _make_client()

        await client.synthesize("hello", language="hi")

        # normalize_language("hi") -> Language.HINDI -> "hi-IN"
        assert mock.tts.call_args.args[1] == Language.HINDI.value

    @pytest.mark.asyncio
    async def test_passes_text_as_first_arg(self) -> None:
        client, mock = _make_client()

        await client.synthesize("test text", language="hi-IN")

        assert mock.tts.call_args.args[0] == "test text"

    @pytest.mark.asyncio
    async def test_passes_pace(self) -> None:
        client, mock = _make_client()

        await client.synthesize("text", language="hi-IN", pace=1.5)

        assert mock.tts.call_args.kwargs["pace"] == 1.5

    @pytest.mark.asyncio
    async def test_passes_temperature(self) -> None:
        client, mock = _make_client()

        await client.synthesize("text", language="hi-IN", temperature=0.8)

        assert mock.tts.call_args.kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_passes_sample_rate(self) -> None:
        client, mock = _make_client()

        await client.synthesize("text", language="hi-IN", speech_sample_rate=16000)

        assert mock.tts.call_args.kwargs["speech_sample_rate"] == 16000


# ---------------------------------------------------------------------------
# TestTTSSynthesizeReturnValues
# ---------------------------------------------------------------------------


class TestTTSSynthesizeReturnValues:
    """Return value behaviour: bytes on success, None on error."""

    @pytest.mark.asyncio
    async def test_returns_bytes_on_success(self) -> None:
        audio_data = b"\xff\xfe\x00\x01"
        client, _ = _make_client(tts_return=audio_data)

        result = await client.synthesize("hello", language="hi-IN")

        assert result == audio_data
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_returns_none_when_client_raises(self) -> None:
        client, _ = _make_client(tts_side_effect=RuntimeError("TTS service down"))

        result = await client.synthesize("hello", language="hi-IN")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_client_returns_none(self) -> None:
        client, _ = _make_client(tts_return=None)

        result = await client.synthesize("hello", language="hi-IN")

        assert result is None


# ---------------------------------------------------------------------------
# TestTTSDefaultLanguage
# ---------------------------------------------------------------------------


class TestTTSDefaultLanguage:
    """Default language fallback when parameter is omitted or unrecognised."""

    @pytest.mark.asyncio
    async def test_default_language_is_hindi(self) -> None:
        """When no language is specified, defaults to 'hi-IN'."""
        client, mock = _make_client()

        await client.synthesize("namaste")

        # Default language="hi-IN" -> normalize_language("hi-IN") -> Language.HINDI
        assert mock.tts.call_args.args[1] == Language.HINDI.value

    @pytest.mark.asyncio
    async def test_unknown_language_falls_back_to_hindi(self) -> None:
        """Unrecognised language code falls back to Hindi via normalize_language."""
        client, mock = _make_client()

        await client.synthesize("hola", language="es-ES")

        assert mock.tts.call_args.args[1] == Language.HINDI.value
        # Speaker should be "priya" (Hindi default from TTS_SPEAKERS fallback)
        assert mock.tts.call_args.args[2] == "priya"
