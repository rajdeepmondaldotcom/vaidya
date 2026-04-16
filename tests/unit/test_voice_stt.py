"""Tests for STTClient: transcription delegation, return values, error handling.

Covers:
- transcribe delegates to SarvamClient.stt with correct model/mode/language
- transcribe returns (transcript, detected_language, confidence) tuple
- Error propagation when SarvamClient.stt raises
- Various language codes are passed through correctly
- transcribe_codemix convenience method
- is_available static method
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.sarvam.models import STT_MODEL
from vaidya.voice.stt import STTClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(stt_return: tuple[str, str, float] | None = None) -> tuple[STTClient, MagicMock]:
    """Return (STTClient, mock_sarvam) with a canned stt() response."""
    mock_sarvam = MagicMock()
    if stt_return is None:
        stt_return = ("namaste main rajasthan se hoon", "hi-IN", 0.95)
    mock_sarvam.stt = AsyncMock(return_value=stt_return)
    return STTClient(mock_sarvam), mock_sarvam


# ---------------------------------------------------------------------------
# TestSTTTranscribe
# ---------------------------------------------------------------------------


class TestSTTTranscribe:
    """Core transcription tests."""

    @pytest.mark.asyncio
    async def test_delegates_to_sarvam_stt(self) -> None:
        """transcribe calls SarvamClient.stt with model, mode, language."""
        client, mock = _make_client()
        audio = b"\x00\x01\x02"

        await client.transcribe(audio, language="hi-IN", mode="transcribe")

        mock.stt.assert_awaited_once_with(
            audio_file=audio,
            model=STT_MODEL,
            mode="transcribe",
            language="hi-IN",
        )

    @pytest.mark.asyncio
    async def test_returns_transcript_tuple(self) -> None:
        """transcribe returns (transcript, detected_lang, confidence)."""
        client, _ = _make_client(("hello world", "en-IN", 0.88))

        result = await client.transcribe(b"\x00")

        assert result == ("hello world", "en-IN", 0.88)

    @pytest.mark.asyncio
    async def test_return_types(self) -> None:
        """Each element of the return tuple has the correct type."""
        client, _ = _make_client(("text", "hi-IN", 0.75))

        transcript, lang, conf = await client.transcribe(b"\x00")

        assert isinstance(transcript, str)
        assert isinstance(lang, str)
        assert isinstance(conf, float)

    @pytest.mark.asyncio
    async def test_error_propagation(self) -> None:
        """When SarvamClient.stt raises, the error propagates to the caller."""
        mock_sarvam = MagicMock()
        mock_sarvam.stt = AsyncMock(side_effect=RuntimeError("STT service unavailable"))
        client = STTClient(mock_sarvam)

        with pytest.raises(RuntimeError, match="STT service unavailable"):
            await client.transcribe(b"\x00")

    @pytest.mark.asyncio
    async def test_language_hindi_passthrough(self) -> None:
        """Language 'hi-IN' is passed through to SarvamClient.stt."""
        client, mock = _make_client()

        await client.transcribe(b"\x00", language="hi-IN")

        assert mock.stt.call_args.kwargs["language"] == "hi-IN"

    @pytest.mark.asyncio
    async def test_language_tamil_passthrough(self) -> None:
        """Language 'ta-IN' is passed through to SarvamClient.stt."""
        client, mock = _make_client()

        await client.transcribe(b"\x00", language="ta-IN")

        assert mock.stt.call_args.kwargs["language"] == "ta-IN"

    @pytest.mark.asyncio
    async def test_language_bengali_passthrough(self) -> None:
        """Language 'bn-IN' is passed through to SarvamClient.stt."""
        client, mock = _make_client()

        await client.transcribe(b"\x00", language="bn-IN")

        assert mock.stt.call_args.kwargs["language"] == "bn-IN"

    @pytest.mark.asyncio
    async def test_default_language_is_hindi(self) -> None:
        """When no language is specified, defaults to 'hi-IN'."""
        client, mock = _make_client()

        await client.transcribe(b"\x00")

        assert mock.stt.call_args.kwargs["language"] == "hi-IN"


# ---------------------------------------------------------------------------
# TestSTTCodemix
# ---------------------------------------------------------------------------


class TestSTTCodemix:
    """transcribe_codemix convenience method."""

    @pytest.mark.asyncio
    async def test_codemix_uses_codemix_mode(self) -> None:
        """transcribe_codemix delegates with mode='codemix'."""
        client, mock = _make_client()

        await client.transcribe_codemix(b"\x00", language="hi-IN")

        mock.stt.assert_awaited_once_with(
            audio_file=b"\x00",
            model=STT_MODEL,
            mode="codemix",
            language="hi-IN",
        )


# ---------------------------------------------------------------------------
# TestSTTAvailability
# ---------------------------------------------------------------------------


class TestSTTAvailability:
    """Static helper method."""

    def test_is_available_returns_true(self) -> None:
        assert STTClient.is_available() is True
