"""Unit tests for Pipecat voice pipeline construction helpers."""

from __future__ import annotations

import pytest

from vaidya.telephony.pipeline import (
    _TELEPHONY_SAMPLE_RATE,
    PIPECAT_AVAILABLE,
    _build_stt_service,
    _build_tts_service,
    _build_twilio_serializer,
    _build_websocket_params,
)

pytestmark = pytest.mark.skipif(not PIPECAT_AVAILABLE, reason="pipecat-ai is not installed")


class TestVoicePipelineHelpers:
    def test_websocket_params_do_not_include_unsupported_vad_analyzer(self):
        serializer = _build_twilio_serializer(
            stream_sid="MZ123",
            twilio_call_sid="CA123",
            twilio_account_sid="",
            twilio_auth_token="",
        )
        params = _build_websocket_params(serializer)

        assert not hasattr(params, "vad_analyzer")
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True
        assert params.audio_in_sample_rate == _TELEPHONY_SAMPLE_RATE
        assert params.audio_out_sample_rate == _TELEPHONY_SAMPLE_RATE

    def test_stt_service_uses_sarvam_vad_signals_and_telephony_audio(self):
        stt = _build_stt_service("test-key")

        assert stt._init_sample_rate == _TELEPHONY_SAMPLE_RATE
        assert stt._input_audio_codec == "pcm_s16le"
        assert stt._mode == "codemix"
        assert stt._settings.model == "saaras:v3"
        assert stt._settings.language is None
        assert stt._settings.vad_signals is True
        assert stt._settings.high_vad_sensitivity is True
        assert stt._settings.interrupt_min_speech_frames == 3

    def test_tts_service_uses_human_pacing_for_telephony(self):
        tts = _build_tts_service("test-key", "priya", "hi-IN")

        assert tts._init_sample_rate == _TELEPHONY_SAMPLE_RATE
        assert tts._settings.model == "bulbul:v3"
        assert tts._settings.voice == "priya"
        assert tts._settings.language == "hi-IN"
        assert tts._settings.pace == 0.94
        assert tts._settings.temperature == 0.55
        assert tts._settings.min_buffer_size == 35
        assert tts._settings.max_chunk_length == 130

    def test_twilio_serializer_disables_auto_hangup_without_credentials(self):
        serializer = _build_twilio_serializer(
            stream_sid="MZ123",
            twilio_call_sid="CA123",
            twilio_account_sid="",
            twilio_auth_token="",
        )

        assert serializer._params.auto_hang_up is False
        assert serializer._params.sample_rate == _TELEPHONY_SAMPLE_RATE
