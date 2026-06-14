"""Unit tests for SarvamClient and parse_llm_json."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vaidya.sarvam.client import (
    SarvamClient,
    _duration_from_stt_response,
    _duration_from_wav_bytes,
    _estimate_audio_duration_seconds,
    _extract_chat_content,
    _model_latency_class,
    _read_file_like_bytes,
    _retry_async,
    parse_llm_json,
)
from vaidya.sarvam.cost import CostTracker
from vaidya.sarvam.models import TTS_MAX_CHARS_V3
from vaidya.sarvam.resilience import CircuitOpenError, CircuitState


def _wav_bytes(duration_seconds: float = 0.25, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * sample_rate))
    return buffer.getvalue()


class _SyncCall:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _chat_response(content: str = "ok", tokens: int = 12, reasoning: str | None = None):
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


def _fake_sdk(**calls):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=calls.get("chat", _SyncCall(_chat_response()))),
        text=SimpleNamespace(
            translate=calls.get(
                "translate",
                _SyncCall(SimpleNamespace(translated_text="namaste")),
            ),
            identify_language=calls.get(
                "identify_language",
                _SyncCall(SimpleNamespace(language_code="hi-IN", script_code="Deva")),
            ),
            transliterate=calls.get(
                "transliterate",
                _SyncCall(SimpleNamespace(transliterated_text="namaste")),
            ),
        ),
        text_to_speech=SimpleNamespace(
            convert=calls.get("tts", _SyncCall(SimpleNamespace(audios=[b"audio"])))
        ),
        speech_to_text=SimpleNamespace(
            transcribe=calls.get(
                "stt",
                _SyncCall(
                    SimpleNamespace(
                        transcript="hello",
                        language_code="en-IN",
                        language_probability=0.91,
                        timestamps={"end_time_seconds": 2.5},
                    )
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# parse_llm_json
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_valid_json_string(self):
        result = parse_llm_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_wrapped_in_code_fence(self):
        raw = '```json\n{"matches": [{"id": "pmjay"}]}\n```'
        result = parse_llm_json(raw)
        assert result == {"matches": [{"id": "pmjay"}]}

    def test_json_in_plain_code_fence(self):
        raw = '```\n{"status": "ok"}\n```'
        result = parse_llm_json(raw)
        assert result == {"status": "ok"}

    def test_bare_top_level_array_is_wrapped_as_matches(self):
        """The reviewer emits a bare JSON array; it must parse, not error
        (a bare array was the cause of every reviewer batch failing)."""
        raw = '```json\n[{"scheme_id": "PMJAY", "verdict": "eligible"}]\n```'
        result = parse_llm_json(raw)
        assert not result.get("_parse_error")
        assert result["matches"] == [{"scheme_id": "PMJAY", "verdict": "eligible"}]

    def test_bare_array_without_fence(self):
        result = parse_llm_json('[{"a": 1}, {"b": 2}]')
        assert result["matches"] == [{"a": 1}, {"b": 2}]

    def test_json_embedded_in_text(self):
        raw = 'Here is the result: {"eligible": true} Hope this helps!'
        result = parse_llm_json(raw)
        assert result == {"eligible": True}

    def test_empty_input(self):
        result = parse_llm_json("")
        assert result["_parse_error"] is True

    def test_none_input(self):
        result = parse_llm_json(None)
        assert result["_parse_error"] is True

    def test_malformed_json(self):
        result = parse_llm_json("{invalid json here}")
        assert result["_parse_error"] is True
        assert "_raw" in result

    def test_nested_json_extraction(self):
        raw = 'Some preamble\n{"outer": {"inner": [1, 2, 3]}}\nMore text'
        result = parse_llm_json(raw)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_uses_last_balanced_json_object_when_reasoning_has_examples(self):
        raw = (
            'Example: {"state": null}\n'
            "I will now answer with the extracted fields.\n"
            '{"extracted_fields": {"state": "Rajasthan"}, "field_confidence": {"state": 0.9}}'
        )
        result = parse_llm_json(raw)
        assert result["extracted_fields"]["state"] == "Rajasthan"

    def test_pure_text_no_json(self):
        result = parse_llm_json("This is just plain text with no JSON at all.")
        assert result["_parse_error"] is True

    def test_whitespace_around_json(self):
        result = parse_llm_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_bare_array_is_wrapped_not_errored(self):
        # A bare top-level array is valid LLM output (the reviewer emits it)
        # and is wrapped as {"matches": [...]}, not treated as a parse error.
        result = parse_llm_json('[{"a": 1}]')
        assert not result.get("_parse_error")
        assert result["matches"] == [{"a": 1}]


# ---------------------------------------------------------------------------
# set_active_call_id / clear_active_call_id
# ---------------------------------------------------------------------------


class TestCallIdManagement:
    @patch("vaidya.sarvam.client.SarvamAI")
    def test_set_and_clear_call_id(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        assert client._active_call_id == ""

        client.set_active_call_id("call-abc")
        assert client._active_call_id == "call-abc"

        client.clear_active_call_id()
        assert client._active_call_id == ""

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_overwrite_call_id(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.set_active_call_id("call-1")
        client.set_active_call_id("call-2")
        assert client._active_call_id == "call-2"


# ---------------------------------------------------------------------------
# CostTracker integration
# ---------------------------------------------------------------------------


class TestCostTrackerIntegration:
    @patch("vaidya.sarvam.client.SarvamAI")
    def test_client_has_cost_tracker(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        assert isinstance(client.costs, CostTracker)

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_record_llm_free(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_llm(1000, call_id="test-call", model="sarvam-105b")
        assert client.costs.total_cost_inr == 0.0
        assert len(client.costs.entries) == 1

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_record_tts_costs(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_tts(10_000, call_id="test-call", model="bulbul:v3")
        assert client.costs.total_cost_inr == pytest.approx(30.0)

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_record_translate_costs(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_translate(10_000, call_id="test-call", model="mayura:v1")
        assert client.costs.total_cost_inr == pytest.approx(20.0)

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_record_stt_costs(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_stt(3600, call_id="test-call", model="saaras:v3")
        assert client.costs.total_cost_inr == pytest.approx(30.0)

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_cost_for_call_filtering(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_llm(500, call_id="call-A", model="sarvam-105b")
        client.costs.record_tts(1000, call_id="call-A", model="bulbul:v3")
        client.costs.record_tts(2000, call_id="call-B", model="bulbul:v3")

        cost_a = client.costs.cost_for_call("call-A")
        cost_b = client.costs.cost_for_call("call-B")
        assert cost_a == pytest.approx(1000 * 30.0 / 10_000)
        assert cost_b == pytest.approx(2000 * 30.0 / 10_000)

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_total_by_service(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.costs.record_tts(5000, call_id="c1")
        client.costs.record_translate(3000, call_id="c1")
        breakdown = client.costs.total_by_service
        assert "tts" in breakdown
        assert "translate" in breakdown


class TestChatJson:
    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_retries_at_low_reasoning_after_parse_error(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(side_effect=["thinking aloud", '{"ok": true}'])

        result = await client.chat_json(
            "sarvam-30b",
            [{"role": "user", "content": "Return JSON"}],
            reasoning_effort="medium",
        )

        assert result == {"ok": True}
        assert client.chat.await_count == 2
        assert client.chat.await_args_list[0].kwargs["reasoning_effort"] == "medium"
        # Retry pins explicit "low" — omitting the param triggers the
        # model's verbose default reasoning and content comes back None.
        assert client.chat.await_args_list[1].kwargs["reasoning_effort"] == "low"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_parse_error_still_retries_once_without_reasoning_effort(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(return_value="thinking aloud")

        result = await client.chat_json(
            "sarvam-30b",
            [{"role": "user", "content": "Return JSON"}],
            reasoning_effort=None,
        )

        assert result["_parse_error"] is True
        assert client.chat.await_count == 2
        assert client.chat.await_args_list[1].kwargs["reasoning_effort"] == "low"


class TestRetryAsyncTimingLog:
    """_retry_async emits an INFO timing log carrying label + elapsed + attempts."""

    async def test_logs_elapsed_and_attempt_count_on_success(self, caplog):
        with caplog.at_level("INFO", logger="vaidya.sarvam.client"):
            result = await _retry_async(
                lambda: "ok", retries=2, base_delay=0, timeout=1, call_label="sarvam chat"
            )

        assert result == "ok"
        messages = [r.getMessage() for r in caplog.records]
        # A single measurable "<label> done in <secs>s (1 attempt)" line.
        assert any("sarvam chat done in" in m and "1 attempt" in m for m in messages)

    async def test_logs_attempt_count_after_a_retry(self, caplog):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary")
            return "ok"

        with caplog.at_level("INFO", logger="vaidya.sarvam.client"):
            result = await _retry_async(
                flaky, retries=2, base_delay=0, timeout=1, call_label="sarvam chat"
            )

        assert result == "ok"
        messages = [r.getMessage() for r in caplog.records]
        assert any("sarvam chat done in" in m and "2 attempts" in m for m in messages)

    async def test_logs_timing_on_final_failure(self, caplog):
        def always_fails():
            raise RuntimeError("nope")

        with (
            caplog.at_level("INFO", logger="vaidya.sarvam.client"),
            pytest.raises(RuntimeError, match="nope"),
        ):
            await _retry_async(
                always_fails, retries=1, base_delay=0, timeout=1, call_label="sarvam tts"
            )

        messages = [r.getMessage() for r in caplog.records]
        # Even on failure the latency is measurable (2 attempts = 1 + 1 retry).
        assert any("sarvam tts failed after" in m and "2 attempts" in m for m in messages)


class TestPerCallTimeoutForwarding:
    """Public methods accept + forward an optional per-call timeout."""

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_chat_forwards_explicit_timeout_and_caps_retries(self, mock_sarvam_cls):
        # 3 attempts by default; a SHORT timeout (< client default) caps retries
        # at 1 so worst-case stays bounded (no 3x12s stall).
        client = SarvamClient(api_key="test-key-123", timeout=45.0, retry_max_attempts=3)
        with patch(
            "vaidya.sarvam.client._retry_async", new=AsyncMock(return_value=_chat_response("hi"))
        ) as retry:
            await client.chat("sarvam-30b", [{"role": "user", "content": "hi"}], timeout=12.0)

        assert retry.await_args.kwargs["timeout"] == 12.0
        assert retry.await_args.kwargs["retries"] == 1
        assert retry.await_args.kwargs["call_label"] == "sarvam chat"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_chat_default_timeout_uses_ceiling_and_full_retries(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123", timeout=45.0, retry_max_attempts=3)
        with patch(
            "vaidya.sarvam.client._retry_async", new=AsyncMock(return_value=_chat_response("hi"))
        ) as retry:
            await client.chat("sarvam-105b", [{"role": "user", "content": "hi"}])

        # timeout=None -> client default ceiling + full retry budget (3 - 1 = 2).
        assert retry.await_args.kwargs["timeout"] == 45.0
        assert retry.await_args.kwargs["retries"] == 2

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_chat_json_forwards_timeout_to_chat(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(return_value='{"ok": true}')

        await client.chat_json(
            "sarvam-30b", [{"role": "user", "content": "Return JSON"}], timeout=12.0
        )

        assert client.chat.await_args.kwargs["timeout"] == 12.0

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_chat_json_forwards_timeout_on_parse_error_retry(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(side_effect=["not json", '{"ok": true}'])

        result = await client.chat_json(
            "sarvam-30b", [{"role": "user", "content": "Return JSON"}], timeout=12.0
        )

        assert result == {"ok": True}
        # Both the first call and the low-reasoning retry carry the timeout.
        assert client.chat.await_args_list[0].kwargs["timeout"] == 12.0
        assert client.chat.await_args_list[1].kwargs["timeout"] == 12.0

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_translate_forwards_timeout(self, mock_sarvam_cls):
        mock_sarvam_cls.return_value = _fake_sdk()
        client = SarvamClient(api_key="test-key-123", timeout=45.0, retry_max_attempts=3)
        with patch(
            "vaidya.sarvam.client._retry_async",
            new=AsyncMock(return_value=SimpleNamespace(translated_text="namaste")),
        ) as retry:
            await client.translate("hello", "en-IN", "hi-IN", timeout=12.0)

        assert retry.await_args.kwargs["timeout"] == 12.0
        assert retry.await_args.kwargs["retries"] == 1
        assert retry.await_args.kwargs["call_label"] == "sarvam translate"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_tts_forwards_timeout(self, mock_sarvam_cls):
        mock_sarvam_cls.return_value = _fake_sdk()
        client = SarvamClient(api_key="test-key-123", timeout=45.0, retry_max_attempts=3)
        with patch(
            "vaidya.sarvam.client._retry_async",
            new=AsyncMock(return_value=SimpleNamespace(audios=[b"wav"])),
        ) as retry:
            await client.tts("hello", "hi-IN", timeout=12.0)

        assert retry.await_args.kwargs["timeout"] == 12.0
        assert retry.await_args.kwargs["retries"] == 1
        assert retry.await_args.kwargs["call_label"] == "sarvam tts"

    @patch("vaidya.sarvam.client.SarvamAI")
    def test_resolve_timeout_matrix(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123", timeout=45.0, retry_max_attempts=3)

        # None -> client default + full retries.
        assert client._resolve_timeout(None) == (45.0, 2)
        # Shorter than default -> capped to 1 retry.
        assert client._resolve_timeout(12.0) == (12.0, 1)
        # >= default -> keep full retry budget at the requested timeout.
        assert client._resolve_timeout(90.0) == (90.0, 2)


class TestSarvamHelpers:
    async def test_retry_async_retries_then_succeeds(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary")
            return "ok"

        result = await _retry_async(flaky, retries=1, base_delay=0, timeout=1)

        assert result == "ok"
        assert attempts["count"] == 2

    async def test_retry_async_raises_last_exception(self):
        def always_fails():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            await _retry_async(always_fails, retries=1, base_delay=0, timeout=1)

    def test_extract_chat_content_uses_reasoning_when_content_empty(self):
        response = _chat_response(content="", reasoning="reasoned answer")

        assert _extract_chat_content(response) == "reasoned answer"

    @pytest.mark.parametrize(
        ("model", "latency_class"),
        [
            ("sarvam-30b", "fast"),
            ("sarvam-105b", "regular"),
            ("custom", "unknown"),
        ],
    )
    def test_model_latency_class(self, model, latency_class):
        assert _model_latency_class(model) == latency_class

    def test_audio_duration_from_wav_bytes_and_path(self, tmp_path: Path):
        data = _wav_bytes(duration_seconds=0.5)
        wav_path = tmp_path / "sample.wav"
        wav_path.write_bytes(data)

        assert _duration_from_wav_bytes(data) == pytest.approx(0.5)
        assert _estimate_audio_duration_seconds(wav_path) == (pytest.approx(0.5), "wav_path")

    def test_audio_duration_raw_pcm_and_unreadable_inputs(self):
        pcm = b"\x00\x00" * 8000

        assert _estimate_audio_duration_seconds(pcm, raw_sample_rate=8000) == (
            pytest.approx(1.0),
            "raw_pcm_bytes",
        )
        assert _estimate_audio_duration_seconds(object()) == (None, "unknown")
        assert _read_file_like_bytes(object()) is None

    def test_read_file_like_preserves_position(self):
        stream = io.BytesIO(b"abcdef")
        stream.seek(2)

        assert _read_file_like_bytes(stream) == b"abcdef"
        assert stream.tell() == 2

    def test_duration_from_stt_response_checks_timestamps_and_diarized_entries(self):
        response = SimpleNamespace(
            timestamps={"timestamps": {"end_time_seconds": 4.0}},
            diarized_transcript={"entries": [{"end_time_seconds": 5.25}]},
        )

        assert _duration_from_stt_response(response) == pytest.approx(5.25)
        assert _duration_from_stt_response(SimpleNamespace()) is None


class TestTimedApiCall:
    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_success_records_cost_and_resets_breaker(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        recorded: list[float] = []

        result = await client._timed_api_call(
            "LLM call",
            lambda: _async_value(SimpleNamespace(value="raw")),
            lambda elapsed, response: recorded.append(response.value and elapsed),
            lambda response: response.value,
            {"model": "sarvam-30b"},
        )

        assert result == "raw"
        assert recorded and recorded[0] >= 0
        assert client._circuit_breakers.llm.state == CircuitState.CLOSED

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_error_can_return_none_and_open_circuit(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client._circuit_breakers.tts.failure_threshold = 1

        result = await client._timed_api_call(
            "TTS",
            lambda: _async_error(RuntimeError("tts failed")),
            lambda elapsed, response: None,
            lambda response: response,
            {},
            on_error="return_none",
        )

        assert result is None
        assert client._circuit_breakers.tts.state == CircuitState.OPEN

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_open_circuit_return_none(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client._circuit_breakers.tts.state = CircuitState.OPEN
        client._circuit_breakers.tts._last_failure_time = 10**12

        result = await client._timed_api_call(
            "TTS",
            lambda: _async_value("not-called"),
            lambda elapsed, response: None,
            lambda response: response,
            {},
            on_error="return_none",
        )

        assert result is None

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_open_circuit_raises_when_configured(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client._circuit_breakers.llm.state = CircuitState.OPEN
        client._circuit_breakers.llm._last_failure_time = 10**12

        with pytest.raises(CircuitOpenError):
            await client._timed_api_call(
                "LLM call",
                lambda: _async_value("not-called"),
                lambda elapsed, response: None,
                lambda response: response,
                {},
            )


async def _async_value(value):
    return value


async def _async_error(exc: Exception):
    raise exc


class TestSarvamClientMethods:
    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_chat_records_usage_and_forwards_optional_kwargs(self, mock_sarvam_cls):
        chat_call = _SyncCall(_chat_response("hello", tokens=250))
        mock_sarvam_cls.return_value = _fake_sdk(chat=chat_call)
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)
        client.set_active_call_id("call-1")

        result = await client.chat(
            "sarvam-30b",
            [{"role": "user", "content": "hi"}],
            reasoning_effort="low",
            wiki_grounding=True,
            top_p=0.8,
            frequency_penalty=0.1,
            seed=7,
            tools=[{"type": "function"}],
        )

        assert result == "hello"
        assert chat_call.calls[0]["reasoning_effort"] == "low"
        assert chat_call.calls[0]["wiki_grounding"] is True
        entry = client.costs.entries[-1]
        assert entry.service == "llm"
        assert entry.call_id == "call-1"
        assert entry.metadata["latency_class"] == "fast"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_translate_short_circuits_same_language(self, mock_sarvam_cls):
        translate_call = _SyncCall(SimpleNamespace(translated_text="unused"))
        mock_sarvam_cls.return_value = _fake_sdk(translate=translate_call)
        client = SarvamClient(api_key="test-key-123")

        assert await client.translate("hello", "en-IN", "en-IN") == "hello"
        assert translate_call.calls == []

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_translate_records_cost_and_metadata(self, mock_sarvam_cls):
        translate_call = _SyncCall(SimpleNamespace(translated_text="namaste"))
        mock_sarvam_cls.return_value = _fake_sdk(translate=translate_call)
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)

        result = await client.translate(
            "hello",
            "en-IN",
            "hi-IN",
            speaker_gender="Female",
            output_script="roman",
        )

        assert result == "namaste"
        assert translate_call.calls[0]["source_language_code"] == "en-IN"
        assert client.costs.entries[-1].metadata["speaker_gender"] == "Female"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_tts_truncates_records_cost_and_returns_first_audio(self, mock_sarvam_cls):
        tts_call = _SyncCall(SimpleNamespace(audios=[b"wav"]))
        mock_sarvam_cls.return_value = _fake_sdk(tts=tts_call)
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)
        text = "x" * (TTS_MAX_CHARS_V3 + 1)

        audio = await client.tts(text, "hi-IN", speaker="anushka")

        assert audio == b"wav"
        assert tts_call.calls[0]["text"] == text[:TTS_MAX_CHARS_V3]
        assert client.costs.entries[-1].metadata["truncated"] is True

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_tts_failure_returns_none(self, mock_sarvam_cls):
        mock_sarvam_cls.return_value = _fake_sdk(tts=_SyncCall(RuntimeError("down")))
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)

        assert await client.tts("hello", "hi-IN") is None

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_stt_prefers_response_timestamp_for_cost_duration(self, mock_sarvam_cls):
        stt_call = _SyncCall(
            SimpleNamespace(
                transcript="hello",
                language_code="en-IN",
                language_probability=0.9,
                timestamps={"end_time_seconds": 3.0},
            )
        )
        mock_sarvam_cls.return_value = _fake_sdk(stt=stt_call)
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)

        result = await client.stt(
            _wav_bytes(duration_seconds=1.0),
            language="en-IN",
            with_timestamps=True,
            with_diarization=True,
            num_speakers=2,
        )

        assert result == ("hello", "en-IN", 0.9)
        assert stt_call.calls[0]["with_timestamps"] is True
        entry = client.costs.entries[-1]
        assert entry.units == pytest.approx(3.0)
        assert entry.metadata["duration_source"] == "response_timestamps"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_identify_language_records_cost(self, mock_sarvam_cls):
        mock_sarvam_cls.return_value = _fake_sdk(
            identify_language=_SyncCall(SimpleNamespace(language_code="bn-IN"))
        )
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)

        assert await client.identify_language("ami bhalo") == ("bn-IN", "")
        assert client.costs.entries[-1].service == "language_id"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_transliterate_includes_spoken_form_options(self, mock_sarvam_cls):
        transliterate_call = _SyncCall(SimpleNamespace(transliterated_text="namaste"))
        mock_sarvam_cls.return_value = _fake_sdk(transliterate=transliterate_call)
        client = SarvamClient(api_key="test-key-123", retry_max_attempts=1)

        result = await client.transliterate(
            "नमस्ते",
            "hi-IN",
            "en-IN",
            spoken_form=True,
            spoken_form_numerals_language="hindi",
        )

        assert result == "namaste"
        assert transliterate_call.calls[0]["spoken_form"] is True
        assert client.costs.entries[-1].metadata["spoken_form"] is True


def _install_mock_pool(
    client: SarvamClient,
    *,
    head: MagicMock | None = None,
    base: str = "https://api.sarvam.ai",
) -> MagicMock:
    """Wire a mock pooled httpx.Client (and base URL) onto a SarvamClient.

    The SarvamAI SDK is patched in these tests, so the deep attribute path the
    client reaches through is itself a MagicMock; we plant a spec'd httpx.Client
    so ``isinstance(..., httpx.Client)`` in ``_pooled_httpx_client`` passes.
    Returns the mock used as ``.head`` so tests can assert call counts.
    """
    head = head or MagicMock(return_value=MagicMock(status_code=200))
    fake_raw = MagicMock(spec=httpx.Client)
    fake_raw.head = head
    client._client._client_wrapper.httpx_client.httpx_client = fake_raw
    client._client._client_wrapper.get_environment.return_value.base = base
    return head


class TestConnectionPooling:
    """The SDK reuses ONE pooled httpx client, and we tune it conservatively."""

    def test_one_sarvamai_instance_constructed_with_tuned_httpx_client(self):
        # A real (un-patched) client builds a real SarvamAI + tuned httpx pool.
        client = SarvamClient(api_key="test-key-123", timeout=30.0)
        try:
            raw = client._pooled_httpx_client()
            assert isinstance(raw, httpx.Client)
            pool = raw._transport._pool  # type: ignore[attr-defined]
            # Conservative tuning: keepalive lengthened, limits at httpx defaults.
            assert pool._keepalive_expiry == 90.0
            assert pool._max_keepalive_connections == 20
            assert pool._max_connections == 100
            # Read timeout mirrors the configured SarvamClient timeout.
            assert raw.timeout.read == 30.0
        finally:
            raw = client._pooled_httpx_client()
            if raw is not None:
                raw.close()

    def test_httpx_client_passed_into_sdk_constructor(self):
        # The tuned client must actually be handed to the SDK (not discarded).
        with patch("vaidya.sarvam.client.SarvamAI") as mock_cls:
            SarvamClient(api_key="test-key-123", timeout=15.0)
        kwargs = mock_cls.call_args.kwargs
        assert "httpx_client" in kwargs
        passed = kwargs["httpx_client"]
        assert isinstance(passed, httpx.Client)
        pool = passed._transport._pool  # type: ignore[attr-defined]
        assert pool._keepalive_expiry == 90.0
        passed.close()

    @pytest.mark.parametrize("bad_timeout", [0.0, -5.0, None, "not-a-number"])
    def test_build_httpx_client_falls_back_to_default_timeout(self, bad_timeout):
        # Non-positive, None, or non-numeric timeouts fall back to the SDK's
        # 60s default read timeout instead of crashing client construction.
        raw = SarvamClient._build_httpx_client(bad_timeout)  # type: ignore[arg-type]
        try:
            assert raw.timeout.read == 60.0
        finally:
            raw.close()


class TestPrewarm:
    """prewarm() is idempotent, concurrency-safe, and never raises."""

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_opens_connection_and_returns_true(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        head = _install_mock_pool(client)

        assert await client.prewarm() is True
        assert client._prewarmed is True
        assert head.call_count == 1
        # The warm-up targets the configured base URL.
        assert head.call_args.args[0] == "https://api.sarvam.ai"

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_is_idempotent(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        head = _install_mock_pool(client)

        first = await client.prewarm()
        second = await client.prewarm()
        third = await client.prewarm()

        assert (first, second, third) == (True, True, True)
        # Repeat calls collapse to a single network round-trip.
        assert head.call_count == 1

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_concurrent_prewarms_warm_once(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")

        def slow_head(_url):
            # Force overlap so all coroutines contend on the lock together.
            time.sleep(0.02)
            return MagicMock(status_code=200)

        head = MagicMock(side_effect=slow_head)
        _install_mock_pool(client, head=head)

        results = await asyncio.gather(*[client.prewarm() for _ in range(5)])

        assert all(results)
        assert head.call_count == 1

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_swallows_failure_and_returns_false(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        boom = MagicMock(side_effect=httpx.ConnectError("no network"))
        _install_mock_pool(client, head=boom)

        # Must NEVER raise — startup safety.
        result = await client.prewarm()

        assert result is False
        # Not marked warmed, so a later call may retry the handshake.
        assert client._prewarmed is False

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_swallows_timeout(self, mock_sarvam_cls):
        # A tiny timeout makes asyncio.wait_for fire before the (mocked) head
        # could ever return, exercising the TimeoutError branch.
        client = SarvamClient(api_key="test-key-123", timeout=0.01)

        def hang(_url):
            time.sleep(0.5)
            return MagicMock(status_code=200)

        _install_mock_pool(client, head=MagicMock(side_effect=hang))

        # asyncio.wait_for must bound the hang and prewarm swallows the timeout.
        result = await client.prewarm()

        assert result is False
        assert client._prewarmed is False

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_skips_when_pool_unavailable(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        # Simulate the SDK internal layout shifting: not an httpx.Client.
        client._client._client_wrapper.httpx_client.httpx_client = object()
        client._client._client_wrapper.get_environment.return_value.base = "https://api.sarvam.ai"

        result = await client.prewarm()

        assert result is False
        assert client._prewarmed is False

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_prewarm_does_not_touch_circuit_breaker_or_costs(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        _install_mock_pool(client)

        await client.prewarm()

        # Warm-up is not a billable API call and must not disturb resilience state.
        assert client._circuit_breakers.llm.state == CircuitState.CLOSED
        assert client.costs.entries == []


class _AsyncWs:
    def __init__(self, response):
        self.response = response
        self.transcribed: dict | None = None
        self.sent: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def transcribe(self, **kwargs):
        self.transcribed = kwargs

    async def send(self, **kwargs):
        self.sent = kwargs

    async def recv(self):
        return self.response


class _AsyncSarvam:
    def __init__(self, stt_ws: _AsyncWs, tts_ws: _AsyncWs):
        self.stt_ws = stt_ws
        self.tts_ws = tts_ws
        self.stt_kwargs: dict = {}
        self.tts_kwargs: dict = {}
        self.speech_to_text_streaming = SimpleNamespace(connect=self._connect_stt)
        self.text_to_speech_streaming = SimpleNamespace(connect=self._connect_tts)

    def _connect_stt(self, **kwargs):
        self.stt_kwargs = kwargs
        return self.stt_ws

    def _connect_tts(self, **kwargs):
        self.tts_kwargs = kwargs
        return self.tts_ws
