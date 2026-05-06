"""Unit tests for SarvamClient and parse_llm_json."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vaidya.sarvam.client import SarvamClient, parse_llm_json
from vaidya.sarvam.cost import CostTracker

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

    def test_pure_text_no_json(self):
        result = parse_llm_json("This is just plain text with no JSON at all.")
        assert result["_parse_error"] is True

    def test_whitespace_around_json(self):
        result = parse_llm_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_array_json_not_extracted_as_object(self):
        result = parse_llm_json('[{"a": 1}]')
        # Arrays don't start with { so the fallback won't find them
        # but json.loads on the full string should work
        assert isinstance(result, list)
        assert result[0]["a"] == 1


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
    async def test_retries_without_reasoning_effort_after_parse_error(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(side_effect=["thinking aloud", '{"ok": true}'])

        result = await client.chat_json(
            "sarvam-30b",
            [{"role": "user", "content": "Return JSON"}],
            reasoning_effort="low",
        )

        assert result == {"ok": True}
        assert client.chat.await_count == 2
        assert client.chat.await_args_list[0].kwargs["reasoning_effort"] == "low"
        assert client.chat.await_args_list[1].kwargs["reasoning_effort"] is None

    @patch("vaidya.sarvam.client.SarvamAI")
    async def test_does_not_retry_parse_error_without_reasoning_effort(self, mock_sarvam_cls):
        client = SarvamClient(api_key="test-key-123")
        client.chat = AsyncMock(return_value="thinking aloud")

        result = await client.chat_json(
            "sarvam-30b",
            [{"role": "user", "content": "Return JSON"}],
            reasoning_effort=None,
        )

        assert result["_parse_error"] is True
        client.chat.assert_awaited_once()
