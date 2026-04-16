"""Tests for CostTracker: per-service cost computation, aggregation, breakdowns.

Covers:
- Each record_* method (llm, stt, tts, translate, transliterate, language_id, vision)
  computes cost correctly per pricing table
- total_cost_inr aggregates correctly
- total_by_service groups by service
- cost_for_call filters by call_id
- breakdown_for_call returns correct per-service structure
- summary returns expected keys
- LLM cost is 0 (free)
"""

from __future__ import annotations

import pytest

from vaidya.sarvam.cost import CostTracker

# ---------------------------------------------------------------------------
# TestRecordMethods — individual service cost computation
# ---------------------------------------------------------------------------


class TestRecordLlm:
    """LLM is free: cost should always be 0."""

    def test_llm_cost_is_zero(self) -> None:
        tracker = CostTracker()

        tracker.record_llm(1000, call_id="c1")

        assert tracker.total_cost_inr == 0.0

    def test_llm_many_tokens_still_zero(self) -> None:
        tracker = CostTracker()

        tracker.record_llm(1_000_000, call_id="c1")

        assert tracker.total_cost_inr == 0.0

    def test_llm_records_entry(self) -> None:
        tracker = CostTracker()

        tracker.record_llm(500, call_id="c1", latency_ms=120.0, model="sarvam-105b")

        assert len(tracker.entries) == 1
        entry = tracker.entries[0]
        assert entry.service == "llm"
        assert entry.units == 500
        assert entry.call_id == "c1"
        assert entry.model == "sarvam-105b"


class TestRecordStt:
    """STT: Rs 30/hour = Rs 30/3600 per second."""

    def test_stt_one_hour(self) -> None:
        tracker = CostTracker()

        tracker.record_stt(3600.0, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(30.0)

    def test_stt_one_second(self) -> None:
        tracker = CostTracker()

        tracker.record_stt(1.0, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(30.0 / 3600)


class TestRecordTts:
    """TTS: Rs 30/10K chars."""

    def test_tts_10k_chars(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(30.0)

    def test_tts_1_char(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(1, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(30.0 / 10_000)


class TestRecordTranslate:
    """Translation: Rs 20/10K chars."""

    def test_translate_10k_chars(self) -> None:
        tracker = CostTracker()

        tracker.record_translate(10_000, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(20.0)


class TestRecordTransliterate:
    """Transliteration: Rs 20/10K chars."""

    def test_transliterate_10k_chars(self) -> None:
        tracker = CostTracker()

        tracker.record_transliterate(10_000, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(20.0)


class TestRecordLanguageId:
    """Language ID: Rs 3.5/10K chars."""

    def test_language_id_10k_chars(self) -> None:
        tracker = CostTracker()

        tracker.record_language_id(10_000, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(3.5)


class TestRecordVision:
    """Vision: Rs 1.5/page."""

    def test_vision_one_page(self) -> None:
        tracker = CostTracker()

        tracker.record_vision(pages=1, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(1.5)

    def test_vision_five_pages(self) -> None:
        tracker = CostTracker()

        tracker.record_vision(pages=5, call_id="c1")

        assert tracker.total_cost_inr == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# TestAggregation
# ---------------------------------------------------------------------------


class TestTotalCostAggregation:
    """total_cost_inr sums across all entries."""

    def test_aggregates_multiple_services(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="c1")  # 30.0
        tracker.record_translate(10_000, call_id="c1")  # 20.0
        tracker.record_llm(1000, call_id="c1")  # 0.0

        assert tracker.total_cost_inr == pytest.approx(50.0)

    def test_empty_tracker_is_zero(self) -> None:
        tracker = CostTracker()

        assert tracker.total_cost_inr == 0.0


# ---------------------------------------------------------------------------
# TestTotalByService
# ---------------------------------------------------------------------------


class TestTotalByService:
    """total_by_service groups cost by service name."""

    def test_groups_services(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="c1")
        tracker.record_tts(10_000, call_id="c2")
        tracker.record_translate(10_000, call_id="c1")

        by_service = tracker.total_by_service

        assert by_service["tts"] == pytest.approx(60.0)
        assert by_service["translate"] == pytest.approx(20.0)

    def test_empty_tracker_returns_empty_dict(self) -> None:
        tracker = CostTracker()

        assert tracker.total_by_service == {}


# ---------------------------------------------------------------------------
# TestCostForCall
# ---------------------------------------------------------------------------


class TestCostForCall:
    """cost_for_call filters entries by call_id."""

    def test_filters_by_call_id(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="A")
        tracker.record_tts(10_000, call_id="B")
        tracker.record_translate(10_000, call_id="A")

        assert tracker.cost_for_call("A") == pytest.approx(50.0)
        assert tracker.cost_for_call("B") == pytest.approx(30.0)

    def test_unknown_call_id_returns_zero(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="A")

        assert tracker.cost_for_call("nonexistent") == 0.0


# ---------------------------------------------------------------------------
# TestBreakdownForCall
# ---------------------------------------------------------------------------


class TestBreakdownForCall:
    """breakdown_for_call returns detailed per-service cost structure."""

    def test_structure(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(5000, call_id="c1")
        tracker.record_translate(2000, call_id="c1")
        tracker.record_llm(100, call_id="c1")

        bd = tracker.breakdown_for_call("c1")

        assert bd["call_id"] == "c1"
        assert bd["api_call_count"] == 3
        assert "tts" in bd["by_service"]
        assert "translate" in bd["by_service"]
        assert "llm" in bd["by_service"]
        assert bd["by_service"]["tts"]["unit_type"] == "chars"
        assert bd["by_service"]["translate"]["unit_type"] == "chars"
        assert bd["by_service"]["llm"]["unit_type"] == "tokens"

    def test_total_inr_matches_sum(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="c1")
        tracker.record_stt(60.0, call_id="c1")

        bd = tracker.breakdown_for_call("c1")

        expected = 30.0 + (60.0 * 30.0 / 3600)
        assert bd["total_inr"] == pytest.approx(expected, rel=1e-3)

    def test_empty_call_returns_zero(self) -> None:
        tracker = CostTracker()

        bd = tracker.breakdown_for_call("nonexistent")

        assert bd["total_inr"] == 0.0
        assert bd["api_call_count"] == 0
        assert bd["by_service"] == {}


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------


class TestSummary:
    """summary() returns the expected top-level keys."""

    def test_expected_keys(self) -> None:
        tracker = CostTracker()
        tracker.record_tts(100, call_id="c1")

        s = tracker.summary()

        assert "total_inr" in s
        assert "by_service" in s
        assert "call_count" in s
        assert "api_calls" in s
        assert "avg_cost_per_call_inr" in s

    def test_call_count(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(100, call_id="c1")
        tracker.record_tts(200, call_id="c2")
        tracker.record_tts(300, call_id="c1")

        s = tracker.summary()

        assert s["call_count"] == 2  # c1 and c2
        assert s["api_calls"] == 3

    def test_avg_cost_per_call(self) -> None:
        tracker = CostTracker()

        tracker.record_tts(10_000, call_id="c1")  # 30.0
        tracker.record_tts(10_000, call_id="c2")  # 30.0

        s = tracker.summary()

        assert s["avg_cost_per_call_inr"] == pytest.approx(30.0)
