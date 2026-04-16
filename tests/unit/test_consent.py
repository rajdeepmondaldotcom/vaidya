"""Tests for consent tracking (DPDP Act compliance).

Covers:
- record_consent with valid and invalid types
- has_consent default / superseded state
- get_records filtering by call_id
- revoke convenience method
- remove_records_for_calls (DPDP erasure)
- _MAX_RECORDS eviction behaviour
- ConsentRecord.to_dict serialisation
"""

from __future__ import annotations

from datetime import UTC, datetime

from vaidya.compliance.consent import ConsentRecord, ConsentTracker

# ---------------------------------------------------------------------------
# record_consent
# ---------------------------------------------------------------------------


class TestRecordConsent:
    def test_record_data_processing_consent(self) -> None:
        tracker = ConsentTracker()
        record = tracker.record_consent("call-1", "data_processing", granted=True)
        assert record.call_id == "call-1"
        assert record.consent_type == "data_processing"
        assert record.granted is True

    def test_record_recording_consent(self) -> None:
        tracker = ConsentTracker()
        record = tracker.record_consent("call-1", "recording", granted=True)
        assert record.consent_type == "recording"
        assert record.granted is True

    def test_record_consent_invalid_type_raises_value_error(self) -> None:
        tracker = ConsentTracker()
        import pytest

        with pytest.raises(ValueError, match="Invalid consent_type"):
            tracker.record_consent("call-1", "marketing", granted=True)

    def test_record_consent_returns_consent_record(self) -> None:
        tracker = ConsentTracker()
        record = tracker.record_consent("call-1", "data_processing", granted=False)
        assert isinstance(record, ConsentRecord)
        assert record.granted is False


# ---------------------------------------------------------------------------
# has_consent
# ---------------------------------------------------------------------------


class TestHasConsent:
    def test_returns_false_when_no_record_exists(self) -> None:
        tracker = ConsentTracker()
        assert tracker.has_consent("call-1", "data_processing") is False

    def test_returns_true_after_grant(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        assert tracker.has_consent("call-1", "data_processing") is True

    def test_returns_latest_value_when_superseded(self) -> None:
        """grant -> revoke -> re-grant should reflect the last event."""
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "recording", granted=True)
        assert tracker.has_consent("call-1", "recording") is True

        tracker.record_consent("call-1", "recording", granted=False)
        assert tracker.has_consent("call-1", "recording") is False

        tracker.record_consent("call-1", "recording", granted=True)
        assert tracker.has_consent("call-1", "recording") is True


# ---------------------------------------------------------------------------
# get_records
# ---------------------------------------------------------------------------


class TestGetRecords:
    def test_returns_only_records_for_specified_call_id(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        tracker.record_consent("call-2", "data_processing", granted=True)
        tracker.record_consent("call-1", "recording", granted=False)

        records = tracker.get_records("call-1")
        assert len(records) == 2
        assert all(r.call_id == "call-1" for r in records)

    def test_returns_empty_list_for_unknown_call(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        assert tracker.get_records("call-999") == []


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------


class TestRevoke:
    def test_revoke_creates_granted_false_record(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        record = tracker.revoke("call-1", "data_processing")
        assert record.granted is False
        assert tracker.has_consent("call-1", "data_processing") is False


# ---------------------------------------------------------------------------
# remove_records_for_calls (DPDP erasure)
# ---------------------------------------------------------------------------


class TestRemoveRecordsForCalls:
    def test_removes_records_and_returns_count(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        tracker.record_consent("call-1", "recording", granted=True)
        tracker.record_consent("call-2", "data_processing", granted=True)

        removed = tracker.remove_records_for_calls(["call-1"])
        assert removed == 2
        assert tracker.get_records("call-1") == []
        # call-2 should be untouched
        assert len(tracker.get_records("call-2")) == 1

    def test_cleans_index_entries(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        tracker.remove_records_for_calls(["call-1"])
        # Index should also be cleared -- has_consent should return default False
        assert tracker.has_consent("call-1", "data_processing") is False

    def test_returns_zero_when_no_matching_calls(self) -> None:
        tracker = ConsentTracker()
        tracker.record_consent("call-1", "data_processing", granted=True)
        assert tracker.remove_records_for_calls(["call-999"]) == 0


# ---------------------------------------------------------------------------
# _MAX_RECORDS eviction
# ---------------------------------------------------------------------------


class TestMaxRecordsEviction:
    def test_evicts_oldest_when_exceeding_limit(self) -> None:
        tracker = ConsentTracker()
        # Temporarily lower the cap for testing
        original_max = ConsentTracker._MAX_RECORDS
        ConsentTracker._MAX_RECORDS = 5
        try:
            for i in range(7):
                tracker.record_consent(f"call-{i}", "data_processing", granted=True)
            # Only the last 5 records should remain
            assert len(tracker._records) == 5
            # Oldest calls (call-0, call-1) should have been evicted
            remaining_ids = {r.call_id for r in tracker._records}
            assert "call-0" not in remaining_ids
            assert "call-1" not in remaining_ids
            assert "call-6" in remaining_ids
        finally:
            ConsentTracker._MAX_RECORDS = original_max


# ---------------------------------------------------------------------------
# ConsentRecord.to_dict serialisation
# ---------------------------------------------------------------------------


class TestConsentRecordToDict:
    def test_to_dict_serialisation_format(self) -> None:
        ts = datetime(2026, 4, 15, 10, 30, 0, tzinfo=UTC)
        record = ConsentRecord(
            call_id="call-42",
            consented_at=ts,
            consent_type="data_processing",
            granted=True,
        )
        d = record.to_dict()
        assert d["call_id"] == "call-42"
        assert d["consented_at"] == ts.isoformat()
        assert d["consent_type"] == "data_processing"
        assert d["granted"] is True

    def test_to_dict_keys(self) -> None:
        tracker = ConsentTracker()
        record = tracker.record_consent("call-1", "recording", granted=False)
        d = record.to_dict()
        assert set(d.keys()) == {"call_id", "consented_at", "consent_type", "granted"}
