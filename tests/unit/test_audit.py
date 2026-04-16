"""Tests for AuditTrail: JSONL logging, eligibility decisions, generic events.

Covers:
- __init__ creates the audit directory
- log_turn appends a valid JSONL entry to {call_id}.jsonl
- log_turn entry has required fields (event, timestamp, call_id, phase, agent_name,
  input_text, output_text, latency_ms)
- log_eligibility_decision writes eligibility/reviewer/convergence data
- log_event writes generic events
- Multiple log_turn calls produce multiple lines (append-only)
- Each JSONL line is valid JSON (parse every line)
"""

from __future__ import annotations

import json
from pathlib import Path

from vaidya.compliance.audit import AuditTrail
from vaidya.models.scheme import (
    ConvergenceResult,
    DisagreementRecord,
    EligibilityResult,
    EligibilityVerdict,
    ReviewerResult,
    SchemeMatch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed dicts."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _make_scheme_match(
    scheme_id: str = "PMJAY",
    verdict: EligibilityVerdict = EligibilityVerdict.ELIGIBLE,
) -> SchemeMatch:
    return SchemeMatch(
        scheme_id=scheme_id,
        scheme_name="Test Scheme",
        verdict=verdict,
        confidence=0.9,
        reasoning_trace="income below threshold",
        matched_criteria=["income"],
        failed_criteria=[],
        coverage_summary="Rs 5 lakh",
    )


# ---------------------------------------------------------------------------
# TestAuditInit
# ---------------------------------------------------------------------------


class TestAuditInit:
    def test_creates_audit_directory(self, tmp_path: Path) -> None:
        audit_dir = tmp_path / "audit_logs"
        assert not audit_dir.exists()

        AuditTrail(audit_dir=str(audit_dir))

        assert audit_dir.exists()
        assert audit_dir.is_dir()

    def test_does_not_fail_if_dir_exists(self, tmp_path: Path) -> None:
        audit_dir = tmp_path / "audit_logs"
        audit_dir.mkdir()

        trail = AuditTrail(audit_dir=str(audit_dir))

        assert trail is not None


# ---------------------------------------------------------------------------
# TestLogTurn
# ---------------------------------------------------------------------------


class TestLogTurn:
    def test_creates_jsonl_file(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_turn(
            call_id="call-001",
            phase="intake",
            agent_name="intake_agent",
            input_text="Main Rajasthan se hoon",
            output_text="Aapki family mein kitne log hain?",
            latency_ms=150.5,
        )

        path = tmp_path / "call-001.jsonl"
        assert path.exists()

    def test_entry_has_required_fields(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_turn(
            call_id="call-002",
            phase="intake",
            agent_name="intake_agent",
            input_text="user input",
            output_text="agent response",
            latency_ms=200.3,
        )

        entries = _read_jsonl(tmp_path / "call-002.jsonl")
        assert len(entries) == 1
        entry = entries[0]

        assert entry["event"] == "turn"
        assert "timestamp" in entry
        assert entry["call_id"] == "call-002"
        assert entry["phase"] == "intake"
        assert entry["agent_name"] == "intake_agent"
        assert entry["input_text"] == "user input"
        assert entry["output_text"] == "agent response"
        assert entry["latency_ms"] == 200.3

    def test_latency_rounded(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_turn(
            call_id="call-003",
            phase="processing",
            agent_name="eligibility",
            input_text="in",
            output_text="out",
            latency_ms=123.456789,
        )

        entries = _read_jsonl(tmp_path / "call-003.jsonl")
        assert entries[0]["latency_ms"] == 123.5

    def test_multiple_turns_append(self, tmp_path: Path) -> None:
        """Multiple log_turn calls produce multiple JSONL lines (append-only)."""
        trail = AuditTrail(audit_dir=str(tmp_path))

        for i in range(5):
            trail.log_turn(
                call_id="call-multi",
                phase="intake",
                agent_name="intake_agent",
                input_text=f"turn {i}",
                output_text=f"response {i}",
                latency_ms=float(i * 10),
            )

        entries = _read_jsonl(tmp_path / "call-multi.jsonl")
        assert len(entries) == 5
        assert entries[0]["input_text"] == "turn 0"
        assert entries[4]["input_text"] == "turn 4"

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        """Every line in the JSONL file must parse as valid JSON."""
        trail = AuditTrail(audit_dir=str(tmp_path))

        for i in range(3):
            trail.log_turn(
                call_id="call-valid",
                phase="intake",
                agent_name="agent",
                input_text=f"input {i}",
                output_text=f"output {i}",
                latency_ms=10.0,
            )

        raw_lines = (tmp_path / "call-valid.jsonl").read_text().strip().splitlines()
        for line in raw_lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TestLogEligibilityDecision
# ---------------------------------------------------------------------------


class TestLogEligibilityDecision:
    def test_writes_eligibility_data(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        elig = EligibilityResult(
            matches=[_make_scheme_match("PMJAY")],
            processing_time_ms=300.0,
            model_used="sarvam-105b",
            schemes_evaluated=1,
        )

        trail.log_eligibility_decision(
            call_id="call-elig",
            eligibility_result=elig,
            reviewer_result=None,
            convergence_result=None,
        )

        entries = _read_jsonl(tmp_path / "call-elig.jsonl")
        assert len(entries) == 1
        entry = entries[0]

        assert entry["event"] == "eligibility_decision"
        assert "eligibility" in entry
        assert entry["eligibility"]["model_used"] == "sarvam-105b"
        assert entry["eligibility"]["schemes_evaluated"] == 1
        assert len(entry["eligibility"]["matches"]) == 1
        assert entry["eligibility"]["matches"][0]["scheme_id"] == "PMJAY"

    def test_writes_reviewer_data(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        reviewer = ReviewerResult(
            matches=[_make_scheme_match("PMJAY")],
            processing_time_ms=250.0,
            model_used="sarvam-105b",
            transcript_evidence=["User said daily wage"],
        )

        trail.log_eligibility_decision(
            call_id="call-rev",
            eligibility_result=None,
            reviewer_result=reviewer,
            convergence_result=None,
        )

        entries = _read_jsonl(tmp_path / "call-rev.jsonl")
        entry = entries[0]

        assert "reviewer" in entry
        assert entry["reviewer"]["model_used"] == "sarvam-105b"
        assert entry["reviewer"]["transcript_evidence"] == ["User said daily wage"]

    def test_writes_convergence_data(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        match = _make_scheme_match("PMJAY")
        disagreement = DisagreementRecord(
            scheme_id="RSBY",
            scheme_name="Rashtriya Swasthya Bima Yojana",
            eligibility_verdict=EligibilityVerdict.ELIGIBLE,
            reviewer_verdict=EligibilityVerdict.INELIGIBLE,
            eligibility_reasoning="Income matches",
            reviewer_reasoning="Occupation excluded",
            disagreement_field="occupation",
            resolved_from_transcript=False,
            final_verdict=EligibilityVerdict.UNCERTAIN,
            caveat="occupation unclear",
        )
        convergence = ConvergenceResult(
            agreed_eligible=[match],
            agreed_ineligible=["SCHEME-X"],
            disagreements=[disagreement],
            conservative_eligible=[],
        )

        trail.log_eligibility_decision(
            call_id="call-conv",
            eligibility_result=None,
            reviewer_result=None,
            convergence_result=convergence,
        )

        entries = _read_jsonl(tmp_path / "call-conv.jsonl")
        entry = entries[0]

        assert "convergence" in entry
        conv = entry["convergence"]
        assert conv["agreed_eligible"] == ["PMJAY"]
        assert conv["agreed_ineligible"] == ["SCHEME-X"]
        assert len(conv["disagreements"]) == 1
        assert conv["disagreements"][0]["scheme_id"] == "RSBY"
        assert conv["disagreements"][0]["final_verdict"] == "uncertain"

    def test_includes_version_info(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_eligibility_decision(
            call_id="call-ver",
            eligibility_result=None,
            reviewer_result=None,
            convergence_result=None,
        )

        entries = _read_jsonl(tmp_path / "call-ver.jsonl")
        assert "versions" in entries[0]
        assert "pipeline" in entries[0]["versions"]


# ---------------------------------------------------------------------------
# TestLogEvent
# ---------------------------------------------------------------------------


class TestLogEvent:
    def test_writes_generic_event(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_event(
            call_id="call-evt",
            event_type="consent_given",
            data={"method": "voice", "language": "hi-IN"},
        )

        entries = _read_jsonl(tmp_path / "call-evt.jsonl")
        assert len(entries) == 1
        entry = entries[0]

        assert entry["event"] == "consent_given"
        assert entry["call_id"] == "call-evt"
        assert entry["data"]["method"] == "voice"

    def test_event_without_data(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_event(call_id="call-nodata", event_type="session_start")

        entries = _read_jsonl(tmp_path / "call-nodata.jsonl")
        assert entries[0]["event"] == "session_start"
        assert "data" not in entries[0]

    def test_event_has_timestamp(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_event(call_id="call-ts", event_type="test")

        entries = _read_jsonl(tmp_path / "call-ts.jsonl")
        assert "timestamp" in entries[0]
        assert len(entries[0]["timestamp"]) > 0


# ---------------------------------------------------------------------------
# TestGetAuditLog
# ---------------------------------------------------------------------------


class TestGetAuditLog:
    def test_reads_back_entries(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        trail.log_event(call_id="call-read", event_type="event_a")
        trail.log_event(call_id="call-read", event_type="event_b")

        entries = trail.get_audit_log("call-read")

        assert len(entries) == 2
        assert entries[0]["event"] == "event_a"
        assert entries[1]["event"] == "event_b"

    def test_returns_empty_for_missing_call(self, tmp_path: Path) -> None:
        trail = AuditTrail(audit_dir=str(tmp_path))

        entries = trail.get_audit_log("nonexistent-call")

        assert entries == []
