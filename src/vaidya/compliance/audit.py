"""Immutable, append-only audit trail for Vaidya voice sessions.

Every agent turn, eligibility decision, and consent event is logged to a
per-call JSONL file.  The audit directory is designed to be backed up or
shipped to a compliance store without transformation.

Phase 1: local JSONL files.
Phase 2: append-only PostgreSQL or cloud audit store.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import vaidya
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import ConvergenceResult, EligibilityResult, ReviewerResult

logger = logging.getLogger(__name__)


class AuditTrail:
    """Append-only file-based audit trail.

    Each call produces one JSONL file (``{call_id}.jsonl``) where every
    line is a self-contained JSON object.  Append-only writes ensure
    immutability: once a line is written it is never modified.
    """

    def __init__(self, audit_dir: str = "data/audit") -> None:
        self._dir = Path(audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("AuditTrail initialised", extra={"audit_dir": str(self._dir)})

    # ------------------------------------------------------------------
    # Turn logging
    # ------------------------------------------------------------------

    def log_turn(
        self,
        call_id: str,
        phase: str,
        agent_name: str,
        input_text: str,
        output_text: str,
        latency_ms: float,
    ) -> None:
        """Append a conversational turn to the audit log.

        Parameters
        ----------
        call_id:
            Unique call identifier.
        phase:
            Current conversation phase (e.g. ``"intake"``, ``"processing"``).
        agent_name:
            The agent that handled this turn.
        input_text:
            PII-masked user input.
        output_text:
            Agent response text.
        latency_ms:
            Processing time in milliseconds.
        """
        entry = {
            "event": "turn",
            "timestamp": _now_iso(),
            "call_id": call_id,
            "phase": phase,
            "agent_name": agent_name,
            "input_text": input_text,
            "output_text": output_text,
            "latency_ms": round(latency_ms, 1),
        }
        self._append(call_id, entry)

    # ------------------------------------------------------------------
    # Eligibility decision logging
    # ------------------------------------------------------------------

    def log_eligibility_decision(
        self,
        call_id: str,
        eligibility_result: EligibilityResult | None,
        reviewer_result: ReviewerResult | None,
        convergence_result: ConvergenceResult | None,
        context: ConversationContext | None = None,
    ) -> None:
        """Log the full eligibility evaluation with both agents' reasoning traces.

        This is the most critical audit entry -- it captures exactly *why*
        a user was told they are eligible (or not) for each scheme.

        Parameters
        ----------
        context:
            If provided, a SHA-256 hash prefix of the user profile is
            included for re-verification (DPDP Act audit requirement).
        """
        entry: dict[str, Any] = {
            "event": "eligibility_decision",
            "timestamp": _now_iso(),
            "call_id": call_id,
        }

        # Gap 2: Version tracking (PRD Section 11.4)
        entry["versions"] = {
            "pipeline": vaidya.__version__,
            "model_eligibility": eligibility_result.model_used if eligibility_result else None,
            "model_reviewer": reviewer_result.model_used if reviewer_result else None,
            "scheme_data_version": self._get_scheme_version(),
        }

        # Gap 3: User profile hash for re-verification
        if context is not None:
            profile_hash = hashlib.sha256(
                context.user_profile.model_dump_json().encode()
            ).hexdigest()[:16]
            entry["user_profile_hash"] = profile_hash

        if eligibility_result is not None:
            entry["eligibility"] = {
                "model_used": eligibility_result.model_used,
                "schemes_evaluated": eligibility_result.schemes_evaluated,
                "processing_time_ms": eligibility_result.processing_time_ms,
                "matches": [
                    {
                        "scheme_id": m.scheme_id,
                        "verdict": m.verdict.value,
                        "confidence": m.confidence,
                        "reasoning_trace": m.reasoning_trace,
                        "matched_criteria": m.matched_criteria,
                        "failed_criteria": m.failed_criteria,
                    }
                    for m in eligibility_result.matches
                ],
            }

        if reviewer_result is not None:
            entry["reviewer"] = {
                "model_used": reviewer_result.model_used,
                "processing_time_ms": reviewer_result.processing_time_ms,
                "transcript_evidence": reviewer_result.transcript_evidence,
                "matches": [
                    {
                        "scheme_id": m.scheme_id,
                        "verdict": m.verdict.value,
                        "confidence": m.confidence,
                        "reasoning_trace": m.reasoning_trace,
                        "matched_criteria": m.matched_criteria,
                        "failed_criteria": m.failed_criteria,
                    }
                    for m in reviewer_result.matches
                ],
            }

        if convergence_result is not None:
            entry["convergence"] = {
                "agreed_eligible": [m.scheme_id for m in convergence_result.agreed_eligible],
                "agreed_ineligible": convergence_result.agreed_ineligible,
                "conservative_eligible": [
                    m.scheme_id for m in convergence_result.conservative_eligible
                ],
                "disagreements": [
                    {
                        "scheme_id": d.scheme_id,
                        "eligibility_verdict": d.eligibility_verdict.value,
                        "reviewer_verdict": d.reviewer_verdict.value,
                        "disagreement_field": d.disagreement_field,
                        "resolved_from_transcript": d.resolved_from_transcript,
                        "final_verdict": d.final_verdict.value,
                        "caveat": d.caveat,
                    }
                    for d in convergence_result.disagreements
                ],
            }

        self._append(call_id, entry)

    # ------------------------------------------------------------------
    # Generic event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        call_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Log an arbitrary event (consent, error, session lifecycle, etc.)."""
        entry: dict[str, Any] = {
            "event": event_type,
            "timestamp": _now_iso(),
            "call_id": call_id,
        }
        if data:
            entry["data"] = data
        self._append(call_id, entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_audit_log(self, call_id: str) -> list[dict[str, Any]]:
        """Read the complete audit trail for a call.

        Returns an empty list if no log file exists.
        """
        path = self._call_path(call_id)
        if not path.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Corrupt audit line",
                            extra={"call_id": call_id, "line": line_number},
                        )
        except OSError as exc:
            logger.error(
                "Failed to read audit log",
                extra={"call_id": call_id, "error": str(exc)},
            )
        return entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_path(self, call_id: str) -> Path:
        """Return the JSONL file path for a given call.

        Sanitises *call_id* to prevent path-traversal attacks.
        """
        safe_id = "".join(c for c in call_id if c.isalnum() or c in "-_")
        if not safe_id:
            safe_id = "unknown"
        return self._dir / f"{safe_id}.jsonl"

    @staticmethod
    def _get_scheme_version() -> str:
        """Derive scheme data version from loaded scheme records."""
        try:
            from vaidya.schemes.registry import get_schemes

            schemes = get_schemes()
            if schemes:
                dates = [s.effective_date for s in schemes if s.effective_date]
                return max(dates) if dates else "unknown"
        except Exception:
            pass
        return "unknown"

    def _append(self, call_id: str, entry: dict[str, Any]) -> None:
        """Append a single JSON line to the call's audit file."""
        path = self._call_path(call_id)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.error(
                "Failed to write audit entry",
                extra={"call_id": call_id, "error": str(exc)},
            )


def _now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()
