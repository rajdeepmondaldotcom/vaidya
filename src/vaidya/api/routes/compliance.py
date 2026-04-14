"""DPDP Act compliance endpoints — data deletion (right to erasure)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from vaidya.compliance.audit import AuditTrail
from vaidya.compliance.consent import ConsentTracker
from vaidya.dependencies import get_audit_trail, get_consent_tracker, get_session
from vaidya.session.manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.delete("/data/{phone_hash}")
async def delete_user_data(
    phone_hash: str,
    session: SessionManager = Depends(get_session),
    audit: AuditTrail = Depends(get_audit_trail),
    consent: ConsentTracker = Depends(get_consent_tracker),
) -> dict:
    """Delete all records for a phone number hash (DPDP Act compliance).

    Removes: session data, audit trail, consent records.
    Returns confirmation with deleted record count.
    """
    deleted_counts = {
        "sessions": 0,
        "audit_files": 0,
        "consent_records": 0,
    }

    # 1. Find all call_ids associated with this phone_hash from audit files.
    #    Scan audit directory for files and check each for the phone_hash.
    call_ids: list[str] = []
    for audit_file in sorted(audit._dir.glob("*.jsonl")):
        call_id = audit_file.stem
        entries = audit.get_audit_log(call_id)
        for entry in entries:
            data = entry.get("data", {})
            if data.get("phone_hash") == phone_hash:
                call_ids.append(call_id)
                break

    if not call_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No records found for phone_hash: {phone_hash}",
        )

    # 2. Log the deletion event itself before deleting (audit requirement).
    for call_id in call_ids:
        audit.log_event(
            call_id,
            "data_deletion_requested",
            {
                "phone_hash": phone_hash,
                "reason": "DPDP Act right to erasure",
                "call_ids_affected": call_ids,
            },
        )

    # 3. Delete Redis sessions.
    for call_id in call_ids:
        if await session.exists(call_id):
            await session.delete(call_id)
            deleted_counts["sessions"] += 1

    # 4. Delete audit JSONL files.
    for call_id in call_ids:
        audit_path = audit._call_path(call_id)
        if audit_path.exists():
            audit_path.unlink()
            deleted_counts["audit_files"] += 1

    # 5. Remove consent records for these call_ids.
    removed = consent.remove_records_for_calls(call_ids)
    deleted_counts["consent_records"] = removed

    total = sum(deleted_counts.values())
    logger.info(
        "DPDP data deletion completed",
        extra={
            "phone_hash": phone_hash,
            "call_ids": call_ids,
            "deleted_counts": deleted_counts,
        },
    )

    return {
        "status": "deleted",
        "phone_hash": phone_hash,
        "call_ids_affected": call_ids,
        "deleted_counts": deleted_counts,
        "total_records_deleted": total,
    }
