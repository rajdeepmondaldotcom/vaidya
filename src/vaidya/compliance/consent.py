"""Per-call consent tracking for Vaidya voice sessions.

Records explicit consent events (data processing, recording) with
timestamps.  The tracker is designed to be attached to a single service
instance and serialised alongside the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


# Valid consent types -- extend as new consent scopes are added
_VALID_TYPES = frozenset({"data_processing", "recording"})


@dataclass(frozen=True)
class ConsentRecord:
    """An immutable record of a single consent event."""

    call_id: str
    consented_at: datetime
    consent_type: str  # "data_processing" | "recording"
    granted: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON storage / audit logging."""
        return {
            "call_id": self.call_id,
            "consented_at": self.consented_at.isoformat(),
            "consent_type": self.consent_type,
            "granted": self.granted,
        }


class ConsentTracker:
    """In-memory consent store for the lifetime of a service instance.

    Append-only: consent events are never deleted, only superseded by
    newer events for the same ``(call_id, consent_type)`` pair.  The
    ``has_consent`` method always returns the *latest* state.
    """

    _MAX_RECORDS = 10_000  # evict oldest when exceeded

    def __init__(self) -> None:
        self._records: list[ConsentRecord] = []
        # Fast lookup: (call_id, consent_type) -> latest granted value
        self._index: dict[tuple[str, str], bool] = {}

    def record_consent(
        self,
        call_id: str,
        consent_type: str,
        granted: bool,
    ) -> ConsentRecord:
        """Record a consent event.

        Parameters
        ----------
        call_id:
            Unique identifier for the phone call.
        consent_type:
            One of ``"data_processing"`` or ``"recording"``.
        granted:
            Whether the user granted or denied consent.

        Returns
        -------
        ConsentRecord
            The created record.

        Raises
        ------
        ValueError
            If *consent_type* is not a recognised type.
        """
        if consent_type not in _VALID_TYPES:
            raise ValueError(
                f"Invalid consent_type '{consent_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_TYPES))}"
            )

        record = ConsentRecord(
            call_id=call_id,
            consented_at=datetime.now(UTC),
            consent_type=consent_type,
            granted=granted,
        )
        self._records.append(record)
        self._index[(call_id, consent_type)] = granted

        # Evict oldest records if we exceed the cap
        if len(self._records) > self._MAX_RECORDS:
            self._records = self._records[-self._MAX_RECORDS :]

        logger.info(
            "Consent recorded",
            extra={
                "call_id": call_id,
                "consent_type": consent_type,
                "granted": granted,
            },
        )
        return record

    def has_consent(self, call_id: str, consent_type: str) -> bool:
        """Check whether consent has been granted for *call_id* and *consent_type*.

        Returns ``False`` if no consent event was recorded or if the latest
        event was a denial.
        """
        return self._index.get((call_id, consent_type), False)

    def get_records(self, call_id: str) -> list[ConsentRecord]:
        """Return all consent records for a given call, in chronological order."""
        return [r for r in self._records if r.call_id == call_id]

    def revoke(self, call_id: str, consent_type: str) -> ConsentRecord:
        """Record a consent revocation (granted=False).

        Convenience wrapper around :meth:`record_consent`.
        """
        return self.record_consent(call_id, consent_type, granted=False)

    def remove_records_for_calls(self, call_ids: list[str]) -> int:
        """Remove all consent records for the given call IDs (DPDP Act erasure).

        Returns the number of records removed.
        """
        call_id_set = set(call_ids)
        before = len(self._records)
        self._records = [r for r in self._records if r.call_id not in call_id_set]
        # Clean up index entries
        keys_to_remove = [k for k in self._index if k[0] in call_id_set]
        for k in keys_to_remove:
            del self._index[k]
        removed = before - len(self._records)
        logger.info(
            "Consent records removed for DPDP erasure",
            extra={"call_ids": call_ids, "records_removed": removed},
        )
        return removed
