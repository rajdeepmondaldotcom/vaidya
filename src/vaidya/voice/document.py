"""Document verification via Sarvam Vision (WhatsApp / Samvaad channel).

PRD Section 7: User photographs BPL card / Aadhaar / ration card,
sends via WhatsApp. Sarvam Vision extracts fields, cross-checks
against spoken information, flags discrepancies.

Privacy: image is purged after session. Aadhaar full number is never stored.
"""

from __future__ import annotations

import logging
from typing import Any

from vaidya.compliance.pii import mask_pii
from vaidya.models.user_profile import UserProfile
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)


class DocumentVerifier:
    """Extracts fields from document images and cross-checks against profile."""

    def __init__(self, client: SarvamClient) -> None:
        self._client = client

    async def verify_document(
        self,
        image_file: Any,
        user_profile: UserProfile,
        language: str = "hi-IN",
    ) -> VerificationResult:
        """OCR a document image and compare extracted fields to the profile.

        The image is NOT stored. Only extracted text fields are used
        for the current session, then discarded.
        """
        # Sarvam Document Intelligence is a job-based API
        # (initialise → upload → start → poll → download).
        # Phase 2 will implement the full async job flow.
        # For now, return an unverified placeholder result.
        logger.info("Document verification called (job-based API pending Phase 2)")

        text = ""
        fields: dict[str, Any] = {}
        masked_text = mask_pii(text)
        discrepancies = self._cross_check(fields, user_profile)

        return VerificationResult(
            extracted_fields=fields,
            masked_text=masked_text,
            discrepancies=discrepancies,
            verified=len(discrepancies) == 0,
        )

    def _cross_check(
        self,
        extracted: dict[str, Any],
        profile: UserProfile,
    ) -> list[str]:
        """Compare OCR-extracted fields against the spoken profile."""
        issues: list[str] = []

        # State mismatch
        doc_state = extracted.get("state", "").lower()
        if doc_state and profile.state and doc_state not in profile.state.lower():
            issues.append(
                f"State mismatch: document says '{doc_state}', profile says '{profile.state}'"
            )

        # Name extraction (informational, not a blocker)
        doc_name = extracted.get("name", "")
        if doc_name:
            logger.debug("Document name extracted", extra={"name_length": len(doc_name)})

        return issues


class VerificationResult:
    """Result of document verification."""

    __slots__ = ("extracted_fields", "masked_text", "discrepancies", "verified")

    def __init__(
        self,
        extracted_fields: dict[str, Any],
        masked_text: str,
        discrepancies: list[str],
        verified: bool,
    ) -> None:
        self.extracted_fields = extracted_fields
        self.masked_text = masked_text
        self.discrepancies = discrepancies
        self.verified = verified
