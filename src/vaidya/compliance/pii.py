"""PII detection and masking for Vaidya voice transcripts.

Handles Indian PII patterns: Aadhaar numbers, mobile numbers, bank
account numbers, and PAN cards.  Masking is applied *before* text is
stored in the session or audit trail, ensuring PII never persists at rest.

Agents receive the *raw* (unmasked) text for accurate eligibility matching;
only the storage path masks.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Aadhaar: 12 digits, optionally grouped as 4-4-4 with spaces or hyphens
AADHAAR_PATTERN = re.compile(r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b")

# Indian mobile: starts with 6-9, exactly 10 digits
# Negative lookbehind/lookahead to avoid matching inside longer digit runs
PHONE_PATTERN = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")

# Bank account: 9-18 consecutive digits, anchored to a context word to
# reduce false positives on other long numbers.
_BANK_CONTEXT = r"(?:account|khata|a/c|bank)\s*(?:no\.?|number|num)?\s*:?\s*"
BANK_ACCOUNT_PATTERN = re.compile(_BANK_CONTEXT + r"(\d{9,18})", re.IGNORECASE)

# PAN card: ABCDE1234F format
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PIIMatch(NamedTuple):
    """A single PII detection result."""

    pii_type: str
    start: int
    end: int
    masked_value: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mask_pii(text: str) -> str:
    """Return *text* with all detected PII replaced by fixed masks.

    Application order matters: Aadhaar (12 digits) is matched first so
    its sub-spans are not partially caught by the phone pattern (10 digits).
    """
    # 1. Aadhaar -- highest priority, longest digit span
    text = AADHAAR_PATTERN.sub("XXXX-XXXX-XXXX", text)

    # 2. Bank account -- context-anchored, run before generic digit patterns
    text = BANK_ACCOUNT_PATTERN.sub(
        lambda m: m.group(0)[: m.start(1) - m.start()] + "X" * len(m.group(1)),
        text,
    )

    # 3. PAN
    text = PAN_PATTERN.sub("XXXXX0000X", text)

    # 4. Phone -- shortest digit span, last
    text = PHONE_PATTERN.sub("XXXXXXXXXX", text)

    return text


def detect_pii(text: str) -> list[PIIMatch]:
    """Scan *text* and return all PII occurrences with positions.

    Does **not** modify the input -- returns metadata so the caller can
    decide what to do (log, warn, mask selectively).
    """
    findings: list[PIIMatch] = []

    for m in AADHAAR_PATTERN.finditer(text):
        findings.append(
            PIIMatch(
                pii_type="aadhaar",
                start=m.start(),
                end=m.end(),
                masked_value="XXXX-XXXX-XXXX",
            )
        )

    for m in PHONE_PATTERN.finditer(text):
        # Skip if this span overlaps with an already-flagged Aadhaar match
        if any(f.start <= m.start() < f.end for f in findings):
            continue
        findings.append(
            PIIMatch(
                pii_type="phone",
                start=m.start(),
                end=m.end(),
                masked_value="XXXXXXXXXX",
            )
        )

    for m in BANK_ACCOUNT_PATTERN.finditer(text):
        findings.append(
            PIIMatch(
                pii_type="bank_account",
                start=m.start(1),
                end=m.end(1),
                masked_value="X" * len(m.group(1)),
            )
        )

    for m in PAN_PATTERN.finditer(text):
        findings.append(
            PIIMatch(
                pii_type="pan",
                start=m.start(),
                end=m.end(),
                masked_value="XXXXX0000X",
            )
        )

    # Sort by position for deterministic output
    findings.sort(key=lambda f: f.start)
    return findings


def contains_aadhaar(text: str) -> bool:
    """Quick check for Aadhaar numbers in *text*."""
    return bool(AADHAAR_PATTERN.search(text))
