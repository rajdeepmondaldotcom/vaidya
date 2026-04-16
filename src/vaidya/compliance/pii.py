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

# Indian mobile: starts with 6-9, exactly 10 digits, optionally prefixed by +91 or 91
PHONE_PATTERN = re.compile(r"(?:\+?91[\s-]?)?(?<!\d)[6-9]\d{9}(?!\d)")

# Bank account: 9-18 consecutive digits, anchored to a context word to
# reduce false positives on other long numbers.
_BANK_CONTEXT = r"(?:account|khata|a/c|bank)\s*(?:no\.?|number|num)?\s*:?\s*"
BANK_ACCOUNT_PATTERN = re.compile(_BANK_CONTEXT + r"(\d{9,18})", re.IGNORECASE)

# PAN card: ABCDE1234F format
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")


# ---------------------------------------------------------------------------
# PII type definitions for detection
# ---------------------------------------------------------------------------

# Each entry: (pii_type, pattern, masked_value_fn, use_group)
# masked_value_fn receives the match object and returns the masked string.
# use_group: which match group to use for start/end (0 = full match).
_PII_DETECTORS: list[tuple[str, re.Pattern[str], int]] = [
    ("phone", PHONE_PATTERN, 0),  # Phone FIRST - catches +91 prefixed numbers
    ("aadhaar", AADHAAR_PATTERN, 0),  # Aadhaar second - skips phone overlaps
    ("bank_account", BANK_ACCOUNT_PATTERN, 1),
    ("pan", PAN_PATTERN, 0),
]

_PII_MASKS: dict[str, str] = {
    "aadhaar": "XXXX-XXXX-XXXX",
    "phone": "XXXXXXXXXX",
    "pan": "XXXXX0000X",
}


def _mask_for_match(pii_type: str, match: re.Match[str], group: int) -> str:
    """Return the masked value string for a PII match."""
    if pii_type == "bank_account":
        return "X" * len(match.group(group))
    if pii_type == "phone":
        return "X" * len(match.group(group))
    return _PII_MASKS.get(pii_type, "X" * len(match.group(group)))


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
# Internal helper
# ---------------------------------------------------------------------------


def _find_pii_matches(
    text: str,
    pii_type: str,
    pattern: re.Pattern[str],
    group: int,
    existing: list[PIIMatch],
) -> list[PIIMatch]:
    """Find all matches of *pattern* in *text*, skipping overlaps with *existing*.

    Parameters
    ----------
    text:
        The text to scan.
    pii_type:
        Label for the PII type (e.g. ``"aadhaar"``).
    pattern:
        Compiled regex to search with.
    group:
        Which capture group to use for span boundaries (0 = full match).
    existing:
        Already-found matches; overlapping spans are skipped.
    """
    matches: list[PIIMatch] = []
    for m in pattern.finditer(text):
        span_start = m.start(group)
        span_end = m.end(group)
        # Skip if this span overlaps with an already-flagged match
        if any(f.start <= span_start < f.end for f in existing):
            continue
        matches.append(
            PIIMatch(
                pii_type=pii_type,
                start=span_start,
                end=span_end,
                masked_value=_mask_for_match(pii_type, m, group),
            )
        )
    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mask_pii(text: str) -> str:
    """Return *text* with all detected PII replaced by fixed masks.

    Application order matters: phone with country code is matched first so
    that +91-prefixed numbers are caught before Aadhaar grabs them.
    Aadhaar (12 digits) is matched second to avoid partial phone overlap.
    """
    # 1. Phone with country code -- catch these first before Aadhaar grabs them
    text = PHONE_PATTERN.sub(lambda m: "X" * len(m.group()), text)

    # 2. Aadhaar -- highest priority digit span after phones are removed
    text = AADHAAR_PATTERN.sub("XXXX-XXXX-XXXX", text)

    # 3. Bank account -- context-anchored, run before generic digit patterns
    text = BANK_ACCOUNT_PATTERN.sub(
        lambda m: m.group(0)[: m.start(1) - m.start()] + "X" * len(m.group(1)),
        text,
    )

    # 4. PAN
    text = PAN_PATTERN.sub("XXXXX0000X", text)

    return text


def detect_pii(text: str) -> list[PIIMatch]:
    """Scan *text* and return all PII occurrences with positions.

    Does **not** modify the input -- returns metadata so the caller can
    decide what to do (log, warn, mask selectively).
    """
    findings: list[PIIMatch] = []

    for pii_type, pattern, group in _PII_DETECTORS:
        findings.extend(_find_pii_matches(text, pii_type, pattern, group, findings))

    # Sort by position for deterministic output
    findings.sort(key=lambda f: f.start)
    return findings


def contains_aadhaar(text: str) -> bool:
    """Quick check for Aadhaar numbers in *text*."""
    return bool(AADHAAR_PATTERN.search(text))
