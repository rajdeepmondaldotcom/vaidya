"""Domain-specific terms preserved through translation round-trips.

These terms are replaced with numbered tokens before translation
and restored after, preventing the translation API from mangling
scheme names and acronyms.
"""

from __future__ import annotations

import re

PRESERVE_TERMS: list[str] = [
    "Ayushman Bharat",
    "Jan Seva Kendra",
    "PM-JAY",
    "PMJAY",
    "Aadhaar",
    "BPL",
    "SECC",
    "CSC",
    "RSBY",
    "CGHS",
    "ESIC",
    "Jan Arogya",
    "Golden Card",
    "Ayushman Card",
    "14555",
    "104",
    "108",
]

PRESERVE_TERMS_SORTED = sorted(PRESERVE_TERMS, key=len, reverse=True)

PRESERVE_RE = re.compile(
    "|".join(re.escape(t) for t in PRESERVE_TERMS_SORTED),
    re.IGNORECASE,
)
