"""Language detection, code mapping, and supported-language configuration.

Centralises all language-related constants so other modules import from
here rather than hard-coding BCP-47 tags.
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class Language(StrEnum):
    """Languages supported by Vaidya in Phase 1."""

    HINDI = "hi-IN"
    TAMIL = "ta-IN"
    BENGALI = "bn-IN"
    ENGLISH = "en-IN"


# ---------------------------------------------------------------------------
# Normalisation map: various user/API input forms -> Language enum
# ---------------------------------------------------------------------------

SARVAM_LANG_MAP: dict[str, Language] = {
    # Short ISO codes
    "hi": Language.HINDI,
    "ta": Language.TAMIL,
    "bn": Language.BENGALI,
    "en": Language.ENGLISH,
    # BCP-47 codes (case-insensitive matching handled by normalize_language)
    "hi-in": Language.HINDI,
    "ta-in": Language.TAMIL,
    "bn-in": Language.BENGALI,
    "en-in": Language.ENGLISH,
    # Full names
    "hindi": Language.HINDI,
    "tamil": Language.TAMIL,
    "bengali": Language.BENGALI,
    "bangla": Language.BENGALI,
    "english": Language.ENGLISH,
}

# ---------------------------------------------------------------------------
# TTS speaker per language (Bulbul v2 speaker names)
# ---------------------------------------------------------------------------

TTS_SPEAKERS: dict[Language, str] = {
    Language.HINDI: "meera",
    Language.TAMIL: "meera",
    Language.BENGALI: "meera",
    Language.ENGLISH: "meera",
}

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

LANGUAGE_DISPLAY_NAMES: dict[Language, str] = {
    Language.HINDI: "Hindi",
    Language.TAMIL: "Tamil",
    Language.BENGALI: "Bengali",
    Language.ENGLISH: "English",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_language(lang: str) -> Language:
    """Normalise a language string to a :class:`Language` enum value.

    Accepts ISO-639 short codes (``"hi"``), BCP-47 tags (``"hi-IN"``),
    and full names (``"hindi"``).  Defaults to ``Language.HINDI`` for
    unrecognised input.
    """
    resolved = SARVAM_LANG_MAP.get(lang.lower().strip())
    if resolved is None:
        logger.warning("Unrecognised language input, defaulting to Hindi", extra={"input": lang})
        return Language.HINDI
    return resolved


def is_supported(lang: str) -> bool:
    """Return ``True`` if *lang* maps to a supported language."""
    return lang.lower().strip() in SARVAM_LANG_MAP


def get_sarvam_code(lang: Language) -> str:
    """Return the Sarvam API language code (same as enum value)."""
    return lang.value
