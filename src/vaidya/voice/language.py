"""Language detection, code mapping, and supported-language configuration.

Centralises all language-related constants so other modules import from
here rather than hard-coding BCP-47 tags.
"""

from __future__ import annotations

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class Language(StrEnum):
    """11 languages with full voice support (TTS + STT + Translate).

    These are supported by Bulbul v3 (TTS), Saaras v3 (STT), and
    Mayura v1 (translation).
    """

    HINDI = "hi-IN"
    TAMIL = "ta-IN"
    BENGALI = "bn-IN"
    TELUGU = "te-IN"
    GUJARATI = "gu-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    MARATHI = "mr-IN"
    PUNJABI = "pa-IN"
    ODIA = "od-IN"
    ENGLISH = "en-IN"


class TextLanguage(StrEnum):
    """12 additional languages with text-only support (STT + Translate, no TTS).

    Supported by Saaras v3 (STT) and Sarvam Translate v1 (formal mode).
    Output via SMS/WhatsApp text only — no voice synthesis.
    """

    ASSAMESE = "as-IN"
    URDU = "ur-IN"
    NEPALI = "ne-IN"
    KONKANI = "kok-IN"
    MANIPURI = "mni-IN"
    MAITHILI = "mai-IN"
    BODO = "brx-IN"
    DOGRI = "doi-IN"
    KASHMIRI = "ks-IN"
    SANTALI = "sat-IN"
    SINDHI = "sd-IN"
    SANSKRIT = "sa-IN"


# ---------------------------------------------------------------------------
# Normalisation map: various user/API input forms -> Language enum
# ---------------------------------------------------------------------------

SARVAM_LANG_MAP: dict[str, Language] = {
    # Short ISO codes
    "hi": Language.HINDI,
    "ta": Language.TAMIL,
    "bn": Language.BENGALI,
    "te": Language.TELUGU,
    "gu": Language.GUJARATI,
    "kn": Language.KANNADA,
    "ml": Language.MALAYALAM,
    "mr": Language.MARATHI,
    "pa": Language.PUNJABI,
    "od": Language.ODIA,
    "or": Language.ODIA,  # legacy alias
    "en": Language.ENGLISH,
    # BCP-47 codes (case-insensitive matching handled by normalize_language)
    "hi-in": Language.HINDI,
    "ta-in": Language.TAMIL,
    "bn-in": Language.BENGALI,
    "te-in": Language.TELUGU,
    "gu-in": Language.GUJARATI,
    "kn-in": Language.KANNADA,
    "ml-in": Language.MALAYALAM,
    "mr-in": Language.MARATHI,
    "pa-in": Language.PUNJABI,
    "od-in": Language.ODIA,
    "or-in": Language.ODIA,  # legacy alias
    "en-in": Language.ENGLISH,
    # Full names
    "hindi": Language.HINDI,
    "tamil": Language.TAMIL,
    "bengali": Language.BENGALI,
    "bangla": Language.BENGALI,
    "telugu": Language.TELUGU,
    "gujarati": Language.GUJARATI,
    "kannada": Language.KANNADA,
    "malayalam": Language.MALAYALAM,
    "marathi": Language.MARATHI,
    "punjabi": Language.PUNJABI,
    "odia": Language.ODIA,
    "oriya": Language.ODIA,
    "english": Language.ENGLISH,
}

# ---------------------------------------------------------------------------
# TTS speaker per language (Bulbul v3 — 39 speakers, all cross-language)
# All speaker names verified against sarvamai.TextToSpeechSpeaker enum.
# Each language gets a distinct speaker for variety.
# ---------------------------------------------------------------------------

TTS_SPEAKERS: dict[Language, str] = {
    Language.HINDI: "priya",
    Language.TAMIL: "kavitha",
    Language.BENGALI: "rupali",
    Language.TELUGU: "shreya",
    Language.GUJARATI: "roopa",
    Language.KANNADA: "ishita",
    Language.MALAYALAM: "suhani",
    Language.MARATHI: "neha",
    Language.PUNJABI: "simran",
    Language.ODIA: "pooja",
    Language.ENGLISH: "amelia",
}

# ---------------------------------------------------------------------------
# Display names (English + native script)
# ---------------------------------------------------------------------------

LANGUAGE_DISPLAY_NAMES: dict[Language | TextLanguage, str] = {
    Language.HINDI: "Hindi",
    Language.TAMIL: "Tamil",
    Language.BENGALI: "Bengali",
    Language.TELUGU: "Telugu",
    Language.GUJARATI: "Gujarati",
    Language.KANNADA: "Kannada",
    Language.MALAYALAM: "Malayalam",
    Language.MARATHI: "Marathi",
    Language.PUNJABI: "Punjabi",
    Language.ODIA: "Odia",
    Language.ENGLISH: "English",
    TextLanguage.ASSAMESE: "Assamese",
    TextLanguage.URDU: "Urdu",
    TextLanguage.NEPALI: "Nepali",
    TextLanguage.KONKANI: "Konkani",
    TextLanguage.MANIPURI: "Manipuri",
    TextLanguage.MAITHILI: "Maithili",
    TextLanguage.BODO: "Bodo",
    TextLanguage.DOGRI: "Dogri",
    TextLanguage.KASHMIRI: "Kashmiri",
    TextLanguage.SANTALI: "Santali",
    TextLanguage.SINDHI: "Sindhi",
    TextLanguage.SANSKRIT: "Sanskrit",
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


def is_voice_language(lang: str) -> bool:
    """Return ``True`` if *lang* is one of the 11 TTS-capable languages."""
    return is_supported(lang)


def is_text_language(lang: str) -> bool:
    """Return ``True`` if *lang* is a text-only (non-TTS) language."""
    lower = lang.lower().strip()
    for tl in TextLanguage:
        code = tl.value.lower()
        short = code.split("-")[0]
        if lower in (code, short, tl.name.lower()):
            return True
    return False


def is_any_supported_language(lang: str) -> bool:
    """Return ``True`` if *lang* is supported in any tier (voice or text)."""
    return is_voice_language(lang) or is_text_language(lang)


async def detect_language(client: object, text: str) -> Language:
    """Detect the language of *text* using the Sarvam Language ID API.

    Falls back to :data:`Language.HINDI` if detection fails or the
    detected language is unsupported.

    *client* should be a :class:`~vaidya.sarvam.client.SarvamClient`.
    """
    try:
        lang_code, _script = await client.identify_language(text)  # type: ignore[attr-defined]
        return normalize_language(lang_code)
    except Exception:
        logger.warning("Language detection failed, defaulting to Hindi")
        return Language.HINDI
