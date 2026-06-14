"""Language detection, code mapping, and supported-language configuration.

Centralises all language-related constants so other modules import from
here rather than hard-coding BCP-47 tags.
"""

from __future__ import annotations

import logging
import re
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
# TTS speaker per language (Bulbul v3 — all cross-language)
# Names must be valid for BOTH bulbul:v3 surfaces: the REST API and the
# streaming API accept different speaker lists (e.g. "anushka"/"abhilash"
# are streaming-only, "amelia" is v2-only and 400s everywhere). Only use
# names from the intersection of the two lists.
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
    Language.ENGLISH: "ritu",
}

# ---------------------------------------------------------------------------
# Display names (English + native script)
# ---------------------------------------------------------------------------

LANGUAGE_AUTONYMS: dict[Language, str] = {
    Language.HINDI: "हिन्दी",
    Language.TAMIL: "தமிழ்",
    Language.BENGALI: "বাংলা",
    Language.TELUGU: "తెలుగు",
    Language.GUJARATI: "ગુજરાતી",
    Language.KANNADA: "ಕನ್ನಡ",
    Language.MALAYALAM: "മലയാളം",
    Language.MARATHI: "मराठी",
    Language.PUNJABI: "ਪੰਜਾਬੀ",
    Language.ODIA: "ଓଡ଼ିଆ",
    Language.ENGLISH: "English",
}

# Menu-number shortcut so users can reply "2" to pick Tamil, etc.
LANGUAGE_MENU_ORDER: tuple[Language, ...] = (
    Language.HINDI,
    Language.TAMIL,
    Language.BENGALI,
    Language.TELUGU,
    Language.GUJARATI,
    Language.KANNADA,
    Language.MALAYALAM,
    Language.MARATHI,
    Language.PUNJABI,
    Language.ODIA,
    Language.ENGLISH,
)

# Extra lexical keys for text detection (autonyms + common romanisations).
# STT transcribes spoken language names in whatever script matches its
# language guess — "Bengali" said by a Hindi-tagged caller comes back as
# "बंगाली" — so every voice language needs its common cross-script names.
_TEXT_KEYWORDS: dict[str, Language] = {
    "हिन्दी": Language.HINDI,
    "हिंदी": Language.HINDI,
    "হিন্দি": Language.HINDI,
    "হিন্দী": Language.HINDI,
    "தமிழ்": Language.TAMIL,
    "tamizh": Language.TAMIL,
    "तमिल": Language.TAMIL,
    "বাংলা": Language.BENGALI,
    "bangla": Language.BENGALI,
    "बंगाली": Language.BENGALI,
    "बांग्ला": Language.BENGALI,
    "बंगला": Language.BENGALI,
    "বাংলায়": Language.BENGALI,
    "తెలుగు": Language.TELUGU,
    "तेलुगु": Language.TELUGU,
    "तेलगु": Language.TELUGU,
    "ગુજરાતી": Language.GUJARATI,
    "गुजराती": Language.GUJARATI,
    "ಕನ್ನಡ": Language.KANNADA,
    "कन्नड़": Language.KANNADA,
    "कन्नड": Language.KANNADA,
    "മലയാളം": Language.MALAYALAM,
    "मलयालम": Language.MALAYALAM,
    "मराठी": Language.MARATHI,
    "ਪੰਜਾਬੀ": Language.PUNJABI,
    "पंजाबी": Language.PUNJABI,
    "ଓଡ଼ିଆ": Language.ODIA,
    "ଓଡିଆ": Language.ODIA,
    "उड़िया": Language.ODIA,
    "ओड़िया": Language.ODIA,
    "ओडिया": Language.ODIA,
    "अंग्रेजी": Language.ENGLISH,
    "इंग्लिश": Language.ENGLISH,
    "ইংরেজি": Language.ENGLISH,
    "ইংলিশ": Language.ENGLISH,
}


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


# Short acknowledgements and greetings that say nothing reliable about the
# caller's language ("Okay" is universal). STT still tags them with *some*
# language, so locking the session language on one of these mis-routes the
# whole call. Romanized + Devanagari + Bengali cover the common cases.
_FILLER_UTTERANCES: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "oke",
        "okey",
        "k",
        "hmm",
        "hm",
        "huh",
        "ha",
        "haan",
        "han",
        "haanji",
        "haan ji",
        "ji",
        "ji haan",
        "yes",
        "yeah",
        "yep",
        "yo",
        "no",
        "nahi",
        "hello",
        "hallo",
        "helo",
        "hi",
        "hey",
        "namaste",
        "namaskar",
        "theek hai",
        "thik hai",
        "theek",
        "thik",
        "accha",
        "acha",
        "achha",
        "ஓகே",
        "சரி",
        "సరే",
        "ಸರಿ",
        "ശരി",
        "ठीक है",
        "ठीक",
        "अच्छा",
        "हाँ",
        "हां",
        "जी",
        "ओके",
        "हेलो",
        "नमस्ते",
        "ঠিক আছে",
        "আচ্ছা",
        "হ্যাঁ",
        "ওকে",
        "হেলো",
        "নমস্কার",
    }
)


def is_filler_utterance(text: str) -> bool:
    """Return ``True`` for short acknowledgements/greetings ("Okay", "हाँ").

    These carry no reliable language signal, so callers should not lock or
    switch the session language based on them.
    """
    cleaned = "".join(ch for ch in text.strip().lower() if ch not in ".,!?।|'\"-").strip()
    return bool(cleaned) and cleaned in _FILLER_UTTERANCES


def detect_language_from_text(text: str) -> Language | None:
    """Best-effort lexical language match for a short user utterance.

    Used on the text channel's first turn (before any session language has
    been confirmed) to pick the user's chosen language from a menu-style
    response like ``"Tamil"``, ``"tamizh"``, ``"தமிழ்"``, or ``"2"``.

    Returns ``None`` when the input is empty, too long to be a menu answer,
    or doesn't lexically match any known language. In that case callers
    should re-prompt or fall back to Sarvam language detection.
    """
    if not text:
        return None

    cleaned = text.strip().lower()
    if not cleaned:
        return None

    # Pure numeric menu pick, e.g. "2" -> Tamil.
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(LANGUAGE_MENU_ORDER):
            return LANGUAGE_MENU_ORDER[idx]
        return None

    # Short utterances: match the whole string against the known maps.
    if cleaned in SARVAM_LANG_MAP:
        return SARVAM_LANG_MAP[cleaned]

    if text.strip() in _TEXT_KEYWORDS:
        return _TEXT_KEYWORDS[text.strip()]

    # Longer utterances: token-by-token search. Only the first unambiguous
    # hit wins -- if two different languages appear (e.g. "Hindi ya Tamil?")
    # we bail and let the caller re-prompt.
    hits: set[Language] = set()
    for token in cleaned.replace(",", " ").split():
        if token in SARVAM_LANG_MAP:
            hits.add(SARVAM_LANG_MAP[token])
    for autonym, lang in _TEXT_KEYWORDS.items():
        if autonym in text:
            hits.add(lang)

    if len(hits) == 1:
        return next(iter(hits))
    return None


# Unicode script blocks -> spoken language. Saaras returns regional scripts
# reliably even when its language TAG is wrong, so the script the transcript is
# WRITTEN IN is a stronger language signal than the tag for Indic speech.
_SCRIPT_RANGES: tuple[tuple[int, int, Language], ...] = (
    (0x0980, 0x09FF, Language.BENGALI),  # Bengali (also Assamese)
    (0x0900, 0x097F, Language.HINDI),  # Devanagari (Hindi/Marathi -> Hindi)
    (0x0B80, 0x0BFF, Language.TAMIL),
    (0x0C00, 0x0C7F, Language.TELUGU),
    (0x0A80, 0x0AFF, Language.GUJARATI),
    (0x0C80, 0x0CFF, Language.KANNADA),
    (0x0D00, 0x0D7F, Language.MALAYALAM),
    (0x0A00, 0x0A7F, Language.PUNJABI),  # Gurmukhi
    (0x0B00, 0x0B7F, Language.ODIA),
)


def detect_script_language(text: str) -> Language | None:
    """Detect a caller's language from the dominant Indic script in *text*.

    Saaras STT mis-tags short, proper-noun-heavy regional utterances (e.g. a
    Bengali "<place>-e thaki, <place>") as ``en-IN`` while still transcribing
    the words in the correct native script, so the script is a more reliable
    language signal than the STT language tag. Returns ``None`` for empty /
    Latin-only / no-Indic-script input, where the caller falls back to the tag.
    Devanagari maps to Hindi (it is also Marathi's script, but script alone
    cannot disambiguate and Hindi is the safe default for this service).
    """
    if not text:
        return None
    counts: dict[Language, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=lambda lang: counts[lang])


# Unambiguous English content words. Deliberately EXCLUDES tokens that also
# occur in romanized Hindi/Bengali ("me", "to", "a", "in", "hai", "ho") so a
# romanized regional answer is NOT misread as English.
_ENGLISH_MARKERS: frozenset[str] = frozenset(
    {
        "the",
        "is",
        "are",
        "was",
        "have",
        "what",
        "want",
        "need",
        "know",
        "yes",
        "help",
        "from",
        "there",
        "people",
        "live",
        "house",
        "family",
        "insurance",
        "money",
        "month",
        "year",
        "scheme",
        "schemes",
        "treatment",
        "hospital",
        "nothing",
        "everything",
        "please",
        "thanks",
        "thank",
        "anyone",
        "anything",
        "just",
        "about",
    }
)


def looks_like_english(text: str) -> bool:
    """True when *text* is plausibly an English sentence (not romanized Indic).

    Requires at least two unambiguous English marker words so a romanized
    regional answer ("Ami Paschim Banga thaki Howrah") is NOT misread as
    English. Used to gate switching the call TO English on a bare ``en-IN`` STT
    tag, which Saaras emits for many regional utterances it fails to script.
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if len(tokens) < 2:
        return False
    return sum(1 for t in tokens if t in _ENGLISH_MARKERS) >= 2


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
