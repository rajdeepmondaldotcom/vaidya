"""Model constants and routing configuration for Sarvam AI services.

Updated to latest Sarvam API models as of April 2026.
Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LLM Models (Chat Completions)
# Endpoint: POST https://api.sarvam.ai/v1/chat/completions
# Docs: https://docs.sarvam.ai/api-reference-docs/chat/chat-completions
# ---------------------------------------------------------------------------

# Flagship: 105B MoE, 128K context, best for complex reasoning + eligibility
SARVAM_105B = "sarvam-105b"

# Efficient: 30B MoE, 64K context, good for intake/guidance/routing
SARVAM_30B = "sarvam-30b"

# Default model for agents (use 105B for best accuracy)
DEFAULT_LLM_MODEL = SARVAM_105B

# ---------------------------------------------------------------------------
# Speech-to-Text (STT)
# Endpoint: POST https://api.sarvam.ai/v1/speech-to-text/transcribe
# Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
# ---------------------------------------------------------------------------

# Saaras v3: 23 languages, modes: transcribe/translate/verbatim/translit/codemix
STT_MODEL = "saaras:v3"

# ---------------------------------------------------------------------------
# Text-to-Speech (TTS)
# Endpoint: POST https://api.sarvam.ai/v1/text-to-speech/convert
# Docs: https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
# ---------------------------------------------------------------------------

# Bulbul v3: 45 speakers, temperature control, 11 languages, 2500 char limit
TTS_MODEL = "bulbul:v3"

# ---------------------------------------------------------------------------
# Translation
# Endpoint: POST https://api.sarvam.ai/v1/translate
# Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/mayura
# ---------------------------------------------------------------------------

# Colloquial/code-mixed: 11 languages, 4 styles
TRANSLATE_MODEL_MAYURA = "mayura:v1"

# Formal/structured: 23 languages
TRANSLATE_MODEL_SARVAM = "sarvam-translate:v1"

# Default for voice conversations (modern-colloquial style)
DEFAULT_TRANSLATE_MODEL = TRANSLATE_MODEL_MAYURA

# ---------------------------------------------------------------------------
# Vision (Document Intelligence)
# Endpoint: POST https://api.sarvam.ai/v1/document/analyze
# Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-vision
# Pricing: ₹1.5 per page
# ---------------------------------------------------------------------------

VISION_MODEL = "sarvam-vision"

# ---------------------------------------------------------------------------
# Language support — canonical enum lives in voice.language, re-exported here
# ---------------------------------------------------------------------------

from vaidya.voice.language import (  # noqa: E402, I001
    Language,
    TTS_SPEAKERS as TTS_SPEAKER,  # noqa: F401
)

# BCP-47 codes required by Sarvam APIs (same as Language enum values)
LANGUAGE_TO_SARVAM_CODE: dict[Language, str] = {lang: lang.value for lang in Language}

__all__ = [
    "DEFAULT_LLM_MODEL",
    "DEFAULT_TRANSLATE_MODEL",
    "LANGUAGE_TO_SARVAM_CODE",
    "Language",
    "SARVAM_105B",
    "SARVAM_30B",
    "STT_MODEL",
    "TTS_MODEL",
    "TTS_MAX_CHARS_V3",
]

# STT modes for saaras:v3
STT_MODE_TRANSCRIBE = "transcribe"
STT_MODE_TRANSLATE = "translate"
STT_MODE_CODEMIX = "codemix"
STT_MODE_VERBATIM = "verbatim"
STT_MODE_TRANSLIT = "translit"

# ---------------------------------------------------------------------------
# Reasoning effort levels (for chat completions)
# ---------------------------------------------------------------------------

REASONING_EFFORT_HIGH = "high"
REASONING_EFFORT_MEDIUM = "medium"
REASONING_EFFORT_LOW = "low"

# ---------------------------------------------------------------------------
# TTS settings (bulbul:v3)
# ---------------------------------------------------------------------------

TTS_MAX_CHARS_V3 = 2500  # bulbul:v3 supports 2500 chars (v2 was 1500)
TTS_MAX_CHARS_V2 = 1500
TTS_SAMPLE_RATE_TELEPHONY = 8000  # for phone calls
TTS_SAMPLE_RATE_DEFAULT = 24000  # for demo/web
TTS_DEFAULT_TEMPERATURE = 0.6  # expressiveness (v3 only)
TTS_DEFAULT_PACE = 1.0  # speech speed

# ---------------------------------------------------------------------------
# Translation settings
# ---------------------------------------------------------------------------

TRANSLATE_MODE_FORMAL = "formal"
TRANSLATE_MODE_COLLOQUIAL = "modern-colloquial"
TRANSLATE_MODE_CLASSIC = "classic-colloquial"
TRANSLATE_MODE_CODEMIXED = "code-mixed"

TRANSLATE_MAX_CHARS_MAYURA = 1000  # mayura:v1
TRANSLATE_MAX_CHARS_SARVAM = 2000  # sarvam-translate:v1


def get_translate_model(lang_code: str) -> str:
    """Return the appropriate translate model for a language code.

    Mayura v1 supports 11 TTS languages (colloquial/spoken style).
    Sarvam Translate v1 supports all 23 scheduled languages (formal style).
    """
    from vaidya.voice.language import is_voice_language

    if is_voice_language(lang_code):
        return TRANSLATE_MODEL_MAYURA
    return TRANSLATE_MODEL_SARVAM


# ---------------------------------------------------------------------------
# Audio format support
# ---------------------------------------------------------------------------

TTS_SUPPORTED_CODECS = ["wav", "mp3", "linear16", "mulaw", "alaw", "opus", "flac", "aac"]
STT_SUPPORTED_FORMATS = ["wav", "mp3", "aac", "flac", "ogg"]
