"""Model constants and routing configuration for Sarvam AI services.

Updated to latest Sarvam API models as of April 2026.
Docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models
"""

from __future__ import annotations

from enum import StrEnum

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
# Language support
# ---------------------------------------------------------------------------


class Language(StrEnum):
    """Languages supported by Vaidya voice calls (Phase 1)."""

    HINDI = "hi"
    TAMIL = "ta"
    BENGALI = "bn"
    ENGLISH = "en"


# BCP-47 codes required by Sarvam APIs
LANGUAGE_TO_SARVAM_CODE: dict[Language, str] = {
    Language.HINDI: "hi-IN",
    Language.TAMIL: "ta-IN",
    Language.BENGALI: "bn-IN",
    Language.ENGLISH: "en-IN",
}

# TTS speakers — bulbul:v3 has 45 voices, pick natural-sounding per language
TTS_SPEAKER: dict[Language, str] = {
    Language.HINDI: "priya",
    Language.TAMIL: "kavitha",
    Language.BENGALI: "priya",
    Language.ENGLISH: "amelia",
}

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
