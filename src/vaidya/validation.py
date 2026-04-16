"""Input validation helpers for language and state."""

from __future__ import annotations

from vaidya.utils.states import state_name_to_code
from vaidya.voice.language import Language, TextLanguage, is_supported, normalize_language


class ValidationError(ValueError):
    """Raised when input validation fails."""


def validate_language(lang: str, channel: str = "voice") -> str:
    """Normalize and validate a language code.

    For ``voice`` channel, only the 11 TTS languages are accepted.
    For ``text`` channels (whatsapp, sms, web), all 23 scheduled
    languages are accepted.

    Returns the normalized BCP-47 language code.

    Raises
    ------
    ValidationError
        If the language is not supported for the given channel.
    """
    # Check voice languages first (11 TTS)
    if is_supported(lang):
        return normalize_language(lang).value

    # Check text-only languages (12 additional)
    lower = lang.lower().strip()
    for tl in TextLanguage:
        code = tl.value.lower()
        short = code.split("-")[0]
        if lower in (code, short, tl.name.lower()):
            if channel == "voice":
                raise ValidationError(
                    f"Language '{lang}' is only supported on text channels "
                    f"(whatsapp, sms, web), not voice. "
                    f"Supported voice languages: {', '.join(lang_.value for lang_ in Language)}"
                )
            return tl.value

    raise ValidationError(
        f"Language '{lang}' is not supported. "
        f"Supported languages: {', '.join(lang_.value for lang_ in Language)}, "
        f"{', '.join(tl.value for tl in TextLanguage)}"
    )


def validate_state(state: str) -> str:
    """Normalize and validate an Indian state/UT name.

    Returns the 2-letter state code.

    Raises
    ------
    ValidationError
        If the state cannot be resolved.
    """
    code = state_name_to_code(state)
    if code is None:
        raise ValidationError(
            f"State '{state}' not recognized. "
            "Please provide a valid Indian state or union territory name."
        )
    return code
