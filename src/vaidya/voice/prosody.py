"""Small text normalizations that make TTS sound less synthetic."""

from __future__ import annotations

import re

_PROFILE_PACE: dict[str, float] = {
    "default": 0.94,
    "repair": 0.88,
    "distress": 0.85,
    "results": 0.92,
    "processing": 0.94,
}


def format_for_tts(text: str, *, profile: str = "default") -> str:
    """Normalize assistant text into a more speakable TTS string."""
    del profile
    spoken = text.replace("\r", "\n")
    spoken = re.sub(r"\n{2,}", ". ", spoken)
    spoken = re.sub(r"\n", " ", spoken)
    spoken = spoken.replace("...", ". ")
    # Underscores in any leaked internal token (field names, scheme IDs)
    # otherwise get spoken as "underscore". Space them out.
    spoken = spoken.replace("_", " ")
    spoken = re.sub(r"\bRs\.?\s*", "rupees ", spoken)
    spoken = spoken.replace("₹", "rupees ")
    spoken = spoken.replace("PM-JAY", "P M JAY")
    spoken = spoken.replace("AI", "A I")
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken


def pace_for_profile(profile: str, fallback: float) -> float:
    """Return a human-paced Sarvam TTS pace for a response profile."""
    return _PROFILE_PACE.get(profile, fallback)


def temperature_for_profile(profile: str, fallback: float) -> float:
    """Keep voice expressive but stable for important healthcare guidance."""
    if profile in {"repair", "distress", "results", "processing", "default"}:
        return 0.55
    return fallback
