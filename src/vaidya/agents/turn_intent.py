"""Deterministic turn-intent classification for voice repair UX."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from vaidya.models.conversation import ConversationPhase

TurnAction = Literal[
    "continue",
    "repeat",
    "restart",
    "end",
    "wait",
    "side_question",
    "correction",
    "low_confidence",
]

_TEXT_LOW_CONFIDENCE = 0.65
# Phone STT confidence dips on accented/8 kHz/noisy speech even for perfectly
# usable answers, and blocking the turn with "audio cut, I'll re-ask" on every
# such dip stalls the call. Keep voice repair for only the genuinely garbled
# (very low confidence); the downstream intake LLM is robust to imperfect
# transcripts and the confirmation step catches any real mis-read.
_VOICE_LOW_CONFIDENCE = 0.30

_REPEAT_PATTERNS = (
    "repeat",
    "say again",
    "again please",
    "what did you say",
    "did not understand",
    "didn't understand",
    "samajh nahi",
    "samajh nahin",
    "kya bola",
    "phir se",
    "dobara",
    "fir se",
)
_RESTART_PATTERNS = (
    "restart",
    "start over",
    "start again",
    "shuru se",
    "phir se shuru",
    "fir se shuru",
    "naya shuru",
    "dobara shuru",
)
_END_PATTERNS = (
    "bye",
    "goodbye",
    "end call",
    "stop call",
    "hang up",
    "call cut",
    "band karo",
    "khatam karo",
    "nahi chahiye",
    "nahin chahiye",
)
_WAIT_PATTERNS = (
    "wait",
    "hold on",
    "one minute",
    "ek minute",
    "ruk",
    "rukiye",
    "ruk jaiye",
    "sochne do",
    "thoda time",
)
_SIDE_QUESTION_PATTERNS = (
    "why",
    "kyun",
    "kisliye",
    "safe",
    "privacy",
    "data",
    "share",
    "aadhaar kyun",
    "income kyun",
    "kaise use",
)
_CORRECTION_PATTERNS = (
    "wrong",
    "incorrect",
    "galat",
    "actually",
    "correction",
    "sudhar",
    "badal",
)
_FILLER_WORDS = frozenset({"hmm", "um", "umm", "uh", "haan?", "hello?"})


@dataclass(frozen=True)
class TurnIntent:
    """A pre-route action detected from the user's utterance."""

    action: TurnAction
    repair_type: str = ""
    metadata: dict[str, str | float | bool] = field(default_factory=dict)


def classify_turn_intent(
    user_input: str,
    *,
    phase: ConversationPhase,
    stt_confidence: float,
    channel: str,
) -> TurnIntent:
    """Classify common repair intents without an LLM call.

    Commands are checked before STT confidence so a noisy but clear "repeat"
    still routes correctly. Voice low-confidence repair is conservative
    because phone STT confidence can be noisy even for usable utterances.
    """

    text = _normalize(user_input)
    if not text:
        return TurnIntent("continue")

    if _has_pattern(text, _END_PATTERNS):
        return TurnIntent("end", "end")
    if _has_pattern(text, _RESTART_PATTERNS):
        return TurnIntent("restart", "restart")
    if _has_pattern(text, _REPEAT_PATTERNS):
        return TurnIntent("repeat", "repeat")
    if _has_pattern(text, _WAIT_PATTERNS):
        return TurnIntent("wait", "wait")

    if phase == ConversationPhase.INTAKE:
        if _looks_like_side_question(text):
            return TurnIntent("side_question", "side_question")
        if _has_pattern(text, _CORRECTION_PATTERNS):
            return TurnIntent("correction", "correction")

    threshold = _VOICE_LOW_CONFIDENCE if channel == "voice" else _TEXT_LOW_CONFIDENCE
    if stt_confidence < threshold or text in _FILLER_WORDS:
        return TurnIntent(
            "low_confidence",
            "low_confidence",
            {"stt_confidence": round(stt_confidence, 3)},
        )

    return TurnIntent("continue")


def _normalize(text: str) -> str:
    """Lowercase and collapse non-word separators while preserving spaces."""

    cleaned = re.sub(r"[^\w\s?]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _looks_like_side_question(text: str) -> bool:
    if "?" in text and any(word in text for word in ("why", "kyun", "kaise", "what")):
        return True
    return _has_pattern(text, _SIDE_QUESTION_PATTERNS)
