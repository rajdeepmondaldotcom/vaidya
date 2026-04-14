"""API cost tracking for Sarvam AI usage.

Tracks per-call and per-session costs based on Sarvam's pricing:
- LLM (sarvam-105b/30b): Free
- STT (saaras:v3): Rs 30/hour
- TTS (bulbul:v3): Rs 30/10K chars
- Translation (mayura:v1): Rs 20/10K chars
- Transliteration: Rs 20/10K chars
- Language ID: Rs 3.5/10K chars
- Vision: Rs 1.5/page
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing in INR per unit
_PRICING = {
    "stt_per_second": 30.0 / 3600,  # Rs 30/hour
    "tts_per_char": 30.0 / 10_000,  # Rs 30/10K chars
    "translate_per_char": 20.0 / 10_000,  # Rs 20/10K chars
    "transliterate_per_char": 20.0 / 10_000,  # Rs 20/10K chars
    "language_id_per_char": 3.5 / 10_000,  # Rs 3.5/10K chars
    "vision_per_page": 1.5,  # Rs 1.5/page
    "llm_per_token": 0.0,  # Free
}


@dataclass
class CostEntry:
    """A single API cost event."""

    service: str
    units: float
    cost_inr: float
    call_id: str = ""


@dataclass
class CostTracker:
    """Tracks cumulative API costs across all calls."""

    entries: list[CostEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_llm(self, tokens: int, call_id: str = "") -> None:
        """Record an LLM call (free)."""
        self._add("llm", tokens, tokens * _PRICING["llm_per_token"], call_id)

    def record_stt(self, duration_seconds: float, call_id: str = "") -> None:
        """Record STT usage."""
        cost = duration_seconds * _PRICING["stt_per_second"]
        self._add("stt", duration_seconds, cost, call_id)

    def record_tts(self, char_count: int, call_id: str = "") -> None:
        """Record TTS usage."""
        cost = char_count * _PRICING["tts_per_char"]
        self._add("tts", char_count, cost, call_id)

    def record_translate(self, char_count: int, call_id: str = "") -> None:
        """Record translation usage."""
        cost = char_count * _PRICING["translate_per_char"]
        self._add("translate", char_count, cost, call_id)

    def record_transliterate(self, char_count: int, call_id: str = "") -> None:
        """Record transliteration usage."""
        cost = char_count * _PRICING["transliterate_per_char"]
        self._add("transliterate", char_count, cost, call_id)

    def record_language_id(self, char_count: int, call_id: str = "") -> None:
        """Record language identification usage."""
        cost = char_count * _PRICING["language_id_per_char"]
        self._add("language_id", char_count, cost, call_id)

    def record_vision(self, pages: int = 1, call_id: str = "") -> None:
        """Record vision/document intelligence usage."""
        cost = pages * _PRICING["vision_per_page"]
        self._add("vision", pages, cost, call_id)

    @property
    def total_cost_inr(self) -> float:
        """Total cost across all entries."""
        with self._lock:
            return sum(e.cost_inr for e in self.entries)

    @property
    def total_by_service(self) -> dict[str, float]:
        """Cost breakdown by service."""
        totals: dict[str, float] = {}
        with self._lock:
            for e in self.entries:
                totals[e.service] = totals.get(e.service, 0) + e.cost_inr
        return totals

    def cost_for_call(self, call_id: str) -> float:
        """Total cost for a specific call."""
        with self._lock:
            return sum(e.cost_inr for e in self.entries if e.call_id == call_id)

    def summary(self) -> dict:
        """Full cost summary."""
        by_service = self.total_by_service
        return {
            "total_inr": round(self.total_cost_inr, 4),
            "by_service": {k: round(v, 4) for k, v in by_service.items()},
            "call_count": len({e.call_id for e in self.entries if e.call_id}),
            "api_calls": len(self.entries),
        }

    def _add(self, service: str, units: float, cost: float, call_id: str) -> None:
        with self._lock:
            self.entries.append(CostEntry(service, units, cost, call_id))
