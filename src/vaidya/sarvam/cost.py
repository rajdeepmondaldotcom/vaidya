"""API cost tracking for Sarvam AI usage.

Tracks per-call and per-session costs based on Sarvam's pricing:
- LLM (sarvam-105b/30b): Free
- STT (saaras:v3): Rs 30/hour; (saarika:v2.5): Rs 30/hour
- TTS (bulbul:v3): Rs 30/10K chars; (bulbul:v2): Rs 15/10K chars
- Translation (mayura:v1 / sarvam-translate:v1): Rs 20/10K chars
- Transliteration: Rs 20/10K chars
- Language ID: Rs 3.5/10K chars
- Vision (sarvam-vision): Rs 1.5/page

Docs: https://www.sarvam.ai/api-pricing
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime

logger = logging.getLogger(__name__)

_TIERED_PRICING: dict[str, dict[str, float]] = {
    "stt": {
        "saaras:v3": 30.0 / 3600,
        "saarika:v2.5": 30.0 / 3600,
        "_default": 30.0 / 3600,  # Rs 30/hour
    },
    "tts": {
        "bulbul:v3": 30.0 / 10_000,  # Rs 30/10K chars
        "bulbul:v2": 15.0 / 10_000,  # Rs 15/10K chars
        "_default": 30.0 / 10_000,
    },
    "translate": {
        "mayura:v1": 20.0 / 10_000,
        "sarvam-translate:v1": 20.0 / 10_000,
        "_default": 20.0 / 10_000,  # Rs 20/10K chars
    },
    "transliterate": {"_default": 20.0 / 10_000},
    "language_id": {"_default": 3.5 / 10_000},
    "vision": {"sarvam-vision": 1.5, "_default": 1.5},  # Rs 1.5/page
    "llm": {"sarvam-105b": 0.0, "sarvam-30b": 0.0, "_default": 0.0},
}


def _get_rate(service: str, model: str = "") -> float:
    """Resolve the per-unit INR rate for *service* and optional *model*."""
    tier = _TIERED_PRICING.get(service, {})
    return tier.get(model, tier.get("_default", 0.0))


_SERVICE_UNIT_TYPES: dict[str, str] = {
    "stt": "seconds",
    "llm": "tokens",
    "translate": "chars",
    "tts": "chars",
    "transliterate": "chars",
    "language_id": "chars",
    "vision": "pages",
}


@dataclass
class CostAlertConfig:
    """Thresholds that trigger logger.warning when exceeded."""

    per_call_threshold_inr: float = 50.0
    per_session_threshold_inr: float = 200.0


@dataclass
class CostEntry:
    """A single API cost event."""

    service: str
    units: float
    cost_inr: float
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    model: str = ""


@dataclass
class CostTracker:
    """Tracks cumulative API costs across all calls with tiered pricing."""

    entries: list[CostEntry] = field(default_factory=list)
    alert_config: CostAlertConfig = field(default_factory=CostAlertConfig)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_llm(
        self,
        tokens: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        """Record an LLM call (free, but tracked for monitoring)."""
        cost = tokens * _get_rate("llm", model)
        self._add("llm", tokens, cost, call_id, latency_ms, model)

    def record_stt(
        self,
        duration_seconds: float,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "saaras:v3",
    ) -> None:
        """Record STT usage (charged per audio-second)."""
        cost = duration_seconds * _get_rate("stt", model)
        self._add("stt", duration_seconds, cost, call_id, latency_ms, model)

    def record_tts(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "bulbul:v3",
    ) -> None:
        """Record TTS usage (charged per character)."""
        cost = char_count * _get_rate("tts", model)
        self._add("tts", char_count, cost, call_id, latency_ms, model)

    def record_translate(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "mayura:v1",
    ) -> None:
        """Record translation usage (charged per character)."""
        cost = char_count * _get_rate("translate", model)
        self._add("translate", char_count, cost, call_id, latency_ms, model)

    def record_transliterate(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        """Record transliteration usage (charged per character)."""
        cost = char_count * _get_rate("transliterate", model)
        self._add("transliterate", char_count, cost, call_id, latency_ms, model)

    def record_language_id(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        """Record language identification usage (charged per character)."""
        cost = char_count * _get_rate("language_id", model)
        self._add("language_id", char_count, cost, call_id, latency_ms, model)

    def record_vision(
        self,
        pages: int = 1,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "sarvam-vision",
    ) -> None:
        """Record vision/document intelligence usage (charged per page)."""
        cost = pages * _get_rate("vision", model)
        self._add("vision", pages, cost, call_id, latency_ms, model)

    def _snapshot_entries(self) -> list[CostEntry]:
        """Return a shallow copy of entries under the lock."""
        with self._lock:
            return list(self.entries)

    @staticmethod
    def _aggregate_by_service(entries: list[CostEntry]) -> dict[str, float]:
        """Aggregate costs by service name from a list of entries."""
        totals: dict[str, float] = {}
        for e in entries:
            totals[e.service] = totals.get(e.service, 0) + e.cost_inr
        return totals

    @property
    def total_cost_inr(self) -> float:
        """Total cost across all entries."""
        with self._lock:
            return sum(e.cost_inr for e in self.entries)

    @property
    def total_by_service(self) -> dict[str, float]:
        """Cost breakdown by service."""
        return self._aggregate_by_service(self._snapshot_entries())

    def cost_for_call(self, call_id: str) -> float:
        """Total cost for a specific call."""
        with self._lock:
            return sum(e.cost_inr for e in self.entries if e.call_id == call_id)

    @staticmethod
    def _build_service_breakdown(entries: list[CostEntry]) -> dict[str, dict]:
        """Aggregate entries into per-service cost/unit breakdowns."""
        by_service: dict[str, dict] = {}
        for e in entries:
            if e.service not in by_service:
                by_service[e.service] = {
                    "cost_inr": 0.0,
                    "units": 0.0,
                    "unit_type": _SERVICE_UNIT_TYPES.get(e.service, "units"),
                }
            svc = by_service[e.service]
            svc["cost_inr"] = round(svc["cost_inr"] + e.cost_inr, 4)
            svc["units"] = round(svc["units"] + e.units, 4)
        return by_service

    def breakdown_for_call(self, call_id: str) -> dict:
        """Return a detailed cost breakdown for a specific call."""
        with self._lock:
            call_entries = [e for e in self.entries if e.call_id == call_id]

        return {
            "call_id": call_id,
            "total_inr": round(sum(e.cost_inr for e in call_entries), 4),
            "by_service": self._build_service_breakdown(call_entries),
            "api_call_count": len(call_entries),
        }

    def summary(self) -> dict:
        """Full cost summary."""
        snapshot = self._snapshot_entries()
        by_service = self._aggregate_by_service(snapshot)
        call_ids = {e.call_id for e in snapshot if e.call_id}
        call_count = len(call_ids)
        total = sum(e.cost_inr for e in snapshot)
        avg_per_call = total / call_count if call_count > 0 else 0.0

        return {
            "total_inr": round(total, 4),
            "by_service": {k: round(v, 4) for k, v in by_service.items()},
            "call_count": call_count,
            "api_calls": len(snapshot),
            "avg_cost_per_call_inr": round(avg_per_call, 4),
        }

    def _per_call_summaries(self, snapshot: list[CostEntry]) -> list[dict]:
        """Build per-call breakdown dicts from a snapshot."""
        call_ids = sorted({e.call_id for e in snapshot if e.call_id})
        result: list[dict] = []
        for cid in call_ids:
            call_entries = [e for e in snapshot if e.call_id == cid]
            call_by_service = self._aggregate_by_service(call_entries)
            result.append(
                {
                    "call_id": cid,
                    "total_inr": round(sum(e.cost_inr for e in call_entries), 4),
                    "by_service": {k: round(v, 4) for k, v in call_by_service.items()},
                    "api_calls": len(call_entries),
                }
            )
        return result

    @staticmethod
    def _cost_projections(avg_cost_per_call: float) -> dict[str, float]:
        """Project costs at various daily call volumes."""
        daily_100 = round(avg_cost_per_call * 100, 2)
        return {
            "daily_100_calls_inr": daily_100,
            "monthly_100_calls_per_day_inr": round(daily_100 * 30, 2),
            "monthly_10k_calls_per_day_inr": round(avg_cost_per_call * 10_000 * 30, 2),
        }

    def detailed_summary(self) -> dict:
        """Extended summary with per-call breakdown and projections."""
        base = self.summary()
        snapshot = self._snapshot_entries()
        base["per_call"] = self._per_call_summaries(snapshot)
        base["projections"] = self._cost_projections(base["avg_cost_per_call_inr"])
        return base

    def daily_summary(self, day: date) -> dict:
        """Aggregate costs for a specific calendar day."""
        snapshot = self._snapshot_entries()
        day_entries = [e for e in snapshot if datetime.fromtimestamp(e.timestamp).date() == day]
        by_service = self._aggregate_by_service(day_entries)
        total = sum(e.cost_inr for e in day_entries)
        return {
            "date": day.isoformat(),
            "total_inr": round(total, 4),
            "by_service": {k: round(v, 4) for k, v in by_service.items()},
            "api_calls": len(day_entries),
        }

    def monthly_summary(self, year: int, month: int) -> dict:
        """Aggregate costs for a specific calendar month."""
        snapshot = self._snapshot_entries()
        month_entries = [
            e
            for e in snapshot
            if (dt := datetime.fromtimestamp(e.timestamp))
            and dt.year == year
            and dt.month == month
        ]
        by_service = self._aggregate_by_service(month_entries)
        total = sum(e.cost_inr for e in month_entries)
        return {
            "year": year,
            "month": month,
            "total_inr": round(total, 4),
            "by_service": {k: round(v, 4) for k, v in by_service.items()},
            "api_calls": len(month_entries),
        }

    def check_alerts(self, call_id: str = "") -> list[str]:
        """Return warning messages if any cost thresholds are exceeded."""
        warnings: list[str] = []
        if call_id:
            call_cost = self.cost_for_call(call_id)
            if call_cost > self.alert_config.per_call_threshold_inr:
                warnings.append(
                    f"Call {call_id} cost Rs {call_cost:.2f} exceeds "
                    f"threshold Rs {self.alert_config.per_call_threshold_inr:.2f}"
                )
        if self.total_cost_inr > self.alert_config.per_session_threshold_inr:
            warnings.append(
                f"Total session cost Rs {self.total_cost_inr:.2f} exceeds "
                f"threshold Rs {self.alert_config.per_session_threshold_inr:.2f}"
            )
        return warnings

    def _add(
        self,
        service: str,
        units: float,
        cost: float,
        call_id: str,
        latency_ms: float = 0.0,
        model: str = "",
    ) -> None:
        with self._lock:
            self.entries.append(
                CostEntry(
                    service=service,
                    units=units,
                    cost_inr=cost,
                    call_id=call_id,
                    latency_ms=latency_ms,
                    model=model,
                )
            )
        if call_id:
            for w in self.check_alerts(call_id):
                logger.warning("Cost alert: %s", w)
