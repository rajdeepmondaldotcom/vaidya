"""API cost tracking for Sarvam AI and optional telephony usage.

Tracks per-call and per-session costs based on Sarvam's pricing:
- LLM (sarvam-105b/30b): Free
- STT (saaras:v3): Rs 30/hour
- STT with diarization: Rs 45/hour
- TTS (bulbul:v3): Rs 30/10K chars; (bulbul:v2): Rs 15/10K chars
- Translation (mayura:v1 / sarvam-translate:v1): Rs 20/10K chars
- Transliteration: Rs 20/10K chars
- Language ID: Rs 3.5/10K chars
- Vision (sarvam-vision): Rs 1.5/page

Docs: https://docs.sarvam.ai/api-reference-docs/pricing
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

_STT_STANDARD_RATE_PER_SECOND = 30.0 / 3600
_STT_DIARIZATION_RATE_PER_SECOND = 45.0 / 3600

_TIERED_PRICING: dict[str, dict[str, float]] = {
    "stt": {
        "saaras:v3": _STT_STANDARD_RATE_PER_SECOND,
        "saarika:v2.5": _STT_STANDARD_RATE_PER_SECOND,
        "_diarization": _STT_DIARIZATION_RATE_PER_SECOND,
        "_default": _STT_STANDARD_RATE_PER_SECOND,
    },
    "tts": {
        "bulbul:v3": 30.0 / 10_000,
        "bulbul:v2": 15.0 / 10_000,
        "_default": 30.0 / 10_000,
    },
    "translate": {
        "mayura:v1": 20.0 / 10_000,
        "sarvam-translate:v1": 20.0 / 10_000,
        "_default": 20.0 / 10_000,
    },
    "transliterate": {"_default": 20.0 / 10_000},
    "language_id": {"_default": 3.5 / 10_000},
    "vision": {"sarvam-vision": 1.5, "_default": 1.5},
    "llm": {"sarvam-105b": 0.0, "sarvam-30b": 0.0, "_default": 0.0},
    "telephony": {"_default": 0.0},
}


def _get_rate(
    service: str,
    model: str = "",
    *,
    with_diarization: bool = False,
    explicit_rate: float | None = None,
) -> float:
    """Resolve the per-unit INR rate for a service/model/feature combination."""
    if explicit_rate is not None:
        return explicit_rate
    if service == "stt" and with_diarization:
        return _TIERED_PRICING["stt"]["_diarization"]
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
    "telephony": "minutes",
}

_SERVICE_BILLABLE_UNIT_TYPES: dict[str, str] = {
    **_SERVICE_UNIT_TYPES,
    "stt": "billable_seconds",
    "telephony": "billable_minutes",
}


@dataclass
class CostAlertConfig:
    """Thresholds that trigger logger.warning when exceeded."""

    per_call_threshold_inr: float = 50.0
    per_session_threshold_inr: float = 200.0


@dataclass
class CostEntry:
    """A single API cost event with enough detail to audit the total."""

    service: str
    units: float
    cost_inr: float
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    model: str = ""
    mode: str = ""
    billable_units: float = 0.0
    rate_inr_per_unit: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.billable_units == 0.0 and self.units:
            self.billable_units = self.units


@dataclass
class CostTracker:
    """Tracks cumulative API costs across calls with model/mode-aware pricing."""

    entries: list[CostEntry] = field(default_factory=list)
    alert_config: CostAlertConfig = field(default_factory=CostAlertConfig)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_llm(
        self,
        tokens: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
        mode: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an LLM call. Sarvam chat models are currently free."""
        billable = max(tokens, 0)
        rate = _get_rate("llm", model)
        self._add(
            "llm",
            tokens,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_stt(
        self,
        duration_seconds: float,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "saaras:v3",
        mode: str = "transcribe",
        with_diarization: bool = False,
        billable_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record STT usage, billed per audio second rounded up per request/session."""
        billable = (
            billable_seconds
            if billable_seconds is not None
            else float(math.ceil(max(duration_seconds, 0.0)))
        )
        rate = _get_rate("stt", model, with_diarization=with_diarization)
        self._add(
            "stt",
            duration_seconds,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata={"with_diarization": with_diarization, **(metadata or {})},
        )

    def record_tts(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "bulbul:v3",
        mode: str = "rest",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record TTS usage, billed per character."""
        billable = max(char_count, 0)
        rate = _get_rate("tts", model)
        self._add(
            "tts",
            char_count,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_translate(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "mayura:v1",
        mode: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record translation usage, billed per character."""
        billable = max(char_count, 0)
        rate = _get_rate("translate", model)
        self._add(
            "translate",
            char_count,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_transliterate(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
        mode: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record transliteration usage, billed per character."""
        billable = max(char_count, 0)
        rate = _get_rate("transliterate", model)
        self._add(
            "transliterate",
            char_count,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_language_id(
        self,
        char_count: int,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "",
        mode: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record language identification usage, billed per character."""
        billable = max(char_count, 0)
        rate = _get_rate("language_id", model)
        self._add(
            "language_id",
            char_count,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_vision(
        self,
        pages: int = 1,
        call_id: str = "",
        latency_ms: float = 0.0,
        model: str = "sarvam-vision",
        mode: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record vision/document intelligence usage, billed per page."""
        billable = max(pages, 0)
        rate = _get_rate("vision", model)
        self._add(
            "vision",
            pages,
            billable * rate,
            call_id,
            latency_ms,
            model,
            mode=mode,
            billable_units=billable,
            rate_inr_per_unit=rate,
            metadata=metadata,
        )

    def record_telephony(
        self,
        duration_seconds: float,
        *,
        rate_per_minute_inr: float,
        call_id: str = "",
        provider: str = "",
        mode: str = "voice",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record provider telephony cost when a deployment rate is configured."""
        if rate_per_minute_inr <= 0 or duration_seconds <= 0:
            return
        billable_minutes = float(math.ceil(duration_seconds / 60))
        self._add(
            "telephony",
            duration_seconds / 60,
            billable_minutes * rate_per_minute_inr,
            call_id,
            0.0,
            provider,
            mode=mode,
            billable_units=billable_minutes,
            rate_inr_per_unit=rate_per_minute_inr,
            metadata=metadata,
        )

    def _snapshot_entries(self) -> list[CostEntry]:
        """Return a shallow copy of entries under the lock."""
        with self._lock:
            return list(self.entries)

    @staticmethod
    def _aggregate_by_service(entries: list[CostEntry]) -> dict[str, float]:
        """Aggregate costs by service name from a list of entries."""
        totals: dict[str, float] = {}
        for e in entries:
            totals[e.service] = totals.get(e.service, 0.0) + e.cost_inr
        return totals

    @staticmethod
    def _aggregate_by_dimension(entries: list[CostEntry], attr: str) -> dict[str, float]:
        """Aggregate costs by a string-like CostEntry attribute."""
        totals: dict[str, float] = {}
        for e in entries:
            key = str(getattr(e, attr) or "default")
            totals[key] = totals.get(key, 0.0) + e.cost_inr
        return totals

    @staticmethod
    def _aggregate_by_service_model_mode(entries: list[CostEntry]) -> dict[str, float]:
        """Aggregate costs by service/model/mode to expose mixed-mode usage."""
        totals: dict[str, float] = {}
        for e in entries:
            key = f"{e.service}:{e.model or 'default'}:{e.mode or 'default'}"
            totals[key] = totals.get(key, 0.0) + e.cost_inr
        return totals

    @staticmethod
    def _round_money_map(values: dict[str, float]) -> dict[str, float]:
        return {k: round(v, 4) for k, v in values.items()}

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
    def _build_service_breakdown(entries: list[CostEntry]) -> dict[str, dict[str, Any]]:
        """Aggregate entries into per-service cost/unit breakdowns."""
        by_service: dict[str, dict[str, Any]] = {}
        for e in entries:
            if e.service not in by_service:
                by_service[e.service] = {
                    "cost_inr": 0.0,
                    "units": 0.0,
                    "unit_type": _SERVICE_UNIT_TYPES.get(e.service, "units"),
                    "billable_units": 0.0,
                    "billable_unit_type": _SERVICE_BILLABLE_UNIT_TYPES.get(
                        e.service,
                        _SERVICE_UNIT_TYPES.get(e.service, "units"),
                    ),
                    "by_model": {},
                    "by_mode": {},
                }
            svc = by_service[e.service]
            svc["cost_inr"] = round(svc["cost_inr"] + e.cost_inr, 4)
            svc["units"] = round(svc["units"] + e.units, 4)
            svc["billable_units"] = round(svc["billable_units"] + e.billable_units, 4)
            model = e.model or "default"
            mode = e.mode or "default"
            svc["by_model"][model] = round(svc["by_model"].get(model, 0.0) + e.cost_inr, 4)
            svc["by_mode"][mode] = round(svc["by_mode"].get(mode, 0.0) + e.cost_inr, 4)
        return by_service

    @staticmethod
    def _entry_dict(entry: CostEntry) -> dict[str, Any]:
        """Serialize one entry with the full calculation details."""
        return {
            "service": entry.service,
            "model": entry.model,
            "mode": entry.mode,
            "units": round(entry.units, 4),
            "unit_type": _SERVICE_UNIT_TYPES.get(entry.service, "units"),
            "billable_units": round(entry.billable_units, 4),
            "billable_unit_type": _SERVICE_BILLABLE_UNIT_TYPES.get(
                entry.service,
                _SERVICE_UNIT_TYPES.get(entry.service, "units"),
            ),
            "rate_inr_per_unit": entry.rate_inr_per_unit,
            "cost_inr": round(entry.cost_inr, 4),
            "latency_ms": round(entry.latency_ms, 1),
            "timestamp": datetime.fromtimestamp(entry.timestamp).isoformat(),
            "metadata": entry.metadata,
        }

    def breakdown_for_call(self, call_id: str) -> dict[str, Any]:
        """Return a detailed cost breakdown for a specific call."""
        with self._lock:
            call_entries = [e for e in self.entries if e.call_id == call_id]

        return {
            "call_id": call_id,
            "total_inr": round(sum(e.cost_inr for e in call_entries), 4),
            "by_service": self._build_service_breakdown(call_entries),
            "api_call_count": len(call_entries),
            "entries": [self._entry_dict(e) for e in call_entries],
        }

    def summary(self) -> dict[str, Any]:
        """Full cost summary."""
        snapshot = self._snapshot_entries()
        by_service = self._aggregate_by_service(snapshot)
        call_ids = {e.call_id for e in snapshot if e.call_id}
        call_count = len(call_ids)
        total = sum(e.cost_inr for e in snapshot)
        avg_per_call = total / call_count if call_count > 0 else 0.0

        return {
            "total_inr": round(total, 4),
            "by_service": self._round_money_map(by_service),
            "by_model": self._round_money_map(self._aggregate_by_dimension(snapshot, "model")),
            "by_mode": self._round_money_map(self._aggregate_by_dimension(snapshot, "mode")),
            "by_service_model_mode": self._round_money_map(
                self._aggregate_by_service_model_mode(snapshot)
            ),
            "call_count": call_count,
            "api_calls": len(snapshot),
            "avg_cost_per_call_inr": round(avg_per_call, 4),
        }

    def _per_call_summaries(self, snapshot: list[CostEntry]) -> list[dict[str, Any]]:
        """Build per-call breakdown dicts from a snapshot."""
        call_ids = sorted({e.call_id for e in snapshot if e.call_id})
        result: list[dict[str, Any]] = []
        for cid in call_ids:
            call_entries = [e for e in snapshot if e.call_id == cid]
            call_by_service = self._aggregate_by_service(call_entries)
            result.append(
                {
                    "call_id": cid,
                    "total_inr": round(sum(e.cost_inr for e in call_entries), 4),
                    "by_service": self._round_money_map(call_by_service),
                    "by_service_model_mode": self._round_money_map(
                        self._aggregate_by_service_model_mode(call_entries)
                    ),
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

    def detailed_summary(self) -> dict[str, Any]:
        """Extended summary with per-call breakdown and projections."""
        base = self.summary()
        snapshot = self._snapshot_entries()
        base["per_call"] = self._per_call_summaries(snapshot)
        base["projections"] = self._cost_projections(base["avg_cost_per_call_inr"])
        return base

    def daily_summary(self, day: date) -> dict[str, Any]:
        """Aggregate costs for a specific calendar day."""
        snapshot = self._snapshot_entries()
        day_entries = [e for e in snapshot if datetime.fromtimestamp(e.timestamp).date() == day]
        by_service = self._aggregate_by_service(day_entries)
        total = sum(e.cost_inr for e in day_entries)
        return {
            "date": day.isoformat(),
            "total_inr": round(total, 4),
            "by_service": self._round_money_map(by_service),
            "by_service_model_mode": self._round_money_map(
                self._aggregate_by_service_model_mode(day_entries)
            ),
            "api_calls": len(day_entries),
        }

    def monthly_summary(self, year: int, month: int) -> dict[str, Any]:
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
            "by_service": self._round_money_map(by_service),
            "by_service_model_mode": self._round_money_map(
                self._aggregate_by_service_model_mode(month_entries)
            ),
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
        *,
        mode: str = "",
        billable_units: float | None = None,
        rate_inr_per_unit: float = 0.0,
        metadata: dict[str, Any] | None = None,
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
                    mode=mode,
                    billable_units=billable_units if billable_units is not None else units,
                    rate_inr_per_unit=rate_inr_per_unit,
                    metadata=metadata or {},
                )
            )
        if call_id:
            for w in self.check_alerts(call_id):
                logger.warning("Cost alert: %s", w)
