"""Health check and cost monitoring endpoints."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from vaidya import __version__

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_dependency(
    name: str,
    check_fn: Callable[[], Any | Awaitable[Any]],
    *,
    is_async: bool = False,
    not_init_detail: str = "",
) -> dict[str, object]:
    """Run a single dependency health check and return its status dict.

    Parameters
    ----------
    name:
        Dependency name (used in warning log messages).
    check_fn:
        A callable that returns a truthy value on success.
    is_async:
        Whether *check_fn* is a coroutine function.
    not_init_detail:
        If provided, this string is used when the dependency is ``None``.
    """
    try:
        result = (await check_fn()) if is_async else check_fn()
        return {"ok": bool(result)}
    except Exception as exc:
        logger.warning("Ready check: %s exception", name, extra={"error": str(exc)})
        return {"ok": False}


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness check with deep dependency verification.

    Checks:
    - Redis: PING via session manager
    - ChromaDB: heartbeat via knowledge store
    - Sarvam API key: configured (non-empty)
    - Scheme data: loaded count > 0

    Returns HTTP 200 with ``"status": "ready"`` when all checks pass,
    or HTTP 503 with ``"status": "degraded"`` if any check fails.
    """
    checks: dict[str, dict[str, object]] = {}

    # --- Redis ---
    session = getattr(request.app.state, "session", None)
    if session is not None:
        checks["redis"] = await _check_dependency("Redis", session.ping, is_async=True)
    else:
        checks["redis"] = {"ok": False, "detail": "session manager not initialised"}

    # --- ChromaDB ---
    store = getattr(request.app.state, "store", None)
    if store is not None:
        checks["chromadb"] = await _check_dependency("ChromaDB", store.is_healthy)
    else:
        checks["chromadb"] = {"ok": False, "detail": "knowledge store not initialised"}

    # --- Sarvam API key ---
    settings = getattr(request.app.state, "settings", None)
    sarvam_ok = bool(settings and settings.sarvam_api_key)
    checks["sarvam_api_key"] = {"ok": sarvam_ok}

    # --- Scheme data ---
    if store is not None:
        try:
            scheme_count = store.count
        except Exception:
            scheme_count = 0
        schemes_ok = scheme_count > 0
        checks["schemes"] = {"ok": schemes_ok, "count": scheme_count}
    else:
        checks["schemes"] = {"ok": False, "count": 0}

    all_ok = all(c["ok"] for c in checks.values())
    status = "ready" if all_ok else "degraded"
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={"status": status, "version": __version__, "checks": checks},
        status_code=status_code,
    )


@router.get("/costs")
async def costs(
    request: Request,
    detailed: bool = Query(default=False, description="Include per-call breakdown"),
) -> dict[str, Any]:
    """API cost summary (Sarvam usage tracking).

    Pass ``?detailed=true`` for per-call breakdowns and monthly projections.
    Includes a ``recent_calls`` list with the last 10 call IDs and their totals.
    """
    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        result = cast(
            dict[str, Any],
            client.costs.detailed_summary() if detailed else client.costs.summary(),
        )

        # Add recent_calls: last 10 unique call IDs with totals
        all_call_ids = []
        seen: set[str] = set()
        for entry in reversed(client.costs.entries):
            if entry.call_id and entry.call_id not in seen:
                seen.add(entry.call_id)
                all_call_ids.append(entry.call_id)
            if len(all_call_ids) >= 10:
                break
        result["recent_calls"] = [
            {"call_id": cid, "total_inr": round(client.costs.cost_for_call(cid), 4)}
            for cid in all_call_ids
        ]
        return result
    return {
        "total_inr": 0,
        "by_service": {},
        "by_model": {},
        "by_mode": {},
        "by_service_model_mode": {},
        "call_count": 0,
        "api_calls": 0,
        "avg_cost_per_call_inr": 0,
        "recent_calls": [],
    }


@router.get("/costs/{call_id}")
async def costs_for_call(
    call_id: str,
    request: Request,
) -> dict[str, Any]:
    """Per-call cost breakdown (Sarvam usage tracking)."""
    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        return cast(dict[str, Any], client.costs.breakdown_for_call(call_id))
    return {
        "call_id": call_id,
        "total_inr": 0,
        "by_service": {},
        "api_call_count": 0,
        "entries": [],
    }


@router.get("/costs/daily/{day}")
async def costs_daily(
    day: str,
    request: Request,
) -> dict[str, Any]:
    """Daily cost summary. Pass date as YYYY-MM-DD."""
    from datetime import date as _date

    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        try:
            d = _date.fromisoformat(day)
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}
        return cast(dict[str, Any], client.costs.daily_summary(d))
    return {
        "date": day,
        "total_inr": 0,
        "by_service": {},
        "by_service_model_mode": {},
        "api_calls": 0,
    }


@router.get("/costs/monthly/{year}/{month}")
async def costs_monthly(
    year: int,
    month: int,
    request: Request,
) -> dict[str, Any]:
    """Monthly cost summary."""
    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        return cast(dict[str, Any], client.costs.monthly_summary(year, month))
    return {
        "year": year,
        "month": month,
        "total_inr": 0,
        "by_service": {},
        "by_service_model_mode": {},
        "api_calls": 0,
    }


@router.get("/costs/alerts")
async def cost_alerts(request: Request) -> dict[str, list[str]]:
    """Active cost threshold warnings."""
    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        return {"alerts": client.costs.check_alerts()}
    return {"alerts": []}
