"""Health check and cost monitoring endpoints."""

from fastapi import APIRouter, Request

from vaidya import __version__

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness check."""
    return {"status": "ready", "version": __version__}


@router.get("/costs")
async def costs(request: Request) -> dict:
    """API cost summary (Sarvam usage tracking)."""
    client = getattr(request.app.state, "client", None)
    if client and hasattr(client, "costs"):
        return client.costs.summary()
    return {"total_inr": 0, "by_service": {}, "call_count": 0, "api_calls": 0}
