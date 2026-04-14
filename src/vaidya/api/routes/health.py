"""Health check endpoints."""

from fastapi import APIRouter

from vaidya import __version__

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness check — verifies dependencies are connected."""
    # Phase 1: basic check. Phase 2: verify Redis + ChromaDB connectivity.
    return {"status": "ready", "version": __version__}
