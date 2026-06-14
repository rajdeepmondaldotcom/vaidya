"""Request middleware for timing, request IDs, rate limiting, and error handling."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

_RATE_LIMIT_BODY = json.dumps({"detail": "Rate limit exceeded"}).encode("utf-8")


def _rate_limit_response() -> Response:
    """Build an HTTP 429 rate-limit response."""
    return Response(
        content=_RATE_LIMIT_BODY,
        status_code=429,
        media_type="application/json",
    )


class RateLimiter:
    """In-memory sliding-window rate limiter.

    NOTE: This is per-process only. For multi-replica deployments,
    replace with Redis-backed rate limiting.
    """

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Return ``True`` if the request is within the rate limit."""
        now = time.time()
        window = self._windows[key]
        cutoff = now - window_seconds
        # Remove expired entries from the left (oldest first)
        while window and window[0] <= cutoff:
            window.popleft()
        # Check limit
        if len(window) >= max_requests:
            return False
        window.append(now)
        return True


# Shared instance used by the middleware
_rate_limiter = RateLimiter()

# Route-specific rate limit rules: (max_requests, window_seconds)
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "conversation_start": (10, 60),  # 10 requests per minute per IP
    "conversation_turn": (30, 60),  # 30 requests per minute per call_id
    "simulate": (5, 60),  # 5 requests per minute per IP
}


def _match_rule(path: str, client_ip: str) -> tuple[str, int, int] | None:
    """Return ``(limit_key, max_requests, window_seconds)`` for the path, or None."""
    if path == "/conversation/start":
        max_req, window = _RATE_LIMITS["conversation_start"]
        return f"conv_start:{client_ip}", max_req, window

    if path.startswith("/conversation/") and path.endswith("/turn"):
        parts = path.strip("/").split("/")
        if len(parts) == 3:
            call_id = parts[1]
            max_req, window = _RATE_LIMITS["conversation_turn"]
            return f"conv_turn:{call_id}", max_req, window

    if path.startswith("/simulate"):
        max_req, window = _RATE_LIMITS["simulate"]
        return f"simulate:{client_ip}", max_req, window

    return None


def _rate_limiting_enabled() -> bool:
    """Rate limiting is on by default; set RATE_LIMIT_ENABLED=false to disable.

    Production keeps the protective per-route limits. Local benchmarking against
    the text-simulation endpoint (the eval suite fires far more than the 5/min
    /simulate limit) sets this false so a measurement run isn't throttled.
    """
    return os.environ.get("RATE_LIMIT_ENABLED", "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-route rate limits and return HTTP 429 when exceeded."""

    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._enabled = _rate_limiting_enabled()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only rate-limit POST endpoints (when enabled)
        if not self._enabled or request.method != "POST":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        rule = _match_rule(request.url.path, client_ip)

        if rule is not None:
            limit_key, max_req, window = rule
            if not _rate_limiter.is_allowed(limit_key, max_req, window):
                return _rate_limit_response()

        return await call_next(request)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to every request/response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Bind request_id to structlog context for all downstream log calls
        try:
            import structlog

            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)
        except ImportError:
            pass

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.0f}"

        logger.info(
            "%s %s %d %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        return response
