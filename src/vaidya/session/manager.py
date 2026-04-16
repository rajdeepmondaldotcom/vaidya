"""Redis-backed session management for Vaidya voice sessions.

Each phone call gets a :class:`ConversationContext` stored in Redis with
a configurable TTL (default 30 min).  Dropped-call recovery is implicit:
if the same caller rings back within the TTL, the session resumes from
where it left off.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime

import redis.asyncio as aioredis

from vaidya.models.conversation import ConversationContext, ConversationPhase

logger = logging.getLogger(__name__)

_KEY_PREFIX = "vaidya:session:"
_PHONE_PREFIX = "vaidya:phone:"


class SessionManager:
    """Redis-backed conversation session state.

    Lifecycle::

        mgr = SessionManager("redis://localhost:6379/0")
        ctx = await mgr.create("call-123", phone_hash="abc", language="hi-IN")
        # ... agent turns ...
        await mgr.update(ctx)
        # ... later ...
        ctx = await mgr.get("call-123")  # resumes
        await mgr.delete("call-123")     # cleanup
        await mgr.close()
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 1800, max_connections: int = 10) -> None:
        self._redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
            socket_connect_timeout=5.0,
            socket_timeout=10.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self._ttl = ttl_seconds

    async def _persist_session(
        self,
        context: ConversationContext,
        *,
        log_action: str = "persist",
    ) -> bool:
        """Serialise *context* to Redis with TTL, plus phone index if present."""
        key = f"{_KEY_PREFIX}{context.call_id}"
        context.updated_at = datetime.now(UTC)
        try:
            data = context.model_dump_json()
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.setex(key, self._ttl, data)
                if context.phone_number_hash:
                    pipe.setex(
                        f"{_PHONE_PREFIX}{context.phone_number_hash}",
                        self._ttl,
                        context.call_id,
                    )
                await pipe.execute()
            return True
        except Exception as exc:
            logger.error(
                "Redis session %s failed",
                log_action,
                extra={"call_id": context.call_id, "error": str(exc)},
            )
            return False

    async def create(
        self,
        call_id: str,
        phone_hash: str = "",
        language: str = "hi-IN",
    ) -> ConversationContext | None:
        """Create a new conversation session and persist it."""
        context = ConversationContext(
            call_id=call_id,
            phone_number_hash=phone_hash,
            language=language,
            phase=ConversationPhase.WELCOME,
        )

        persisted = await self._persist_session(context, log_action="create")
        if not persisted:
            logger.error("Failed to persist new session", extra={"call_id": call_id})
            return None

        logger.info(
            "Session created",
            extra={"call_id": call_id, "language": language},
        )
        return context

    async def get(self, call_id: str) -> ConversationContext | None:
        """Load a session from Redis, returning ``None`` if expired or missing."""
        key = f"{_KEY_PREFIX}{call_id}"
        try:
            data = await self._redis.get(key)
        except Exception as exc:
            logger.error(
                "Redis GET failed",
                extra={"call_id": call_id, "error": str(exc)},
            )
            return None

        if data is None:
            return None

        try:
            return ConversationContext.model_validate_json(data)
        except Exception as exc:
            logger.error(
                "Session deserialisation failed",
                extra={"call_id": call_id, "error": str(exc)},
            )
            return None

    async def find_by_phone(self, phone_hash: str) -> str | None:
        """Look up a call_id by phone hash for dropped-call recovery."""
        if not phone_hash:
            return None
        phone_key = f"{_PHONE_PREFIX}{phone_hash}"
        try:
            return await self._redis.get(phone_key)
        except Exception as exc:
            logger.error(
                "Redis phone index GET failed",
                extra={"phone_hash": phone_hash[:8], "error": str(exc)},
            )
            return None

    async def update(self, context: ConversationContext) -> None:
        """Serialise *context* to Redis and refresh the TTL."""
        await self._persist_session(context, log_action="update")

    async def _collect_delete_keys(self, call_id: str, session_key: str) -> list[str]:
        """Collect all Redis keys to delete for a session (session + phone index)."""
        keys = [session_key]
        try:
            data = await self._redis.get(session_key)
            if data is not None:
                context = ConversationContext.model_validate_json(data)
                if context.phone_number_hash:
                    keys.append(f"{_PHONE_PREFIX}{context.phone_number_hash}")
        except Exception as exc:
            logger.debug(
                "Best-effort phone index cleanup failed during delete",
                extra={"call_id": call_id, "error": str(exc)},
            )
        return keys

    async def delete(self, call_id: str) -> None:
        """Remove a session and its phone index from Redis."""
        key = f"{_KEY_PREFIX}{call_id}"
        try:
            keys_to_delete = await self._collect_delete_keys(call_id, key)
            await self._redis.delete(*keys_to_delete)
            logger.info("Session deleted", extra={"call_id": call_id})
        except Exception as exc:
            logger.error(
                "Redis DELETE failed",
                extra={"call_id": call_id, "error": str(exc)},
            )

    async def exists(self, call_id: str) -> bool:
        """Check whether a session exists (without loading it)."""
        key = f"{_KEY_PREFIX}{call_id}"
        try:
            return bool(await self._redis.exists(key))
        except Exception as exc:
            logger.error(
                "Redis EXISTS failed",
                extra={"call_id": call_id, "error": str(exc)},
            )
            return False

    async def ping(self) -> bool:
        """Check Redis connectivity via PING."""
        try:
            return bool(await self._redis.ping())
        except Exception as exc:
            logger.error("Redis PING failed", extra={"error": str(exc)})
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        try:
            await self._redis.aclose()
            logger.info("SessionManager Redis connection closed")
        except Exception as exc:
            logger.error("Redis close failed", extra={"error": str(exc)})

    @staticmethod
    def generate_call_id(phone_hash: str) -> str:
        """Generate a unique 16-char hex call ID from phone hash and timestamp."""
        raw = f"{phone_hash}-{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
