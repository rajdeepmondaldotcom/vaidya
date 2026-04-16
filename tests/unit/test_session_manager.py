"""Tests for Redis-backed SessionManager.

Covers:
- create: returns ConversationContext in WELCOME phase, persists via setex,
  creates phone->call_id secondary index
- get: loads and deserializes stored context, returns None for missing/corrupted
- update: serializes context and refreshes TTL
- delete: removes session key and phone index
- exists: True/False checks
- generate_call_id: produces unique 16-char hex strings
- find_by_phone: lookup via secondary index, None when missing
- ping: True on healthy Redis, False on error
- Error handling: Redis failures log gracefully
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.session.manager import SessionManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis() -> AsyncMock:
    """An AsyncMock that stands in for a redis.asyncio connection."""
    r = AsyncMock()
    r.setex = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.delete = AsyncMock()
    r.exists = AsyncMock(return_value=0)
    r.ping = AsyncMock(return_value=True)
    r.aclose = AsyncMock()

    # Mock pipeline() — synchronous call returning an async context manager
    pipe_mock = AsyncMock()
    pipe_mock.setex = MagicMock()  # pipeline commands are sync (buffered)
    pipe_mock.execute = AsyncMock(return_value=[True])
    pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
    pipe_mock.__aexit__ = AsyncMock(return_value=False)
    r.pipeline = MagicMock(return_value=pipe_mock)  # pipeline() is sync
    r._pipe_mock = pipe_mock  # expose for assertions
    return r


@pytest.fixture()
def manager(mock_redis: AsyncMock) -> SessionManager:
    """SessionManager with its internal Redis client replaced by a mock."""
    with patch("vaidya.session.manager.aioredis.from_url", return_value=mock_redis):
        mgr = SessionManager("redis://localhost:6379/0", ttl_seconds=1800)
    return mgr


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_returns_welcome_phase(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        ctx = await manager.create("call-1", phone_hash="ph1", language="hi-IN")

        assert isinstance(ctx, ConversationContext)
        assert ctx.phase == ConversationPhase.WELCOME
        assert ctx.call_id == "call-1"
        assert ctx.language == "hi-IN"

    @pytest.mark.asyncio
    async def test_create_persists_via_setex(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        await manager.create("call-2", phone_hash="ph2", language="ta-IN")

        # create() uses a pipeline; check the pipe's setex calls
        pipe = mock_redis._pipe_mock
        session_setex_calls = [c for c in pipe.setex.call_args_list if "vaidya:session:" in str(c)]
        assert len(session_setex_calls) >= 1

    @pytest.mark.asyncio
    async def test_create_sets_phone_index(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        await manager.create("call-3", phone_hash="phonehash123", language="hi-IN")

        # Should have a setex call for the phone index key in the pipeline
        pipe = mock_redis._pipe_mock
        phone_setex_calls = [c for c in pipe.setex.call_args_list if "vaidya:phone:" in str(c)]
        assert len(phone_setex_calls) == 1

    @pytest.mark.asyncio
    async def test_create_no_phone_index_when_empty_hash(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        await manager.create("call-4", phone_hash="", language="hi-IN")

        # Only the session setex should be called, not the phone index
        pipe = mock_redis._pipe_mock
        phone_setex_calls = [c for c in pipe.setex.call_args_list if "vaidya:phone:" in str(c)]
        assert len(phone_setex_calls) == 0


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_get_loads_stored_context(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        ctx = ConversationContext(
            call_id="call-5",
            phone_number_hash="ph5",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        mock_redis.get.return_value = ctx.model_dump_json()

        loaded = await manager.get("call-5")

        assert loaded is not None
        assert loaded.call_id == "call-5"
        assert loaded.phase == ConversationPhase.INTAKE
        assert loaded.language == "hi-IN"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None

        result = await manager.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_deserialization_error(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = "not-valid-json{{{corrupt"

        result = await manager.get("bad-data")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_redis_error(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.side_effect = ConnectionError("Redis down")

        result = await manager.get("call-err")
        assert result is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_calls_setex_with_ttl(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        ctx = ConversationContext(
            call_id="call-6",
            phone_number_hash="ph6",
            language="bn-IN",
            phase=ConversationPhase.PROCESSING,
        )

        await manager.update(ctx)

        # update() now uses a pipeline
        pipe = mock_redis._pipe_mock
        session_setex_calls = [
            c for c in pipe.setex.call_args_list if "vaidya:session:call-6" in str(c)
        ]
        assert len(session_setex_calls) >= 1

    @pytest.mark.asyncio
    async def test_update_refreshes_updated_at(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        ctx = ConversationContext(
            call_id="call-7",
            phone_number_hash="ph7",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        original_ts = ctx.updated_at

        await manager.update(ctx)

        assert ctx.updated_at >= original_ts


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_session_key(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        # Return a stored context so delete can find the phone hash
        ctx = ConversationContext(
            call_id="call-8",
            phone_number_hash="ph8",
            language="hi-IN",
            phase=ConversationPhase.WELCOME,
        )
        mock_redis.get.return_value = ctx.model_dump_json()

        await manager.delete("call-8")

        # Should delete both the session key and the phone index in a single call
        mock_redis.delete.assert_called_once()
        deleted_keys = mock_redis.delete.call_args[0]
        assert "vaidya:session:call-8" in deleted_keys
        assert "vaidya:phone:ph8" in deleted_keys

    @pytest.mark.asyncio
    async def test_delete_handles_missing_session(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None

        # Should not raise
        await manager.delete("nonexistent")

        # Still calls delete on the session key
        mock_redis.delete.assert_called_once_with("vaidya:session:nonexistent")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    @pytest.mark.asyncio
    async def test_exists_returns_true(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.exists.return_value = 1

        assert await manager.exists("call-10") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.exists.return_value = 0

        assert await manager.exists("call-11") is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_redis_error(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.exists.side_effect = ConnectionError("Redis down")

        assert await manager.exists("call-err") is False


# ---------------------------------------------------------------------------
# generate_call_id
# ---------------------------------------------------------------------------


class TestGenerateCallId:
    def test_produces_16_char_hex(self) -> None:
        call_id = SessionManager.generate_call_id("some_phone_hash")

        assert len(call_id) == 16
        # Should be valid hex
        int(call_id, 16)

    def test_unique_across_calls(self) -> None:
        ids = {SessionManager.generate_call_id(f"phone_{i}") for i in range(50)}
        # All 50 should be unique (time-based + different phone hashes)
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# find_by_phone
# ---------------------------------------------------------------------------


class TestFindByPhone:
    @pytest.mark.asyncio
    async def test_returns_call_id_from_index(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = "call-recovered"

        result = await manager.find_by_phone("phonehash_abc")
        assert result == "call-recovered"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None

        result = await manager.find_by_phone("unknown_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_hash(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        result = await manager.find_by_phone("")
        assert result is None
        # Should not even call Redis
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_error(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.side_effect = ConnectionError("Redis down")

        result = await manager.find_by_phone("phonehash_err")
        assert result is None


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_returns_true_on_healthy_redis(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.ping.return_value = True

        assert await manager.ping() is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_redis_error(
        self, manager: SessionManager, mock_redis: AsyncMock
    ) -> None:
        mock_redis.ping.side_effect = ConnectionError("Redis down")

        assert await manager.ping() is False
