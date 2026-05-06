"""HTTP conversation route tests for request-to-manager wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vaidya.api.routes.conversation import router as conversation_router
from vaidya.dependencies import get_client, get_conversation_manager


@pytest.fixture()
async def client_and_manager():
    app = FastAPI(title="Vaidya Conversation Route Test")
    app.include_router(conversation_router, prefix="/conversation")

    manager = MagicMock()
    manager.start_conversation = AsyncMock(return_value=("call-123", "welcome"))
    manager.handle_turn = AsyncMock(return_value="next response")
    manager.get_context = AsyncMock(return_value=None)
    sarvam_client = MagicMock()
    sarvam_client.costs.cost_for_call.return_value = 0.0

    app.dependency_overrides[get_conversation_manager] = lambda: manager
    app.dependency_overrides[get_client] = lambda: sarvam_client

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client, manager


class TestStartConversationRoute:
    async def test_passes_channel_to_manager(self, client_and_manager):
        client, manager = client_and_manager

        response = await client.post(
            "/conversation/start",
            json={
                "phone_number_hash": "hash-123",
                "language": "hi-IN",
                "channel": "voice",
            },
        )

        assert response.status_code == 200
        manager.start_conversation.assert_awaited_once_with(
            phone_hash="hash-123",
            language="hi-IN",
            channel="voice",
        )

    async def test_text_channel_still_passes_through(self, client_and_manager):
        client, manager = client_and_manager

        response = await client.post(
            "/conversation/start",
            json={
                "phone_number_hash": "hash-456",
                "language": "ta-IN",
                "channel": "web",
            },
        )

        assert response.status_code == 200
        manager.start_conversation.assert_awaited_once_with(
            phone_hash="hash-456",
            language="ta-IN",
            channel="web",
        )


class TestConversationTurnRoute:
    async def test_passes_channel_to_manager(self, client_and_manager):
        client, manager = client_and_manager

        response = await client.post(
            "/conversation/call-123/turn",
            json={
                "text": "Tamil",
                "language": "ta-IN",
                "stt_confidence": 0.82,
                "channel": "voice",
            },
        )

        assert response.status_code == 200
        manager.handle_turn.assert_awaited_once_with(
            call_id="call-123",
            user_text="Tamil",
            language="ta-IN",
            stt_confidence=0.82,
            channel="voice",
        )

    async def test_turn_defaults_to_text_channel(self, client_and_manager):
        client, manager = client_and_manager

        response = await client.post(
            "/conversation/call-123/turn",
            json={"text": "hello", "language": "en-IN"},
        )

        assert response.status_code == 200
        manager.handle_turn.assert_awaited_once_with(
            call_id="call-123",
            user_text="hello",
            language="en-IN",
            stt_confidence=1.0,
            channel="text",
        )
