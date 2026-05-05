"""HTTP conversation route tests for request-to-manager wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vaidya.api.routes.conversation import router as conversation_router
from vaidya.dependencies import get_conversation_manager


@pytest.fixture()
async def client_and_manager():
    app = FastAPI(title="Vaidya Conversation Route Test")
    app.include_router(conversation_router, prefix="/conversation")

    manager = MagicMock()
    manager.start_conversation = AsyncMock(return_value=("call-123", "welcome"))

    app.dependency_overrides[get_conversation_manager] = lambda: manager

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
