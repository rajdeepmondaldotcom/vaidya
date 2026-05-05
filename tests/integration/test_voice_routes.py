"""Tests for Twilio voice webhook behavior."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from twilio.request_validator import RequestValidator

from vaidya.api.routes.voice import (
    _phone_hash_from_call_data,
)
from vaidya.api.routes.voice import (
    router as voice_router,
)


def _build_app(settings: MagicMock) -> FastAPI:
    app = FastAPI(title="Vaidya Voice Route Test")
    app.include_router(voice_router, prefix="/voice")
    app.state.settings = settings
    return app


def _settings(**overrides) -> MagicMock:
    values = {
        "voice_websocket_url": "wss://voice.example.com/voice/stream",
        "voice_status_callback_url": "",
        "twilio_auth_token": "",
    }
    values.update(overrides)
    return MagicMock(**values)


def _signature(url: str, params: dict[str, str], token: str) -> str:
    return RequestValidator(token).compute_signature(url, params)


@pytest.fixture()
async def client_factory():
    clients = []

    async def _make(settings: MagicMock) -> AsyncClient:
        client = AsyncClient(
            transport=ASGITransport(app=_build_app(settings)),
            base_url="http://test",
        )
        await client.__aenter__()
        clients.append(client)
        return client

    yield _make

    for client in clients:
        await client.__aexit__(None, None, None)


class TestIncomingCall:
    async def test_valid_twilio_signature_returns_stream_with_hashed_phone(
        self,
        client_factory,
    ):
        token = "test-token"
        settings = _settings(
            twilio_auth_token=token,
            voice_status_callback_url="https://voice.example.com/voice/status",
        )
        client = await client_factory(settings)
        params = {"From": "+15551234567", "CallSid": "CA123"}
        headers = {"X-Twilio-Signature": _signature("http://test/voice/incoming", params, token)}

        response = await client.post("/voice/incoming", data=params, headers=headers)

        assert response.status_code == 200
        body = response.text
        expected_hash = hashlib.sha256(params["From"].encode("utf-8")).hexdigest()[:16]
        assert f'<Parameter name="phone_hash" value="{expected_hash}" />' in body
        assert "+15551234567" not in body
        assert 'statusCallback="https://voice.example.com/voice/status"' in body
        assert 'statusCallbackMethod="POST"' in body

    async def test_invalid_twilio_signature_rejects(self, client_factory):
        settings = _settings(twilio_auth_token="test-token")
        client = await client_factory(settings)

        response = await client.post(
            "/voice/incoming",
            data={"From": "+15551234567", "CallSid": "CA123"},
            headers={"X-Twilio-Signature": "bad"},
        )

        assert response.status_code == 403

    async def test_missing_websocket_url_returns_apology(self, client_factory):
        settings = _settings(voice_websocket_url="")
        client = await client_factory(settings)

        response = await client.post(
            "/voice/incoming",
            data={"From": "+15551234567", "CallSid": "CA123"},
        )

        assert response.status_code == 200
        assert "not yet configured" in response.text
        assert "<Hangup/>" in response.text


class TestPhoneHashFromCallData:
    def test_uses_valid_custom_phone_hash(self):
        call_data = {"body": {"phone_hash": "abcd1234abcd1234"}, "call_id": "CA1"}

        assert _phone_hash_from_call_data(call_data) == "abcd1234abcd1234"

    def test_hashes_invalid_custom_phone_hash(self):
        call_data = {"body": {"phone_hash": "+15551234567"}, "call_id": "CA1"}

        expected = hashlib.sha256(b"+15551234567").hexdigest()[:16]
        assert _phone_hash_from_call_data(call_data) == expected

    def test_falls_back_to_call_sid(self):
        call_data = {"body": {}, "call_id": "CA123", "stream_id": "MZ123"}

        expected = hashlib.sha256(b"CA123").hexdigest()[:16]
        assert _phone_hash_from_call_data(call_data) == expected
