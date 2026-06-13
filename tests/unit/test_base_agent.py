"""Tests for BaseAgent shared infrastructure.

Covers:
- safe_process: wraps process() with exception handling and fallback
- _fallback_response: returns localised error message
- process: raises NotImplementedError by default
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.agents.base import BaseAgent
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value="response")
    client.chat_json = AsyncMock(return_value={})
    return client


def _make_context(call_id: str = "test-call") -> ConversationContext:
    return ConversationContext(
        call_id=call_id,
        phone_number_hash="hash",
        language="hi-IN",
        phase=ConversationPhase.INTAKE,
    )


class WorkingAgent(BaseAgent):
    """Agent that returns a simple response."""

    async def process(self, context, user_input):
        return AgentResponse(text="success", metadata={"agent": "working"})


class FailingAgent(BaseAgent):
    """Agent whose process() always raises."""

    async def process(self, context, user_input):
        raise RuntimeError("LLM API timeout")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSafeProcess:
    @pytest.mark.asyncio
    async def test_returns_agent_response_on_success(self) -> None:
        agent = WorkingAgent(client=_mock_client(), model="test", agent_name="working")
        result = await agent.safe_process(_make_context(), "hello")
        assert result.text == "success"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_returns_fallback_on_exception(self) -> None:
        agent = FailingAgent(client=_mock_client(), model="test", agent_name="failing")
        result = await agent.safe_process(_make_context(), "hello")
        assert result.error == "agent_processing_failed"
        assert result.text  # should have some fallback text

    @pytest.mark.asyncio
    async def test_fallback_response_uses_language(self) -> None:
        agent = FailingAgent(client=_mock_client(), model="test", agent_name="failing")
        ctx = _make_context()
        ctx.language = "en-IN"
        result = await agent.safe_process(ctx, "hello")
        assert result.error == "agent_processing_failed"


class TestProcessRaisesNotImplemented:
    @pytest.mark.asyncio
    async def test_base_agent_process_raises(self) -> None:
        agent = BaseAgent(client=_mock_client(), model="test", agent_name="base")
        with pytest.raises(NotImplementedError):
            await agent.process(_make_context(), "hello")


class TestFallbackResponse:
    def test_returns_agent_response_with_error(self) -> None:
        agent = BaseAgent(client=_mock_client(), model="test", agent_name="base")
        result = agent._fallback_response("hi-IN")
        assert isinstance(result, AgentResponse)
        assert result.error == "agent_processing_failed"
        assert result.text  # non-empty fallback text


class TestLlmTimeoutForwarding:
    """The per-instance _llm_timeout is threaded into every client LLM call."""

    def test_defaults_to_none(self) -> None:
        # Untouched agents ride the client's default ceiling (the slow tail).
        agent = BaseAgent(client=_mock_client(), model="test", agent_name="base")
        assert agent._llm_timeout is None

    @pytest.mark.asyncio
    async def test_call_llm_forwards_default_none_timeout(self) -> None:
        client = _mock_client()
        agent = BaseAgent(client=client, model="test", agent_name="base")

        await agent._call_llm("system", "user")

        assert client.chat.await_args.kwargs["timeout"] is None

    @pytest.mark.asyncio
    async def test_call_llm_forwards_short_timeout(self) -> None:
        client = _mock_client()
        agent = BaseAgent(client=client, model="test", agent_name="base")
        agent._llm_timeout = 12.0  # what app.py sets on conversational agents

        await agent._call_llm("system", "user")

        assert client.chat.await_args.kwargs["timeout"] == 12.0

    @pytest.mark.asyncio
    async def test_call_llm_json_forwards_short_timeout(self) -> None:
        client = _mock_client()
        agent = BaseAgent(client=client, model="test", agent_name="base")
        agent._llm_timeout = 12.0

        await agent._call_llm_json("system", "user")

        assert client.chat_json.await_args.kwargs["timeout"] == 12.0
