"""Base agent protocol and shared implementation."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from vaidya.i18n import get_msg
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)


@runtime_checkable
class Agent(Protocol):
    """Structural interface for all Vaidya agents.

    Every agent receives the full ConversationContext and the current
    user utterance. Returns an AgentResponse that the orchestrator
    routes back to the voice pipeline.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse: ...


class BaseAgent:
    """Shared implementation for agents that use the Sarvam LLM.

    Not required to inherit — the Protocol is the contract, this is convenience.
    Provides: LLM call with timing, JSON parsing, error capture, fallback responses.
    """

    def __init__(self, client: SarvamClient, model: str, agent_name: str) -> None:
        self._client = client
        self._model = model
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        max_tokens: int = 2048,
        wiki_grounding: bool = False,
    ) -> str:
        """Call LLM with system + user messages. Returns raw text.

        When *model* is provided it overrides ``self._model`` for this
        single call, avoiding mutation of instance state.
        """
        effective_model = model or self._model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._client.chat(
            effective_model,
            messages,
            temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
        )

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        reasoning_effort: str | None = None,
        max_tokens: int = 2048,
        wiki_grounding: bool = False,
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response.

        When *model* is provided it overrides ``self._model`` for this
        single call, avoiding mutation of instance state.
        """
        effective_model = model or self._model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._client.chat_json(
            effective_model,
            messages,
            temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
        )

    async def safe_process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Wrapper that catches exceptions and returns a fallback.

        Agents should implement ``process()`` with business logic only.
        The orchestrator calls ``safe_process()`` to ensure uniform error
        handling across all agents.
        """
        try:
            return await self.process(context, user_input)
        except Exception as exc:
            logger.error(
                "%s agent failed",
                self._name,
                extra={"error": str(exc)[:200], "call_id": context.call_id},
                exc_info=True,
            )
            return self._fallback_response(context.language)

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Process a turn. Subclasses must override."""
        raise NotImplementedError

    def _fallback_response(self, language: str) -> AgentResponse:
        """Fallback response when agent processing fails."""
        text = get_msg("base_agent", "fallback", language)
        return AgentResponse(text=text, error="agent_processing_failed")
