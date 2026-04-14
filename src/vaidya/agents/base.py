"""Base agent protocol and shared implementation."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

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
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        max_tokens: int = 2048,
        wiki_grounding: bool = False,
    ) -> str:
        """Call LLM with system + user messages. Returns raw text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._client.chat(
            self._model,
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
        temperature: float = 0.1,
        reasoning_effort: str | None = None,
        max_tokens: int = 2048,
        wiki_grounding: bool = False,
    ) -> dict[str, Any]:
        """Call LLM and parse JSON response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._client.chat_json(
            self._model,
            messages,
            temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            wiki_grounding=wiki_grounding,
        )

    def _fallback_response(self, language: str) -> AgentResponse:
        """Fallback response when agent processing fails."""
        fallbacks = {
            "hi-IN": "Maaf kijiye, thodi dikkat aa rahi hai. Kya aap phir se bata sakte hain?",
            "ta-IN": "Mannikkavum, sila thozhilnutpa prachanai. Thayavu seythu meendum sollunga?",
            "bn-IN": "Dukkhito, ektu somossa hocche. Abar bolben please?",
        }
        text = fallbacks.get(language, fallbacks["hi-IN"])
        return AgentResponse(text=text, error="agent_processing_failed")
