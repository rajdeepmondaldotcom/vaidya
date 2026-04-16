"""ConversationManager: the main turn-orchestration layer.

Sits between the HTTP/voice layer and the :class:`Orchestrator`.  Handles:

- Session lifecycle (create, load, save via Redis)
- PII masking on stored transcripts
- Translation between user language and agent language
- Audit logging for every turn
- STT confidence pass-through
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from vaidya.agents.orchestrator import Orchestrator
from vaidya.compliance.audit import AuditTrail
from vaidya.compliance.consent import ConsentTracker
from vaidya.compliance.pii import mask_pii as _default_mask_pii
from vaidya.i18n import get_msg
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.pipeline.translator import Translator
from vaidya.session.manager import SessionManager

if TYPE_CHECKING:
    from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

# Internal processing language for agents (English keeps prompts stable)
_AGENT_LANG = "en-IN"

PiiMaskerFn = Callable[[str], str]


class ConversationManager:
    """Top-level turn handler that wires session, PII, translation, and audit.

    Typical usage from an API route::

        mgr = ConversationManager(orchestrator, session, translator, audit)
        call_id, welcome = await mgr.start_conversation(phone_hash, "hi-IN")
        response = await mgr.handle_turn(call_id, "mujhe madad chahiye")
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        session_manager: SessionManager,
        translator: Translator,
        audit_trail: AuditTrail,
        consent_tracker: ConsentTracker | None = None,
        pii_masker: PiiMaskerFn | None = None,
        sarvam_client: SarvamClient | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._session = session_manager
        self._translator = translator
        self._audit = audit_trail
        self._consent = consent_tracker
        self._mask_pii = pii_masker or _default_mask_pii
        self._sarvam_client = sarvam_client
        self._turn_locks: dict[str, asyncio.Lock] = {}

    async def start_conversation(
        self,
        phone_hash: str,
        language: str = "hi-IN",
    ) -> tuple[str, str]:
        """Create a new session and return ``(call_id, welcome_message)``.

        Resumes an existing non-terminal session if one exists for
        *phone_hash* (dropped-call recovery).
        """
        recovered = await self._try_recover_session(phone_hash)
        if recovered is not None:
            return recovered

        call_id, context = await self._create_new_session(phone_hash, language)
        welcome_text = await self._generate_welcome(call_id, context, language)
        return call_id, welcome_text

    async def _try_recover_session(
        self,
        phone_hash: str,
    ) -> tuple[str, str] | None:
        """Check for an existing non-terminal session and resume it."""
        existing_call_id = await self._session.find_by_phone(phone_hash)
        if not existing_call_id:
            return None

        existing_ctx = await self._session.get(existing_call_id)
        if existing_ctx is None or existing_ctx.phase == ConversationPhase.CLOSURE:
            return None

        logger.info(
            "Dropped-call recovery: resuming session",
            extra={"call_id": existing_call_id, "phase": existing_ctx.phase.value},
        )
        self._audit.log_event(
            existing_call_id,
            "session_resumed",
            {"phone_hash": phone_hash, "language": existing_ctx.language},
        )
        return existing_call_id, get_msg("conversation", "resume", existing_ctx.language)

    async def _create_new_session(
        self,
        phone_hash: str,
        language: str,
    ) -> tuple[str, ConversationContext]:
        """Create a fresh session, record consent, and return (call_id, context)."""
        call_id = SessionManager.generate_call_id(phone_hash)
        context = await self._session.create(
            call_id=call_id,
            phone_hash=phone_hash,
            language=language,
        )
        if context is None:
            raise RuntimeError("Failed to create session")

        if self._consent is not None:
            self._consent.record_consent(
                call_id=call_id,
                consent_type="data_processing",
                granted=True,
            )
            self._audit.log_event(
                call_id,
                "consent_recorded",
                {
                    "consent_type": "data_processing",
                    "granted": True,
                    "source": "implicit_demo_default",
                },
            )

        return call_id, context

    async def _generate_welcome(
        self,
        call_id: str,
        context: ConversationContext,
        language: str,
    ) -> str:
        """Generate welcome message, persist context, and log session start."""
        try:
            response = await self._orchestrator.handle_turn(
                context,
                user_input="",
                stt_confidence=1.0,
            )
            welcome_text = response.text
        except Exception as exc:
            logger.error(
                "Welcome generation failed",
                extra={"call_id": call_id, "error": str(exc)},
            )
            welcome_text = self._default_welcome(language)

        await self._session.update(context)
        self._audit.log_event(
            call_id,
            "session_start",
            {
                "phone_hash": context.phone_number_hash,
                "language": language,
            },
        )
        return welcome_text

    async def handle_turn(
        self,
        call_id: str,
        user_text: str,
        language: str | None = None,
        stt_confidence: float = 1.0,
    ) -> str:
        """Process one user turn and return the agent's text response."""
        # Prune stale locks to prevent unbounded growth
        if len(self._turn_locks) > 1000:
            oldest_keys = list(self._turn_locks.keys())[:-500]
            for k in oldest_keys:
                del self._turn_locks[k]
        if call_id not in self._turn_locks:
            self._turn_locks[call_id] = asyncio.Lock()
        async with self._turn_locks[call_id]:
            return await self._handle_turn_locked(call_id, user_text, language, stt_confidence)

    async def _load_context_or_fail(
        self,
        call_id: str,
    ) -> ConversationContext | None:
        """Load the session context, returning ``None`` if expired or missing."""
        context = await self._session.get(call_id)
        if context is None:
            logger.warning("Session not found", extra={"call_id": call_id})
        return context

    async def _translate_to_agent_language(
        self,
        user_text: str,
        turn_language: str,
    ) -> str:
        """Translate user input to the internal agent language if needed."""
        if turn_language != _AGENT_LANG:
            return await self._translator.translate_if_needed(
                user_text,
                turn_language,
                _AGENT_LANG,
            )
        return user_text

    async def _execute_orchestrator(
        self,
        context: ConversationContext,
        agent_input: str,
        stt_confidence: float,
    ) -> AgentResponse:
        """Call the orchestrator and return its response (may raise)."""
        return await self._orchestrator.handle_turn(
            context,
            agent_input,
            stt_confidence=stt_confidence,
        )

    async def _translate_to_user_language(
        self,
        response: AgentResponse,
        turn_language: str,
    ) -> str:
        """Translate the agent response back to the user's language if needed."""
        if turn_language != _AGENT_LANG and not response.already_localized:
            return await self._translator.translate_if_needed(
                response.text,
                _AGENT_LANG,
                turn_language,
            )
        return response.text

    def _audit_turn(
        self,
        call_id: str,
        context: ConversationContext,
        masked_text: str,
        response_text: str,
        response: AgentResponse | None,
        elapsed_ms: float,
    ) -> None:
        """Log the turn and any eligibility decision to the audit trail."""
        agent_name = "orchestrator"
        if response is not None:
            agent_name = response.metadata.get("agent", "orchestrator")

        self._audit.log_turn(
            call_id=call_id,
            phase=context.phase.value,
            agent_name=agent_name,
            input_text=masked_text,
            output_text=self._mask_pii(response_text),
            latency_ms=elapsed_ms,
        )

        if response is not None and response.convergence_result is not None:
            self._audit.log_eligibility_decision(
                call_id=call_id,
                eligibility_result=context.eligibility_result,
                reviewer_result=context.reviewer_result,
                convergence_result=response.convergence_result,
                context=context,
            )

    async def _handle_turn_locked(
        self,
        call_id: str,
        user_text: str,
        language: str | None,
        stt_confidence: float,
    ) -> str:
        """Inner turn handler, called under per-session lock."""
        start = time.perf_counter()

        context = await self._load_context_or_fail(call_id)
        if context is None:
            self._turn_locks.pop(call_id, None)
            return self._session_expired_message(language or "hi-IN")

        turn_language = language or context.language
        masked_text = self._mask_pii(user_text)

        if self._sarvam_client:
            self._sarvam_client.set_active_call_id(call_id)

        agent_input = await self._translate_to_agent_language(user_text, turn_language)

        try:
            response = await self._execute_orchestrator(context, agent_input, stt_confidence)
        except Exception as exc:
            logger.error(
                "Orchestrator failed",
                extra={"call_id": call_id, "error": str(exc)[:200]},
            )
            if self._sarvam_client:
                self._sarvam_client.clear_active_call_id()
            elapsed = (time.perf_counter() - start) * 1000
            self._audit_turn(call_id, context, masked_text, "[error]", None, elapsed)
            return self._error_message(turn_language)

        response_text = await self._translate_to_user_language(response, turn_language)
        self._finalize_cost_tracking(call_id, context)
        await self._session.update(context)

        elapsed = (time.perf_counter() - start) * 1000
        self._audit_turn(call_id, context, masked_text, response_text, response, elapsed)
        return response_text

    def _finalize_cost_tracking(
        self,
        call_id: str,
        context: ConversationContext,
    ) -> None:
        """Record session cost on context and clear the active call_id."""
        if self._sarvam_client:
            session_cost = self._sarvam_client.costs.cost_for_call(call_id)
            context.metadata["session_cost_inr"] = round(session_cost, 4)
            self._sarvam_client.clear_active_call_id()

    async def get_context(self, call_id: str) -> ConversationContext | None:
        """Load and return the current context for *call_id*."""
        return await self._session.get(call_id)

    async def end_conversation(self, call_id: str) -> None:
        """Explicitly end a session and clean up Redis."""
        self._audit.log_event(call_id, "session_end")
        await self._session.delete(call_id)
        self._turn_locks.pop(call_id, None)
        logger.info("Conversation ended", extra={"call_id": call_id})

    @staticmethod
    def _default_welcome(language: str) -> str:
        """Default welcome when orchestrator fails."""
        return get_msg("conversation", "default_welcome", language)

    @staticmethod
    def _session_expired_message(language: str) -> str:
        """Message when the session is not found / expired."""
        return get_msg("conversation", "session_expired", language)

    @staticmethod
    def _error_message(language: str) -> str:
        """Generic error message in user's language."""
        return get_msg("conversation", "error", language)
