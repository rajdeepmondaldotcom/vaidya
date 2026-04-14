"""ConversationManager: the main turn-orchestration layer.

Sits between the HTTP/voice layer and the :class:`Orchestrator`.  Handles:

- Session lifecycle (create, load, save via Redis)
- PII masking on stored transcripts
- Translation between user language and agent language
- Audit logging for every turn
- STT confidence pass-through
"""

from __future__ import annotations

import logging
import time

from vaidya.agents.orchestrator import Orchestrator
from vaidya.compliance.audit import AuditTrail
from vaidya.compliance.consent import ConsentTracker
from vaidya.compliance.pii import mask_pii
from vaidya.models.conversation import ConversationContext
from vaidya.pipeline.translator import Translator
from vaidya.session.manager import SessionManager

logger = logging.getLogger(__name__)

# Internal processing language for agents (English keeps prompts stable)
_AGENT_LANG = "en-IN"


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
    ) -> None:
        self._orchestrator = orchestrator
        self._session = session_manager
        self._translator = translator
        self._audit = audit_trail
        self._consent = consent_tracker

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start_conversation(
        self,
        phone_hash: str,
        language: str = "hi-IN",
    ) -> tuple[str, str]:
        """Create a new session and return ``(call_id, welcome_message)``.

        If a session for *phone_hash* already exists within the TTL, a
        fresh session is created regardless (new call = new context).
        """
        call_id = SessionManager.generate_call_id(phone_hash)

        context = await self._session.create(
            call_id=call_id,
            phone_hash=phone_hash,
            language=language,
        )

        # Record initial consent for data processing (default for demo)
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

        # Generate welcome via orchestrator (first "turn" with empty input)
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

        # Persist the updated context (now has the welcome turn)
        await self._session.update(context)

        # Audit
        self._audit.log_event(
            call_id,
            "session_start",
            {
                "phone_hash": phone_hash,
                "language": language,
            },
        )

        return call_id, welcome_text

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    async def handle_turn(
        self,
        call_id: str,
        user_text: str,
        language: str | None = None,
        stt_confidence: float = 1.0,
    ) -> str:
        """Process one user turn and return the agent's text response.

        Steps:

        1. Load context from Redis.
        2. PII-mask the user text for storage.
        3. Translate user text to agent language if needed.
        4. Call ``orchestrator.handle_turn()``.
        5. Translate response back to user language if needed.
        6. Update context in Redis.
        7. Log audit entry.
        8. Return the response text.

        Parameters
        ----------
        call_id:
            Session identifier returned by :meth:`start_conversation`.
        user_text:
            The user's utterance (text from STT or direct text input).
        language:
            Override session language for this turn (e.g. if STT detected
            a different language).  ``None`` keeps the session language.
        stt_confidence:
            Confidence score from STT (1.0 for text input).

        Returns
        -------
        str
            The agent's response text in the user's language.
        """
        start = time.perf_counter()

        # 1. Load context
        context = await self._session.get(call_id)
        if context is None:
            logger.warning("Session not found", extra={"call_id": call_id})
            return self._session_expired_message(language or "hi-IN")

        # Override language if provided
        turn_language = language or context.language

        # 2. PII-mask for storage (agents still get raw text)
        masked_text = mask_pii(user_text)

        # 3. Translate to agent processing language if user speaks non-English
        agent_input = user_text
        if turn_language != _AGENT_LANG:
            agent_input = await self._translator.translate_if_needed(
                user_text,
                turn_language,
                _AGENT_LANG,
            )

        # 4. Call orchestrator
        try:
            response = await self._orchestrator.handle_turn(
                context,
                agent_input,
                stt_confidence=stt_confidence,
            )
        except Exception as exc:
            logger.error(
                "Orchestrator failed",
                extra={"call_id": call_id, "error": str(exc)},
            )
            elapsed = (time.perf_counter() - start) * 1000
            self._audit.log_turn(
                call_id=call_id,
                phase=context.phase.value,
                agent_name="orchestrator",
                input_text=masked_text,
                output_text="[error]",
                latency_ms=elapsed,
            )
            return self._error_message(turn_language)

        # 5. Translate response back to user language
        response_text = response.text
        if turn_language != _AGENT_LANG:
            response_text = await self._translator.translate_if_needed(
                response.text,
                _AGENT_LANG,
                turn_language,
            )

        # 6. Update context in Redis
        await self._session.update(context)

        # 7. Audit
        elapsed = (time.perf_counter() - start) * 1000
        self._audit.log_turn(
            call_id=call_id,
            phase=context.phase.value,
            agent_name=response.metadata.get("agent", "orchestrator"),
            input_text=masked_text,
            output_text=mask_pii(response_text),
            latency_ms=elapsed,
        )

        # Log eligibility decision if one was produced this turn
        if response.convergence_result is not None:
            self._audit.log_eligibility_decision(
                call_id=call_id,
                eligibility_result=context.eligibility_result,
                reviewer_result=context.reviewer_result,
                convergence_result=response.convergence_result,
                context=context,
            )

        return response_text

    # ------------------------------------------------------------------
    # Context access
    # ------------------------------------------------------------------

    async def get_context(self, call_id: str) -> ConversationContext | None:
        """Load and return the current context for *call_id*.

        Useful for the ``/conversation/{call_id}/status`` API endpoint.
        """
        return await self._session.get(call_id)

    async def end_conversation(self, call_id: str) -> None:
        """Explicitly end a session and clean up Redis."""
        self._audit.log_event(call_id, "session_end")
        await self._session.delete(call_id)
        logger.info("Conversation ended", extra={"call_id": call_id})

    # ------------------------------------------------------------------
    # Fallback messages
    # ------------------------------------------------------------------

    @staticmethod
    def _default_welcome(language: str) -> str:
        """Default welcome when orchestrator fails."""
        welcomes = {
            "hi-IN": "Namaste! Main Vaidya hoon. Aapko kya jaanna hai?",
            "ta-IN": "Vanakkam! Naan Vaidya. Ungalukku enna theriya vendum?",
            "bn-IN": "Namaskar! Ami Vaidya. Apnar ki jante hobe?",
            "en-IN": "Hello! I am Vaidya. How can I help you?",
        }
        return welcomes.get(language, welcomes["hi-IN"])

    @staticmethod
    def _session_expired_message(language: str) -> str:
        """Message when the session is not found / expired."""
        messages = {
            "hi-IN": "Aapka session khatam ho gaya. Kripya dubara call karein.",
            "ta-IN": "Ungal session mudindhuviddu. Thayavu seythu meendum azhaikavum.",
            "bn-IN": "Apnar session sesh hoye geche. Abar call korun.",
            "en-IN": "Your session has expired. Please call again.",
        }
        return messages.get(language, messages["hi-IN"])

    @staticmethod
    def _error_message(language: str) -> str:
        """Generic error message in user's language."""
        messages = {
            "hi-IN": "Maaf kijiye, thodi dikkat aa rahi hai. Kya aap phir se bata sakte hain?",
            "ta-IN": "Mannikkavum, sila prachanai. Thayavu seythu meendum sollunga?",
            "bn-IN": "Dukkhito, ektu somossa hocche. Abar bolben please?",
            "en-IN": "Sorry, something went wrong. Could you say that again?",
        }
        return messages.get(language, messages["hi-IN"])
