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

from vaidya.agents.constants import PATIENT_SILENCE_STEPS, SILENCE_STEPS
from vaidya.agents.orchestrator import Orchestrator
from vaidya.compliance.audit import AuditTrail
from vaidya.compliance.consent import ConsentTracker
from vaidya.compliance.pii import mask_pii as _default_mask_pii
from vaidya.i18n import get_msg
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.pipeline.translator import Translator
from vaidya.session.manager import SessionManager
from vaidya.voice.language import is_voice_language, normalize_language

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
        channel: str = "text",
    ) -> tuple[str, str]:
        """Create a new session and return ``(call_id, welcome_message)``.

        Resumes an existing non-terminal session if one exists for
        *phone_hash* (dropped-call recovery). ``channel`` is propagated to
        the orchestrator so voice calls get a short greet+Q1 welcome.
        """
        recovered = await self._try_recover_session(phone_hash)
        if recovered is not None:
            return recovered

        call_id, context = await self._create_new_session(phone_hash, language)
        welcome_text = await self._generate_welcome(call_id, context, language, channel)
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
        # WELCOME sessions hold no progress worth resuming — a fresh start
        # is clearer for the caller than "your call got cut".
        if existing_ctx is None or existing_ctx.phase in (
            ConversationPhase.CLOSURE,
            ConversationPhase.WELCOME,
        ):
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
        resume_msg = get_msg("conversation", "resume", existing_ctx.language)
        # Re-ask whatever the bot last said so the caller isn't left in
        # dead air wondering what to do.
        last_bot_text = next(
            (t.text for t in reversed(existing_ctx.transcript) if t.role == "assistant"),
            "",
        )
        if last_bot_text:
            resume_msg = f"{resume_msg} {last_bot_text}"
        return existing_call_id, resume_msg

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

        # Every new session starts by asking the caller which language they'd
        # like to use. The flag tells ``_handle_turn_locked`` to skip inbound
        # translation until the language has been confirmed.
        context.metadata["awaiting_language"] = True

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
        channel: str = "text",
    ) -> str:
        """Generate welcome message, persist context, and log session start."""
        try:
            response = await self._orchestrator.handle_turn(
                context,
                user_input="",
                stt_confidence=1.0,
                channel=channel,
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
        channel: str = "text",
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
            return await self._handle_turn_locked(
                call_id, user_text, language, stt_confidence, channel
            )

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
        channel: str = "text",
    ) -> AgentResponse:
        """Call the orchestrator and return its response (may raise)."""
        return await self._orchestrator.handle_turn(
            context,
            agent_input,
            stt_confidence=stt_confidence,
            channel=channel,
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
        channel: str = "text",
    ) -> str:
        """Inner turn handler, called under per-session lock."""
        start = time.perf_counter()

        context = await self._load_context_or_fail(call_id)
        if context is None:
            self._turn_locks.pop(call_id, None)
            return self._session_expired_message(language or "hi-IN")

        pre_turn_language = context.language
        turn_language = language or context.language
        masked_text = self._mask_pii(user_text)

        if self._sarvam_client:
            self._sarvam_client.set_active_call_id(call_id)

        # On the language-selection turn the caller's input is in *their*
        # language (which we haven't yet confirmed), so translating it
        # against the session default would corrupt the detection input.
        # Pass the raw utterance through in that case.
        if context.metadata.get("awaiting_language"):
            agent_input = user_text
        else:
            agent_input = await self._translate_to_agent_language(user_text, turn_language)

        try:
            response = await self._execute_orchestrator(
                context, agent_input, stt_confidence, channel
            )
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

        # If the orchestrator switched the session language (e.g. the user
        # picked Tamil on the welcome turn), the outbound response is already
        # in the new language -- don't translate it back to the pre-turn one.
        effective_outbound_language = context.language
        response_text = await self._translate_to_user_language(
            response, effective_outbound_language
        )

        if context.language != pre_turn_language:
            self._audit.log_event(
                call_id,
                "language_switched",
                {"from": pre_turn_language, "to": context.language, "channel": channel},
            )

        self._finalize_cost_tracking(call_id, context)
        elapsed = (time.perf_counter() - start) * 1000
        self._remember_turn_metadata(
            context,
            response,
            stt_confidence=stt_confidence,
            channel=channel,
            turn_language=effective_outbound_language,
            elapsed_ms=elapsed,
        )
        await self._session.update(context)

        self._audit_turn(call_id, context, masked_text, response_text, response, elapsed)
        return response_text

    @staticmethod
    def _remember_turn_metadata(
        context: ConversationContext,
        response: AgentResponse,
        *,
        stt_confidence: float,
        channel: str,
        turn_language: str,
        elapsed_ms: float,
    ) -> None:
        response.metadata["conversation_latency_ms"] = round(elapsed_ms, 1)
        response.metadata["stt_confidence"] = round(stt_confidence, 3)
        response.metadata["channel"] = channel
        response.metadata["language"] = turn_language
        context.metadata["last_turn_metadata"] = dict(response.metadata)

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

    async def handle_silence(
        self,
        call_id: str,
        elapsed_seconds: float,
    ) -> tuple[str, bool]:
        """Produce a silence-escalation utterance for the voice edge.

        Returns ``(spoken_text, is_terminal)``. At 6s this is a gentle nudge;
        at 12s it is a phase-aware reprompt:

        - **WELCOME phase** (caller has not yet picked a language): a short
          universal English prompt listing the supported voice languages. We
          deliberately do *not* re-play the full welcome; that feels robotic.
        - **Any other phase**: the standard prefix + the last assistant
          question from the transcript, so the caller hears the exact
          question they were asked.

        At 20s it is a closure line and ``is_terminal=True``, telling the
        caller to hang the call up cleanly.
        """
        context = await self._session.get(call_id)
        language = context.language if context is not None else "hi-IN"
        steps = self._silence_steps_for_context(context)
        step = next(
            (s for s in steps if abs(elapsed_seconds - s[0]) < 1e-6),
            None,
        )
        if step is None:
            return "", False

        _threshold, key, terminal = step

        # Phase-specific short reprompt for language selection.
        if (
            key == "silence_reprompt_prefix"
            and context is not None
            and context.phase == ConversationPhase.WELCOME
        ):
            return get_msg("orchestrator", "silence_welcome_reprompt", language), terminal

        phrase = get_msg("orchestrator", key, language)
        spoken = phrase
        if key == "silence_reprompt_prefix" and context is not None:
            last_question = next(
                (t.text for t in reversed(context.transcript) if t.role == "assistant"),
                "",
            )
            if last_question:
                spoken = f"{phrase}{last_question}"
        if terminal and context is not None:
            context.phase = ConversationPhase.CLOSURE
            await self._session.update(context)
        return spoken, terminal

    async def voice_silence_steps(self, call_id: str) -> list[tuple[float, str, bool]]:
        """Return the active silence schedule for the voice edge."""
        context = await self._session.get(call_id)
        return self._silence_steps_for_context(context)

    @staticmethod
    def _silence_steps_for_context(
        context: ConversationContext | None,
    ) -> list[tuple[float, str, bool]]:
        if context is not None and context.metadata.get("silence_schedule") == "patient":
            return PATIENT_SILENCE_STEPS
        return SILENCE_STEPS

    async def switch_language(self, call_id: str, new_language: str) -> bool:
        """Persist a detected voice language into the session.

        No-ops (returning False) if the language is unsupported for voice
        or already matches the session. Returns True when a switch occurred.
        """
        if not is_voice_language(new_language):
            return False
        normalized = normalize_language(new_language).value
        context = await self._session.get(call_id)
        if context is None:
            return False
        if context.language == normalized:
            # Speaking the session's default language is still a language
            # choice: confirm it so the welcome gate stops re-prompting
            # callers who never say a language name.
            if not context.metadata.get("language_confirmed"):
                context.metadata["language_confirmed"] = True
                context.metadata["language_source"] = "stt"
                await self._session.update(context)
            return False
        previous = context.language
        context.language = normalized
        context.metadata["language_confirmed"] = True
        context.metadata["language_source"] = "stt"
        await self._session.update(context)
        self._audit.log_event(
            call_id,
            "language_switched",
            {"from": previous, "to": normalized},
        )
        return True

    async def mark_voice_disconnected(self, call_id: str) -> None:
        """Record transport disconnect without deleting the recoverable session.

        Voice calls can drop because of network or carrier failures. Keeping
        the Redis session until TTL preserves dropped-call recovery for the
        same hashed caller, while removing only the in-process lock avoids
        memory growth after the WebSocket is gone.
        """
        self._audit.log_event(call_id, "voice_stream_disconnected")
        self._turn_locks.pop(call_id, None)

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
