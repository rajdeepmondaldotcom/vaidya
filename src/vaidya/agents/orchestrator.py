"""Orchestrator: deterministic state machine that drives the conversation.

The orchestrator is NOT an LLM agent for most decisions. It is a pure Python
state machine with explicit phase transitions. LLM is used only for:
1. Completely unexpected input that doesn't fit the state machine
2. Mid-flow intent changes (user asks a question during intake)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from vaidya.agents.silence import SilenceHandler
from vaidya.i18n import get_msg
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import ConvergenceResult, EligibilityVerdict

if TYPE_CHECKING:
    from vaidya.agents.convergence import ConvergenceChecker
    from vaidya.agents.eligibility import EligibilityAgent
    from vaidya.agents.guidance import GuidanceAgent
    from vaidya.agents.intake import IntakeAgent
    from vaidya.agents.reviewer import ReviewerAgent
    from vaidya.compliance.consent import ConsentTracker
    from vaidya.pipeline.degradation import DegradationManager
    from vaidya.prompts.registry import PromptRegistry
    from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

_REPEAT_SMS_THRESHOLD = 3
_REPHRASE_TTS_RATE = 0.80
_DISTRESS_TTS_RATE = 0.85


class Orchestrator:
    """Deterministic state machine with LLM fallback for ambiguous routing."""

    def __init__(
        self,
        client: SarvamClient,
        intake: IntakeAgent,
        eligibility: EligibilityAgent,
        reviewer: ReviewerAgent,
        guidance: GuidanceAgent,
        convergence: ConvergenceChecker,
        prompts: PromptRegistry | None = None,
        fallback_model: str = "sarvam-30b",
        agent_timeout: float = 15.0,
        consent_tracker: ConsentTracker | None = None,
        degradation: DegradationManager | None = None,
    ) -> None:
        self._client = client
        self._intake = intake
        self._eligibility = eligibility
        self._reviewer = reviewer
        self._guidance = guidance
        self._convergence = convergence
        self._prompts = prompts
        self._fallback_model = fallback_model
        self._agent_timeout = agent_timeout
        self._consent_tracker = consent_tracker
        self._degradation = degradation
        self._silence_handler = SilenceHandler()

    async def handle_turn(
        self,
        context: ConversationContext,
        user_input: str,
        stt_confidence: float = 1.0,
        silence_duration_seconds: float = 0.0,
        channel: str = "text",
    ) -> AgentResponse:
        """Main entry point. Routes user input through the state machine.

        ``channel`` is "text" for the simulation / HTTP conversation API and
        "voice" for real phone calls via Pipecat. Voice short-circuits the
        wordy welcome (consent + disclaimer + open-elicitation) and jumps
        straight to the first intake question for a phone-friendly opener.
        """
        start = time.perf_counter()

        silence_response = self._check_silence(context, silence_duration_seconds)
        if silence_response is not None:
            return silence_response

        context.add_turn(
            role="user",
            text=user_input,
            raw_text=user_input,
            language=context.language,
            stt_confidence=stt_confidence,
        )

        response = await self._route_by_phase(context, user_input, stt_confidence, channel)

        elapsed = (time.perf_counter() - start) * 1000
        response.metadata["orchestrator_latency_ms"] = round(elapsed, 1)
        response.metadata["phase"] = context.phase.value

        context.add_turn(
            role="assistant",
            text=response.text,
            raw_text=response.text,
            language=context.language,
        )

        return response

    def _check_silence(
        self,
        context: ConversationContext,
        silence_duration_seconds: float,
    ) -> AgentResponse | None:
        if silence_duration_seconds <= 0:
            return None

        if self._silence_handler.should_end_call(silence_duration_seconds):
            context.phase = ConversationPhase.CLOSURE
            return AgentResponse(
                text=get_msg("orchestrator", "closure", context.language),
                phase_transition=ConversationPhase.CLOSURE,
                metadata={"silence_end_call": True, "trigger_callback": True},
            )

        silence_prompt = self._silence_handler.get_silence_response(
            silence_duration_seconds,
            context.language,
        )
        if silence_prompt is not None:
            return AgentResponse(
                text=silence_prompt,
                metadata={"silence_prompt_seconds": silence_duration_seconds},
            )

        return None

    async def _route_by_phase(
        self,
        context: ConversationContext,
        user_input: str,
        stt_confidence: float,
        channel: str = "text",
    ) -> AgentResponse:
        match context.phase:
            case ConversationPhase.WELCOME:
                return await self._handle_welcome(context, user_input, stt_confidence, channel)
            case ConversationPhase.OPEN_ELICITATION:
                return await self._handle_open_elicitation(context, user_input)
            case ConversationPhase.INTAKE:
                return await self._handle_intake(context, user_input)
            case ConversationPhase.PROCESSING:
                return await self._handle_processing(context)
            case ConversationPhase.RESULTS:
                return await self._handle_results(context, user_input)
            case ConversationPhase.GUIDANCE:
                return await self._handle_guidance(context, user_input)
            case ConversationPhase.CLOSURE:
                return await self._handle_closure(context, user_input)
            case _:
                logger.warning(
                    "Unrecognized conversation phase, falling back to LLM",
                    extra={
                        "phase": context.phase.value
                        if hasattr(context.phase, "value")
                        else str(context.phase),
                        "call_id": context.call_id,
                    },
                )
                return await self._llm_fallback(context, user_input)

    async def _handle_welcome(
        self,
        context: ConversationContext,
        user_input: str,
        stt_confidence: float,
        channel: str = "text",
    ) -> AgentResponse:
        """Phase 1: Language selection (always first) + welcome handshake.

        Regardless of channel, the caller's very first interaction is a
        language-selection turn. We never assume a language: even if the
        API client passed ``language=hi-IN`` to ``start_conversation``,
        that is treated as the *default for the opening prompt only*, and
        the user is immediately asked which language they'd like to use.

        Two-step handshake:

        1. **Turn 1 (no user input yet):** speak a multilingual greeting
           enumerating every supported voice language. Stay in WELCOME.
           ``context.metadata["awaiting_language"] = True`` tells
           :class:`ConversationManager` to skip inbound translation on the
           next turn -- we need the raw user utterance to detect language.

        2. **Turn 2 (user has responded):**
           - Voice: the processor has already called ``switch_language``
             based on the STT-tagged language; ``context.language`` is
             now the caller's choice. We acknowledge in that language
             and ask intake Q1 in the same utterance -> INTAKE.
           - Text: we lexically detect the language from the raw reply
             (names / autonyms / menu numbers). On success we switch the
             session language, acknowledge + speak the disclaimer, and
             transition to OPEN_ELICITATION. On failure (no confident
             match) we re-prompt and stay in WELCOME.

        Consent is recorded silently (for the audit trail) on the very
        first turn but never spoken -- the disclaimer covers it on text,
        and voice onboarding IVR already discloses recording.
        """
        # Fire-and-forget structural consent record; we never speak it.
        self._record_consent_if_needed(context, speak=False)

        user_text = (user_input or "").strip()

        # Turn 1 of the handshake: no user input yet.
        if not user_text:
            context.metadata["awaiting_language"] = True
            key = "welcome_voice" if channel == "voice" else "welcome_text"
            return AgentResponse(
                text=get_msg("orchestrator", key, context.language),
                already_localized=True,
            )

        # Turn 2: user has picked a language.
        if channel == "voice":
            # The voice processor has already switched ``context.language``
            # to the STT-detected language via ``switch_language``.
            new_lang = context.language
            context.metadata["awaiting_language"] = False
            context.phase = ConversationPhase.INTAKE
            context.intake_question_index = 1
            confirmation = get_msg("orchestrator", "language_confirmed", new_lang)
            q1 = get_msg("orchestrator", "intake_q1_voice", new_lang)
            return AgentResponse(
                text=f"{confirmation} {q1}",
                phase_transition=ConversationPhase.INTAKE,
                already_localized=True,
            )

        # Text channel: lexically detect the chosen language.
        from vaidya.voice.language import detect_language_from_text

        detected = detect_language_from_text(user_text)
        if detected is None:
            # Couldn't tell -- re-prompt in a universal, short message and
            # stay in WELCOME. ``awaiting_language`` stays True so the next
            # user turn is also passed through untranslated.
            context.metadata["awaiting_language"] = True
            return AgentResponse(
                text=get_msg(
                    "orchestrator", "language_not_understood", context.language
                ),
                already_localized=True,
            )

        # Commit the detected language onto the session and move on.
        new_lang = detected.value
        context.language = new_lang
        context.metadata["awaiting_language"] = False
        context.metadata["language_confirmed"] = True
        context.phase = ConversationPhase.OPEN_ELICITATION

        confirmation = get_msg("orchestrator", "language_confirmed", new_lang)
        welcome = get_msg("orchestrator", "welcome", new_lang)
        disclaimer = get_msg("orchestrator", "disclaimer", new_lang)
        return AgentResponse(
            text=f"{confirmation} {welcome} {disclaimer}",
            phase_transition=ConversationPhase.OPEN_ELICITATION,
            already_localized=True,
        )

    def _record_consent_if_needed(
        self,
        context: ConversationContext,
        *,
        speak: bool = True,
    ) -> str:
        """Record consent in the tracker and (optionally) return a spoken prefix.

        On voice calls ``speak=False`` records consent silently for audit
        purposes without narrating the 17-word consent request to the caller.
        """
        if context.metadata.get("consent_asked"):
            return ""

        context.metadata["consent_asked"] = True

        if self._consent_tracker is not None:
            self._consent_tracker.record_consent(
                call_id=context.call_id,
                consent_type="recording",
                granted=True,
            )

        if not speak:
            return ""

        consent_text = get_msg("orchestrator", "consent_ask", context.language)
        return f"{consent_text} "

    async def _handle_open_elicitation(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 2: Listen to free-form statement, then transition to intake."""
        context.phase = ConversationPhase.INTAKE
        context.intake_question_index = 0
        return await self._intake.safe_process(context, user_input)

    async def _handle_intake(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 3: Structured intake (5 questions)."""
        response = await self._intake.safe_process(context, user_input)

        if response.updated_profile:
            context.user_profile = response.updated_profile

        if context.user_profile.required_fields_complete or context.intake_question_index >= 5:
            return await self._transition_to_processing(context)

        if context.emotional_distress_detected:
            return await self._fast_track_distress(context)

        return response

    async def _fast_track_distress(self, context: ConversationContext) -> AgentResponse:
        logger.info(
            "Emotional distress detected, fast-tracking to processing",
            extra={"call_id": context.call_id},
        )
        empathy = get_msg("orchestrator", "empathy", context.language)
        processing_response = await self._transition_to_processing(context)
        processing_response.text = f"{empathy} {processing_response.text}"
        processing_response.metadata["tts_speech_rate_factor"] = _DISTRESS_TTS_RATE
        processing_response.metadata["emotional_distress_mode"] = True
        return processing_response

    async def _transition_to_processing(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Transition from intake to processing phase."""
        context.phase = ConversationPhase.PROCESSING
        filler = get_msg("orchestrator", "processing_filler", context.language)
        processing_response = await self._run_eligibility_and_review(context)
        processing_response.text = f"{filler}\n\n{processing_response.text}"
        return processing_response

    async def _handle_processing(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Phase 4: Run Eligibility + Reviewer in parallel.

        Guards against re-entry: if results already exist, skip to results.
        """
        if context.convergence_result is not None:
            context.phase = ConversationPhase.RESULTS
            return await self._guidance.safe_process(context, "")
        return await self._run_eligibility_and_review(context)

    async def _run_eligibility_and_review(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Core parallel execution: Eligibility + Reviewer -> Convergence -> Guidance."""
        eligibility_result, reviewer_result = await self._execute_agents(context)

        context.eligibility_result = eligibility_result
        context.reviewer_result = reviewer_result

        error_response = self._run_convergence(context, eligibility_result, reviewer_result)
        if error_response is not None:
            return error_response

        return await self._transition_to_results_with_guidance(context)

    async def _execute_agents(
        self,
        context: ConversationContext,
    ) -> tuple:
        """Run eligibility + reviewer tasks with timeout; either result may be None."""
        try:
            eligibility_task, reviewer_task = self._create_agent_tasks(context)
            tasks = [eligibility_task] + ([reviewer_task] if reviewer_task else [])
            done, pending = await self._await_agent_tasks(tasks, context)

            eligibility_result = self._collect_task_result(
                eligibility_task,
                done,
                "eligibility",
                context.call_id,
            )
            reviewer_result = (
                self._collect_task_result(
                    reviewer_task,
                    done,
                    "reviewer",
                    context.call_id,
                )
                if reviewer_task
                else None
            )

            return eligibility_result, reviewer_result

        except Exception:
            logger.error(
                "Eligibility/reviewer execution failed",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return None, None

    def _create_agent_tasks(self, context: ConversationContext) -> tuple:
        skip_reviewer = self._should_skip_reviewer(context)

        eligibility_task = asyncio.create_task(
            self._eligibility.safe_process(context, ""),
            name="eligibility",
        )
        reviewer_task = None
        if not skip_reviewer:
            reviewer_task = asyncio.create_task(
                self._reviewer.safe_process(context, ""),
                name="reviewer",
            )

        return eligibility_task, reviewer_task

    def _should_skip_reviewer(self, context: ConversationContext) -> bool:
        if self._degradation is None:
            return False

        from vaidya.pipeline.degradation import DegradationLevel

        if self._degradation.level >= DegradationLevel.NO_REVIEWER:
            logger.info(
                "Skipping reviewer: degradation level >= NO_REVIEWER",
                extra={"call_id": context.call_id},
            )
            return True
        return False

    async def _await_agent_tasks(
        self,
        tasks: list,
        context: ConversationContext,
    ) -> tuple:
        done, pending = await asyncio.wait(
            tasks,
            timeout=self._agent_timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        for task in pending:
            task.cancel()
            logger.warning(
                "Agent timed out", extra={"agent": task.get_name(), "call_id": context.call_id}
            )
            self._record_degradation(task.get_name(), success=False)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        return done, pending

    def _collect_task_result(self, task: asyncio.Task, done: set, agent_name: str, call_id: str):
        result_attr = f"{agent_name}_result"

        if task in done and not task.cancelled():
            try:
                response = task.result()
                self._record_degradation(agent_name, success=True)
                return getattr(response, result_attr)
            except Exception:
                logger.warning(
                    "%s task raised", agent_name, extra={"call_id": call_id}, exc_info=True
                )

        self._record_degradation(agent_name, success=False)
        return None

    def _record_degradation(self, agent_name: str, *, success: bool) -> None:
        if self._degradation is None:
            return
        if success:
            self._degradation.record_success(agent_name)
        else:
            self._degradation.record_failure(agent_name)

    def _run_convergence(
        self,
        context: ConversationContext,
        eligibility_result,
        reviewer_result,
    ) -> AgentResponse | None:
        """Returns error AgentResponse if both agents failed, else None."""
        try:
            if eligibility_result and reviewer_result:
                context.convergence_result = self._convergence.check(
                    eligibility_result,
                    reviewer_result,
                    context,
                )
            elif eligibility_result:
                context.convergence_result = self._build_single_agent_convergence(
                    eligibility_result,
                )
                context.metadata["reviewer_unavailable"] = True
            else:
                return AgentResponse(
                    text=get_msg("orchestrator", "processing_filler", context.language),
                    error="eligibility_processing_failed",
                )
        except Exception:
            logger.error(
                "Convergence check failed",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return AgentResponse(
                text=get_msg("orchestrator", "processing_filler", context.language),
                error="convergence_check_failed",
            )

        return None

    def _build_single_agent_convergence(self, eligibility_result) -> ConvergenceResult:
        return ConvergenceResult(
            agreed_eligible=[
                m for m in eligibility_result.matches if m.verdict == EligibilityVerdict.ELIGIBLE
            ],
            agreed_ineligible=[
                m.scheme_id
                for m in eligibility_result.matches
                if m.verdict == EligibilityVerdict.INELIGIBLE
            ],
            disagreements=[],
            conservative_eligible=[],
        )

    async def _transition_to_results_with_guidance(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Transition to results phase and generate guidance response."""
        context.phase = ConversationPhase.RESULTS
        try:
            return await self._guidance.safe_process(context, "")
        except Exception:
            logger.error(
                "Guidance generation failed in orchestrator",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return AgentResponse(
                text=get_msg("orchestrator", "processing_filler", context.language),
                error="guidance_generation_failed",
            )

    async def _handle_results(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 5: Results delivery -- one scheme at a time."""
        if "scheme_delivery_index" not in context.metadata:
            context.metadata["scheme_delivery_index"] = 0

        eligible = context.convergence_result.all_eligible if context.convergence_result else []
        current_idx = context.metadata.get("scheme_delivery_index", 0)
        if not eligible or current_idx >= len(eligible):
            context.phase = ConversationPhase.GUIDANCE
            return await self._handle_guidance(context, user_input)

        if self._user_declines(user_input, context.language):
            context.phase = ConversationPhase.GUIDANCE
            return await self._guidance.safe_process(context, user_input)

        return await self._deliver_next_scheme(context, user_input)

    def _user_declines(self, user_input: str, language: str) -> bool:
        neg_words = get_msg("orchestrator", "negative_words", language).split(",")
        return any(w in user_input.lower() for w in neg_words)

    async def _deliver_next_scheme(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        eligible = context.convergence_result.all_eligible if context.convergence_result else []
        current_idx: int = context.metadata["scheme_delivery_index"]

        if current_idx >= len(eligible):
            context.phase = ConversationPhase.GUIDANCE
            return await self._guidance.safe_process(context, user_input)

        context.metadata["scheme_delivery_index"] = current_idx + 1
        scheme_response = await self._guidance.safe_process(
            context,
            f"__deliver_scheme_index:{current_idx}",
        )

        if current_idx + 1 < len(eligible):
            more_prompt = get_msg("orchestrator", "more_schemes", context.language)
            scheme_response.text = f"{scheme_response.text}\n\n{more_prompt}"
        else:
            context.phase = ConversationPhase.GUIDANCE
            scheme_response.phase_transition = ConversationPhase.GUIDANCE

        return scheme_response

    async def _handle_guidance(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 6: Action guidance -- documents, CSC directions."""
        response = await self._guidance.safe_process(context, user_input)

        cont_words = get_msg("orchestrator", "continue_words", context.language).split(",")
        if not any(w in user_input.lower() for w in cont_words):
            context.phase = ConversationPhase.CLOSURE
            response.phase_transition = ConversationPhase.CLOSURE

        return response

    async def _handle_closure(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 7: Closure & follow-up."""
        restart_words = get_msg("orchestrator", "restart_words", context.language).split(",")
        if any(w in user_input.lower() for w in restart_words):
            context.phase = ConversationPhase.OPEN_ELICITATION
            return AgentResponse(
                text=get_msg("orchestrator", "restart", context.language),
                phase_transition=ConversationPhase.OPEN_ELICITATION,
            )

        return AgentResponse(text=get_msg("orchestrator", "closure", context.language))

    async def _llm_fallback(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """LLM-based routing for input that doesn't fit the state machine."""
        try:
            category, brief_answer = await self._classify_input(context, user_input)

            match category:
                case "ON_TOPIC":
                    return AgentResponse(text=brief_answer or user_input)
                case "QUESTION":
                    return AgentResponse(text=brief_answer)
                case "REPEAT":
                    return await self._handle_repeat(context)
                case "RESTART":
                    return self._handle_restart(context)
                case "END":
                    return self._handle_end(context)
                case _:
                    return AgentResponse(
                        text=get_msg("orchestrator", "fallback_pardon", context.language),
                    )

        except Exception:
            logger.error(
                "LLM fallback failed",
                extra={"call_id": context.call_id},
                exc_info=True,
            )
            return AgentResponse(
                text=get_msg("orchestrator", "fallback_error", context.language),
                error="llm_fallback_failed",
            )

    async def _classify_input(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> tuple[str, str]:
        """Build prompt, call LLM, parse category and answer."""
        recent = context.transcript[-3:] if context.transcript else []
        recent_text = "\n".join(f"[{t.role}] {t.text}" for t in recent)

        from vaidya.prompts import registry as prompts

        system = prompts.render(
            "orchestrator_fallback",
            phase=context.phase.value,
            language=context.language,
            recent_turns=recent_text,
            user_input=user_input,
        )

        result = await self._client.chat_json(
            self._fallback_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user_input}],
        )

        return result.get("category", "UNKNOWN"), result.get("brief_answer", "")

    async def _handle_repeat(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Repeat escalation: rephrase -> simplify -> offer SMS."""
        repeat_count: int = context.metadata.get("repeat_count", 0) + 1
        context.metadata["repeat_count"] = repeat_count

        last = next(
            (t for t in reversed(context.transcript) if t.role == "assistant"),
            None,
        )
        last_text = (
            last.text if last else get_msg("orchestrator", "fallback_pardon", context.language)
        )

        if repeat_count >= _REPEAT_SMS_THRESHOLD:
            return AgentResponse(
                text=get_msg("orchestrator", "offer_sms", context.language),
                metadata={"repeat_escalation": "sms_offered"},
            )

        rephrased = await self._rephrase_for_user(
            last_text,
            context.language,
            simple=(repeat_count == 2),
        )
        escalation = "simplified" if repeat_count == 2 else "rephrased"
        return AgentResponse(
            text=rephrased,
            metadata={
                "repeat_escalation": escalation,
                "tts_speech_rate_factor": _REPHRASE_TTS_RATE,
            },
        )

    async def _rephrase_for_user(
        self,
        text: str,
        language: str,
        *,
        simple: bool = False,
    ) -> str:
        """Call LLM to rephrase text, with optional simplification."""
        simplify_clause = (
            "very simple words. Use short sentences"
            if simple
            else "different words. Keep the same meaning"
        )
        instruction = (
            f"Rephrase the following in {language} using {simplify_clause}. "
            'Return JSON: {"rephrased": "..."}'
        )
        try:
            result = await self._client.chat_json(
                self._fallback_model,
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
            )
            return result.get("rephrased", text)
        except Exception:
            return text

    def _handle_restart(self, context: ConversationContext) -> AgentResponse:
        context.phase = ConversationPhase.OPEN_ELICITATION
        return AgentResponse(
            text=get_msg("orchestrator", "restart_brief", context.language),
            phase_transition=ConversationPhase.OPEN_ELICITATION,
        )

    def _handle_end(self, context: ConversationContext) -> AgentResponse:
        context.phase = ConversationPhase.CLOSURE
        return AgentResponse(
            text=get_msg("orchestrator", "closure", context.language),
            phase_transition=ConversationPhase.CLOSURE,
        )
