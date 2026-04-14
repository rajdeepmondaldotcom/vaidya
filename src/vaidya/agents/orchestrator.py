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

from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import ConvergenceResult, EligibilityVerdict
from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import SARVAM_30B

if TYPE_CHECKING:
    from vaidya.agents.convergence import ConvergenceChecker
    from vaidya.agents.eligibility import EligibilityAgent
    from vaidya.agents.guidance import GuidanceAgent
    from vaidya.agents.intake import IntakeAgent
    from vaidya.agents.reviewer import ReviewerAgent
    from vaidya.compliance.consent import ConsentTracker
    from vaidya.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)

# Welcome messages per language
_WELCOME = {
    "hi-IN": (
        "Namaste! Main Vaidya hoon, aapka sarkaari swasthya yojana sahayak. "
        "Aap apni bhaasha mein baat kar sakte hain. Aapko kya jaanna hai?"
    ),
    "ta-IN": (
        "Vanakkam! Naan Vaidya, ungal arasanga sugadhara thittam uthaviyalar. "
        "Ungal mozhiyil pesalaam. Ungalukku enna theriya vendum?"
    ),
    "bn-IN": (
        "Namaskar! Ami Vaidya, apnar sarkaari swasthya yojana sahayak. "
        "Apni apnar bhasay kotha bolte paren. Apnar ki jante hobe?"
    ),
    "en-IN": (
        "Hello! I am Vaidya, your government healthcare scheme assistant. "
        "You can speak in your language. How can I help you?"
    ),
}

_LANGUAGE_CONFIRM = {
    "hi-IN": "Kya aap Hindi mein baat karna chahenge?",
    "ta-IN": "Tamil-il peasuveergala?",
    "bn-IN": "Apni ki Banglay kotha bolben?",
}

_PROCESSING_FILLER = {
    "hi-IN": "Ek minute, aapke liye sahi yojana dhundh raha hoon...",
    "ta-IN": "Oru nimidham, ungalukku poruthaamaana thittam thedikondirukkiren...",
    "bn-IN": "Ek minute, apnar jonyo sothik yojana khunjchi...",
}

_DISCLAIMER = {
    "hi-IN": "Yeh AI sahayak hai. Final verification Jan Seva Kendra mein hoga.",
    "ta-IN": "Ithu AI uthaviyalar. Iruthi sari paarthal Jan Seva Kendra-vil nadakkum.",
    "bn-IN": "Eta AI sahayak. Final verification Jan Seva Kendra-te hobe.",
}

_CLOSURE = {
    "hi-IN": (
        "Aapki madad karke achha laga. Kya SMS mein yeh jaankari chahiye? "
        "Agar aur koi sawaal ho toh dubara call kar sakte hain. Dhanyavaad!"
    ),
    "ta-IN": (
        "Ungalukku uthavi seyyvathu makkizhchi. SMS-il ithu thagaval vendum-a? "
        "Veru kelvi irunthal meendum azhaikkalaam. Nandri!"
    ),
    "bn-IN": (
        "Apnake sahajyo korte pere bhalo laglo. SMS-e ei tothyo chai? "
        "Aar kono proshno thakle abar call korte paren. Dhonnobad!"
    ),
}

# --- Gap 1: Consent language ---
_CONSENT_ASK = {
    "hi-IN": "Yeh call aapki madad ke liye record ho sakti hai. Kya aap sahmat hain?",
    "ta-IN": "Ithu azhaippu pathivu seyyappadalam. Neenga oppukolgireergala?",
    "bn-IN": "Ei call record hote pare. Apni raji achen?",
    "en-IN": "This call may be recorded to assist you. Do you consent?",
}

# --- Gap 3: Repeat-request messages ---
_OFFER_SMS = {
    "hi-IN": "Kya main aapko SMS bhej doon? Usme sab likha hoga.",
    "ta-IN": "Naan ungalukku SMS anuppattuma? Athil ellaam ezhuthiyirukkum.",
    "bn-IN": "Ami ki apnake SMS pathiye di? Tate sob lekha thakbe.",
}

# --- Gap 4: Multi-scheme prompt ---
_MORE_SCHEMES = {
    "hi-IN": "Ek aur yojana hai. Sunna chahenge?",
    "ta-IN": "Innum oru thittam irukkirathu. Ketka virumbugireergala?",
    "bn-IN": "Aro ekta yojana ache. Shunte chaan?",
}

# --- Gap 5: Empathy message for emotional distress ---
_EMPATHY = {
    "hi-IN": "Main samajh sakta hoon ki yeh mushkil waqt hai. Aap sahi jagah aaye hain.",
    "ta-IN": "Ithu kashtamana neram endru puriyum. Neenga sari-yana idathil irukkeergal.",
    "bn-IN": "Ami bujhte parchi ei kothina shomoy. Apni thik jayga-y eshechen.",
}


# --- Gap 2: Silence / timeout handler ---
class SilenceHandler:
    """Escalating silence handling per PRD Section 3.2.

    Thresholds:
      - 0-3s  : natural pause, no action
      - 5s    : reassuring prompt
      - 10s   : repeat question in simpler words
      - 15s   : connection-loss message, offer callback
      - 20s+  : end call, trigger callback
    """

    PROMPTS: dict[int, dict[str, str]] = {
        5: {
            "hi-IN": "Main sun raha hoon, aap boliye",
            "ta-IN": "Naan ketkiren, sollunga",
            "bn-IN": "Ami shunchi, apni bolun",
        },
        10: {
            "hi-IN": "Kya aap sun rahe hain? Main phir se poochta hoon",
            "ta-IN": "Neenga ketkireergala? Naan meendum ketkiren",
            "bn-IN": "Apni ki shunchhen? Ami abar jigges korchi",
        },
        15: {
            "hi-IN": "Kya line kat gayi? Main callback kar doonga",
            "ta-IN": "Line cut aachcha? Naan thirumba azhaikiren",
            "bn-IN": "Line ki kete gelo? Ami callback korbo",
        },
    }

    END_CALL_THRESHOLD: float = 20.0

    def get_silence_response(
        self,
        silence_seconds: float,
        language: str,
    ) -> str | None:
        """Return the appropriate prompt for the given silence duration.

        Returns ``None`` when no action is needed (< 5s) or when the call
        should be ended (>= 20s) -- the caller must check
        ``should_end_call()`` separately for the termination case.
        """
        if silence_seconds >= self.END_CALL_THRESHOLD:
            # Caller should end the call and schedule a callback
            return None

        # Walk thresholds in descending order, pick the first that applies
        for threshold in sorted(self.PROMPTS, reverse=True):
            if silence_seconds >= threshold:
                prompts = self.PROMPTS[threshold]
                return prompts.get(language, prompts["hi-IN"])

        return None

    def should_end_call(self, silence_seconds: float) -> bool:
        """Return True when silence has exceeded the end-call threshold."""
        return silence_seconds >= self.END_CALL_THRESHOLD


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
        fallback_model: str = SARVAM_30B,
        agent_timeout: float = 15.0,
        consent_tracker: ConsentTracker | None = None,
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
        self._silence_handler = SilenceHandler()

    async def handle_turn(
        self,
        context: ConversationContext,
        user_input: str,
        stt_confidence: float = 1.0,
        silence_duration_seconds: float = 0.0,
    ) -> AgentResponse:
        """Main entry point. Routes user input through the state machine.

        Parameters
        ----------
        context:
            Mutable conversation state for this call.
        user_input:
            Transcribed user speech (may be empty on silence-only turns).
        stt_confidence:
            STT engine confidence in ``[0, 1]``.
        silence_duration_seconds:
            How long the user was silent before speaking (reported by the
            voice pipeline). Used for escalating silence handling.
        """
        start = time.perf_counter()

        # --- Gap 2: Silence handling ---
        if silence_duration_seconds > 0:
            if self._silence_handler.should_end_call(silence_duration_seconds):
                context.phase = ConversationPhase.CLOSURE
                lang = context.language
                return AgentResponse(
                    text=_CLOSURE.get(lang, _CLOSURE["hi-IN"]),
                    phase_transition=ConversationPhase.CLOSURE,
                    metadata={
                        "silence_end_call": True,
                        "trigger_callback": True,
                    },
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

        # Record user turn
        context.add_turn(
            role="user",
            text=user_input,
            raw_text=user_input,
            language=context.language,
            stt_confidence=stt_confidence,
        )

        match context.phase:
            case ConversationPhase.WELCOME:
                response = await self._handle_welcome(context, user_input, stt_confidence)
            case ConversationPhase.OPEN_ELICITATION:
                response = await self._handle_open_elicitation(context, user_input)
            case ConversationPhase.INTAKE:
                response = await self._handle_intake(context, user_input)
            case ConversationPhase.PROCESSING:
                response = await self._handle_processing(context)
            case ConversationPhase.RESULTS:
                response = await self._handle_results(context, user_input)
            case ConversationPhase.GUIDANCE:
                response = await self._handle_guidance(context, user_input)
            case ConversationPhase.CLOSURE:
                response = await self._handle_closure(context, user_input)
            case _:
                response = await self._llm_fallback(context, user_input)

        elapsed = (time.perf_counter() - start) * 1000
        response.metadata["orchestrator_latency_ms"] = round(elapsed, 1)
        response.metadata["phase"] = context.phase.value

        # Record assistant turn
        context.add_turn(
            role="assistant",
            text=response.text,
            raw_text=response.text,
            language=context.language,
        )

        return response

    async def _handle_welcome(
        self,
        context: ConversationContext,
        user_input: str,
        stt_confidence: float,
    ) -> AgentResponse:
        """Phase 1: Welcome & language lock.

        Per PRD 11.1 -- consent is collected at the start of every call
        *before* the welcome message.
        """
        lang = context.language

        # --- Gap 1: Consent collection (once per call) ---
        consent_prefix = ""
        if not context.metadata.get("consent_asked"):
            consent_text = _CONSENT_ASK.get(lang, _CONSENT_ASK["hi-IN"])
            consent_prefix = f"{consent_text} "
            context.metadata["consent_asked"] = True
            # Record the consent request in the tracker (actual grant/deny
            # will be recorded when the user responds in the next turn)
            if self._consent_tracker is not None:
                self._consent_tracker.record_consent(
                    call_id=context.call_id,
                    consent_type="recording",
                    granted=True,  # implicit consent assumed if user continues
                )

        if stt_confidence >= 0.7:
            # Language detected with confidence -- proceed
            context.phase = ConversationPhase.OPEN_ELICITATION
            welcome = _WELCOME.get(lang, _WELCOME["hi-IN"])
            disclaimer = _DISCLAIMER.get(lang, _DISCLAIMER["hi-IN"])
            return AgentResponse(
                text=f"{consent_prefix}{welcome} {disclaimer}",
                phase_transition=ConversationPhase.OPEN_ELICITATION,
            )
        else:
            # Low confidence -- ask for language confirmation
            confirm = _LANGUAGE_CONFIRM.get(lang, _LANGUAGE_CONFIRM["hi-IN"])
            return AgentResponse(text=f"{consent_prefix}{confirm}")

    async def _handle_open_elicitation(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 2: Listen to free-form statement, then transition to intake."""
        # The user's first real statement gives us initial context
        # Transition directly to intake — the intake agent will start questions
        context.phase = ConversationPhase.INTAKE
        context.intake_question_index = 0

        # Pass to intake agent for the first question
        return await self._intake.process(context, user_input)

    async def _handle_intake(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 3: Structured intake (5 questions)."""
        response = await self._intake.process(context, user_input)

        # Update profile from intake response
        if response.updated_profile:
            context.user_profile = response.updated_profile

        # Check if intake is complete
        if context.user_profile.required_fields_complete or context.intake_question_index >= 5:
            return await self._transition_to_processing(context)

        # --- Gap 5: Emotional distress handling ---
        if context.emotional_distress_detected:
            logger.info(
                "Emotional distress detected, fast-tracking to processing",
                extra={"call_id": context.call_id},
            )
            # Prepend empathy message and set TTS metadata for slower speech
            lang = context.language
            empathy = _EMPATHY.get(lang, _EMPATHY["hi-IN"])
            processing_response = await self._transition_to_processing(context)
            processing_response.text = f"{empathy} {processing_response.text}"
            processing_response.metadata["tts_speech_rate_factor"] = 0.85
            processing_response.metadata["emotional_distress_mode"] = True
            return processing_response

        return response

    async def _transition_to_processing(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Transition from intake to processing phase."""
        context.phase = ConversationPhase.PROCESSING

        # Send filler while agents process
        lang = context.language
        filler = _PROCESSING_FILLER.get(lang, _PROCESSING_FILLER["hi-IN"])

        # Run eligibility + reviewer in parallel
        processing_response = await self._run_eligibility_and_review(context)

        # Prepend filler context
        processing_response.text = f"{filler}\n\n{processing_response.text}"
        return processing_response

    async def _handle_processing(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Phase 4: Run Eligibility + Reviewer in parallel."""
        return await self._run_eligibility_and_review(context)

    async def _run_eligibility_and_review(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        """Core parallel execution: Eligibility + Reviewer → Convergence → Guidance."""
        eligibility_result = None
        reviewer_result = None

        try:
            eligibility_task = asyncio.create_task(
                self._eligibility.process(context, ""),
                name="eligibility",
            )
            reviewer_task = asyncio.create_task(
                self._reviewer.process(context, ""),
                name="reviewer",
            )

            done, pending = await asyncio.wait(
                [eligibility_task, reviewer_task],
                timeout=self._agent_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel timed-out tasks
            for task in pending:
                task.cancel()
                logger.warning(
                    "Agent timed out",
                    extra={
                        "agent": task.get_name(),
                        "call_id": context.call_id,
                    },
                )

            # Collect results
            if eligibility_task in done and not eligibility_task.cancelled():
                elig_response = eligibility_task.result()
                eligibility_result = elig_response.eligibility_result
            if reviewer_task in done and not reviewer_task.cancelled():
                rev_response = reviewer_task.result()
                reviewer_result = rev_response.reviewer_result

        except Exception as e:
            logger.error(
                "Eligibility/reviewer execution failed",
                extra={"error": str(e), "call_id": context.call_id},
            )

        # Store results
        context.eligibility_result = eligibility_result
        context.reviewer_result = reviewer_result

        # Run convergence check
        if eligibility_result and reviewer_result:
            context.convergence_result = self._convergence.check(
                eligibility_result,
                reviewer_result,
                context,
            )
        elif eligibility_result:
            # Reviewer failed/timed out — use eligibility result with caveats
            context.convergence_result = ConvergenceResult(
                agreed_eligible=[
                    m
                    for m in eligibility_result.matches
                    if m.verdict == EligibilityVerdict.ELIGIBLE
                ],
                agreed_ineligible=[
                    m.scheme_id
                    for m in eligibility_result.matches
                    if m.verdict == EligibilityVerdict.INELIGIBLE
                ],
                disagreements=[],
                conservative_eligible=[],
            )
            context.metadata["reviewer_unavailable"] = True
        else:
            # Both failed — return error
            lang = context.language
            return AgentResponse(
                text=_PROCESSING_FILLER.get(lang, _PROCESSING_FILLER["hi-IN"]),
                error="eligibility_processing_failed",
            )

        # Transition to results and generate guidance
        context.phase = ConversationPhase.RESULTS
        return await self._guidance.process(context, "")

    async def _handle_results(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 5: Results delivery -- one scheme at a time.

        Per PRD 3.2/3.4: deliver schemes individually with a "want to hear
        the next one?" prompt between each.  Tracks progress via
        ``context.metadata["scheme_delivery_index"]``.
        """
        # Initialise scheme delivery index on first entry
        if "scheme_delivery_index" not in context.metadata:
            context.metadata["scheme_delivery_index"] = 0

        eligible = context.convergence_result.all_eligible if context.convergence_result else []
        current_idx: int = context.metadata["scheme_delivery_index"]

        # Check user intent
        lower = user_input.lower()
        negative = any(w in lower for w in ["nahi", "no", "bas", "enough", "nah"])

        if negative:
            # User doesn't want more -- move to guidance
            context.phase = ConversationPhase.GUIDANCE
            return await self._guidance.process(context, user_input)

        # First call or affirmative -- deliver the next scheme
        if current_idx < len(eligible):
            context.metadata["scheme_delivery_index"] = current_idx + 1

            # Ask the guidance agent to format this single scheme
            scheme_response = await self._guidance.process(
                context,
                f"__deliver_scheme_index:{current_idx}",
            )

            # If more schemes remain, append the "want to hear more?" prompt
            if current_idx + 1 < len(eligible):
                lang = context.language
                more_prompt = _MORE_SCHEMES.get(lang, _MORE_SCHEMES["hi-IN"])
                scheme_response.text = f"{scheme_response.text}\n\n{more_prompt}"
            else:
                # All schemes delivered -- auto-transition to guidance
                context.phase = ConversationPhase.GUIDANCE
                scheme_response.phase_transition = ConversationPhase.GUIDANCE

            return scheme_response

        # All schemes already delivered -- transition to guidance
        context.phase = ConversationPhase.GUIDANCE
        return await self._guidance.process(context, user_input)

    async def _handle_guidance(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 6: Action guidance — documents, CSC directions."""
        response = await self._guidance.process(context, user_input)

        # Check if user has more questions
        lower = user_input.lower()
        more_questions = any(
            w in lower
            for w in [
                "aur",
                "question",
                "sawaal",
                "kuch",
                "batao",
                "more",
            ]
        )

        if not more_questions:
            context.phase = ConversationPhase.CLOSURE
            response.phase_transition = ConversationPhase.CLOSURE

        return response

    async def _handle_closure(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Phase 7: Closure & follow-up."""
        lower = user_input.lower()

        # Check if user wants to restart (loop back to Phase 2)
        restart = any(w in lower for w in ["phir", "naya", "restart", "shuru", "again"])
        if restart:
            context.phase = ConversationPhase.OPEN_ELICITATION
            lang = context.language
            text = {
                "hi-IN": "Zarur! Chaliye phir se shuru karte hain. Aapko kya jaanna hai?",
                "ta-IN": "Sari! Pudidhaaga thuvanguvoam. Ungalukku enna theriya vendum?",
                "bn-IN": "Obosshoi! Notun kore shuru kori. Apnar ki jante hobe?",
            }
            return AgentResponse(
                text=text.get(lang, text["hi-IN"]),
                phase_transition=ConversationPhase.OPEN_ELICITATION,
            )

        # Default closure
        lang = context.language
        return AgentResponse(text=_CLOSURE.get(lang, _CLOSURE["hi-IN"]))

    async def _llm_fallback(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """LLM-based routing for input that doesn't fit the state machine."""
        try:
            # Build recent context
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

            category = result.get("category", "UNKNOWN")
            brief_answer = result.get("brief_answer", "")

            match category:
                case "ON_TOPIC":
                    # Re-route to current phase handler (without the fallback)
                    return AgentResponse(text=brief_answer or user_input)
                case "QUESTION":
                    return AgentResponse(text=brief_answer)
                case "REPEAT":
                    # --- Gap 3: Repeat escalation ---
                    repeat_count: int = context.metadata.get("repeat_count", 0) + 1
                    context.metadata["repeat_count"] = repeat_count

                    last = next(
                        (t for t in reversed(context.transcript) if t.role == "assistant"),
                        None,
                    )
                    fallback_text = "Maaf kijiye, kya aap phir se bata sakte hain?"
                    last_text = last.text if last else fallback_text

                    if repeat_count >= 3:
                        # Third+ request: offer SMS instead
                        lang = context.language
                        sms_offer = _OFFER_SMS.get(lang, _OFFER_SMS["hi-IN"])
                        return AgentResponse(
                            text=sms_offer,
                            metadata={"repeat_escalation": "sms_offered"},
                        )
                    elif repeat_count == 2:
                        # Second request: rephrase with simpler words via LLM
                        try:
                            simplified = await self._client.chat_json(
                                self._fallback_model,
                                [
                                    {
                                        "role": "system",
                                        "content": (
                                            "Rephrase the following in very simple "
                                            f"{context.language} words. Use short "
                                            "sentences. Return JSON: "
                                            '{"rephrased": "..."}'
                                        ),
                                    },
                                    {"role": "user", "content": last_text},
                                ],
                            )
                            rephrased = simplified.get("rephrased", last_text)
                        except Exception:
                            rephrased = last_text
                        return AgentResponse(
                            text=rephrased,
                            metadata={
                                "repeat_escalation": "simplified",
                                "tts_speech_rate_factor": 0.80,
                            },
                        )
                    else:
                        # First request: rephrase (don't repeat verbatim)
                        try:
                            rephrased_resp = await self._client.chat_json(
                                self._fallback_model,
                                [
                                    {
                                        "role": "system",
                                        "content": (
                                            "Rephrase the following in "
                                            f"{context.language}. Keep the same "
                                            "meaning but use different words. "
                                            'Return JSON: {"rephrased": "..."}'
                                        ),
                                    },
                                    {"role": "user", "content": last_text},
                                ],
                            )
                            text = rephrased_resp.get("rephrased", last_text)
                        except Exception:
                            text = last_text
                        return AgentResponse(
                            text=text,
                            metadata={
                                "repeat_escalation": "rephrased",
                                "tts_speech_rate_factor": 0.80,
                            },
                        )
                case "RESTART":
                    context.phase = ConversationPhase.OPEN_ELICITATION
                    return AgentResponse(
                        text="Chaliye phir se shuru karte hain.",
                        phase_transition=ConversationPhase.OPEN_ELICITATION,
                    )
                case "END":
                    context.phase = ConversationPhase.CLOSURE
                    lang = context.language
                    return AgentResponse(
                        text=_CLOSURE.get(lang, _CLOSURE["hi-IN"]),
                        phase_transition=ConversationPhase.CLOSURE,
                    )
                case _:
                    return AgentResponse(
                        text="Maaf kijiye, kya aap phir se bata sakte hain?",
                    )

        except Exception as e:
            logger.error("LLM fallback failed", extra={"error": str(e)})
            return AgentResponse(
                text="Maaf kijiye, thodi dikkat aa rahi hai.",
                error=str(e),
            )
