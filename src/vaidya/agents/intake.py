"""Intake Agent: progressive 5-question profile elicitation over voice."""

from __future__ import annotations

import logging
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
    UserProfile,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intake question bank (ordered by sensitivity: easy first, income third)
# ---------------------------------------------------------------------------

_QUESTIONS: dict[int, dict[str, str]] = {
    1: {
        "hi-IN": "Sabse pehle, aap kahan rehte hain? Kaunsa state ya sheher?",
        "ta-IN": "Mudhalil, neengal engae vasikireerkal? Enna maanilam allathu nagaram?",
        "bn-IN": "Prothome, apni kothai thaken? Kon state ba shohor?",
        "en-IN": "First of all, where do you live? Which state or city?",
        "field": "state_district",
        "fallback": {
            "hi-IN": "Village ka naam samajh aaya. Kaunsa district ya state hai?",
        },
    },
    2: {
        "hi-IN": "Aapke ghar mein kitne log hain? Aap, bachche, bade — sab milakar?",
        "ta-IN": "Ungal veetil ethanai per irukkiraargal? Neengal, kuzhanthaigal, periyavargal — ellarum serthu?",
        "bn-IN": "Apnar barite kotojon achen? Apni, bacchara,boro ra — sob miliye?",
        "en-IN": "How many people in your household? You, children, elders — everyone?",
        "field": "family_composition",
        "fallback": {
            "hi-IN": "Aap, patni/pati, bachche — kitne sab milakar?",
        },
    },
    3: {
        "hi-IN": (
            "Aapke ghar ka kharcha kaise chalta hai? Naukri, daily mazdoori, ya apna kaam? "
            "Yeh sirf yojana dhundne ke liye hai. Kisi aur ko nahi bataya jaayega."
        ),
        "ta-IN": (
            "Ungal veettu selavu eppadi nadakkiRadhu? Velai, dhina kooli, allathu sondha thozhil? "
            "Idhu thittangalai kaNdupidikka mattum. Veeru yaaridalum sollappadaadhu."
        ),
        "bn-IN": (
            "Apnar barir khoroch ki kore chole? Chakri, doinik mazdoori, na nijer kaaj? "
            "Eta shudhu yojana khunje ber korar jonno. Aar kaauke bolaa hobe na."
        ),
        "en-IN": (
            "How does your household manage expenses? Job, daily wage, or own business? "
            "This is only for finding schemes. It won't be shared with anyone."
        ),
        "field": "income_livelihood",
    },
    4: {
        "hi-IN": "Kya aapke ghar mein kisi ke paas health insurance hai? Company ka ho, ya koi sarkari card?",
        "ta-IN": "Ungal veetil yaaridalaavathu sugadhara kaappeettu irukkiradhaa? Company-yudaiyatho, allathu arasaanga card-o?",
        "bn-IN": "Apnar barite karo ki health insurance ache? Company-r hok, ba kono sorkari card?",
        "en-IN": "Does anyone in your household have health insurance? From company, or any government card?",
        "field": "existing_coverage",
        "fallback": {
            "hi-IN": "Kya company salary se koi paisa kaat-ti hai health ke liye?",
        },
    },
    5: {
        "hi-IN": "Kya koi khaas ilaaj ya bimari ke liye madad chahiye? Ya bas jaanna chahte hain ki kya milta hai?",
        "ta-IN": "Edhaavathu kurippitta sikichai allathu noikku uthavi veNumaa? Allathu enna kidaikkum endru therindhu kolla virumbugireeRkalaa?",
        "bn-IN": "Kono bishesh chikitsa ba roger jonno sahajjo chai? Na shudhu jaante chan ki ki paoa jaay?",
        "en-IN": "Need help for specific treatment? Or just want to know what's available?",
        "field": "health_need",
    },
}

# Mapping from LLM-extracted income descriptions to IncomeCategory
_INCOME_MAP: dict[str, IncomeCategory] = {
    "below_1l": IncomeCategory.BELOW_1L,
    "1l_to_2.5l": IncomeCategory.L1_TO_2_5L,
    "2.5l_to_5l": IncomeCategory.L2_5_TO_5L,
    "above_5l": IncomeCategory.ABOVE_5L,
}

_OCCUPATION_MAP: dict[str, OccupationType] = {
    "daily_wage": OccupationType.DAILY_WAGE,
    "salaried_govt": OccupationType.SALARIED_GOVT,
    "salaried_pvt": OccupationType.SALARIED_PVT,
    "self_employed": OccupationType.SELF_EMPLOYED,
    "farmer": OccupationType.FARMER,
}

_COVERAGE_MAP: dict[str, CoverageType] = {
    "none": CoverageType.NONE,
    "employer": CoverageType.EMPLOYER,
    "govt_scheme": CoverageType.GOVT_SCHEME,
    "private": CoverageType.PRIVATE,
}

# Maximum questions before we move on even with gaps
_MAX_QUESTIONS = 5

# Map profile field names back to the question that collects them,
# used when the user wants to correct a specific part during confirmation.
_FIELD_TO_QUESTION: dict[str, int] = {
    "state": 1,
    "district": 1,
    "location": 1,
    "family_size": 2,
    "family": 2,
    "income": 3,
    "income_bracket": 3,
    "occupation": 3,
    "occupation_type": 3,
    "coverage": 4,
    "existing_coverage": 4,
    "insurance": 4,
    "health_need": 5,
}

# Prompt to ask which part to correct, per language
_CORRECTION_PROMPTS: dict[str, str] = {
    "hi-IN": "Kaunsi baat galat hai? Jagah, ghar ke log, kaam, insurance, ya health?",
    "ta-IN": "Endha thagaval thavaru? Idam, kudumbam, velai, kaappeettu, allathu sugaadharam?",
    "bn-IN": "Kon tothyo bhul? Jayga, poribar, kaaj, insurance, na swasthyo?",
    "en-IN": "Which part is wrong? Location, family, work, insurance, or health?",
}


class IntakeAgent(BaseAgent):
    """Asks 5 progressive questions to build the user profile.

    Question order is deliberate: easy rapport-builder first (location),
    sensitive income question third (after trust is built), and the
    optional health-need question last (shows the system cares).
    """

    def __init__(self, client: SarvamClient, model: str) -> None:
        super().__init__(client=client, model=model, agent_name="intake")

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Process one turn of the intake conversation.

        Returns an AgentResponse containing the next spoken question
        and any profile updates extracted from the user's answer.
        """
        try:
            return await self._process_turn(context, user_input)
        except Exception as exc:
            logger.error(
                "Intake processing failed",
                extra={"error": str(exc), "call_id": context.call_id},
                exc_info=True,
            )
            return self._fallback_response(context.language)

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    async def _process_turn(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Core turn logic: extract fields, update profile, ask next question.

        After all intake questions are answered, a single confirmation pass
        summarises what the user said and asks "Sahi hai?" before proceeding.
        """
        q_index = context.intake_question_index
        profile = context.user_profile.model_copy(deep=True)
        language = context.language

        # ------------------------------------------------------------------
        # Handle confirmation response (user is answering "Sahi hai?")
        # ------------------------------------------------------------------
        if context.metadata.get("confirmation_pending"):
            return await self._handle_confirmation_response(context, profile, user_input, language)

        # ------------------------------------------------------------------
        # Normal intake flow
        # ------------------------------------------------------------------

        # Determine which question we are answering (0 = initial free-form)
        if q_index == 0 and user_input.strip():
            # First turn: user gave a free-form statement. Extract whatever
            # we can, then ask Q1 to fill the gaps.
            extracted = await self._extract_freeform(user_input, language)
            profile = self._apply_extracted(profile, extracted)
            context.intake_question_index = 1
            next_q = self._get_question_text(1, language)
        elif 1 <= q_index <= _MAX_QUESTIONS:
            # Answering question q_index
            extracted = await self._extract_answer(
                user_input,
                q_index,
                profile,
                language,
            )

            # Check for emotional distress
            if extracted.get("distress_detected"):
                context.emotional_distress_detected = True
                profile = self._apply_extracted(profile, extracted)
                # Distress fast-track: skip remaining questions, go to confirmation
                context.intake_question_index = _MAX_QUESTIONS + 1
                empathy = extracted.get("spoken_text", "")
                confirmation = self._build_confirmation(profile, language)
                context.metadata["confirmation_pending"] = True
                spoken = f"{empathy} {confirmation}".strip() if empathy else confirmation
                return AgentResponse(
                    text=spoken,
                    updated_profile=profile,
                    metadata={
                        "intake_distress_detected": True,
                        "confirmation_pending": True,
                    },
                )

            profile = self._apply_extracted(profile, extracted)

            # Handle follow-up if the LLM asks for clarification
            if extracted.get("needs_followup") and not extracted.get("question_complete"):
                spoken = extracted.get("spoken_text", "")
                return AgentResponse(
                    text=spoken,
                    updated_profile=profile,
                    metadata={"intake_q": q_index, "followup": True},
                )

            # Move to the next question
            context.intake_question_index = q_index + 1
            next_index = q_index + 1

            # Check if all questions are done — enter confirmation pass
            if next_index > _MAX_QUESTIONS or profile.required_fields_complete:
                context.intake_question_index = _MAX_QUESTIONS + 1
                ack = self._build_acknowledgement(extracted, language)
                confirmation = self._build_confirmation(profile, language)
                context.metadata["confirmation_pending"] = True
                spoken = f"{ack} {confirmation}".strip() if ack else confirmation
                return AgentResponse(
                    text=spoken,
                    updated_profile=profile,
                    metadata={"confirmation_pending": True},
                )

            next_q = self._get_question_text(next_index, language)
        else:
            # Already past max questions — should not arrive here, but be safe
            return AgentResponse(
                text="",
                updated_profile=profile,
                metadata={"intake_complete": True},
            )

        # Build the spoken response: the LLM's acknowledgement + next question
        # The LLM response already contains the next question via the prompt,
        # but we explicitly construct it to guarantee the right question is asked.
        llm_spoken = ""
        if q_index >= 1:
            # We have an extracted answer; the LLM may have produced an ack line
            llm_spoken = self._build_acknowledgement(extracted, language)

        spoken_text = f"{llm_spoken} {next_q}".strip() if llm_spoken else next_q

        return AgentResponse(
            text=spoken_text,
            updated_profile=profile,
            metadata={
                "intake_q": context.intake_question_index,
                "fields_complete": not profile.missing_fields,
            },
        )

    async def _handle_confirmation_response(
        self,
        context: ConversationContext,
        profile: UserProfile,
        user_input: str,
        language: str,
    ) -> AgentResponse:
        """Handle the user's response to the confirmation summary.

        If yes → mark intake complete and proceed to PROCESSING.
        If no  → ask which part to correct and clear confirmation state.
        """
        # Use the LLM to determine if the user confirmed or denied
        confirmation_result = await self._extract_confirmation(user_input, language)

        if confirmation_result.get("confirmed", False):
            # User confirmed — intake is done
            context.metadata.pop("confirmation_pending", None)
            return AgentResponse(
                text=confirmation_result.get("spoken_text", ""),
                updated_profile=profile,
                metadata={"intake_complete": True},
            )

        # User wants to correct something
        context.metadata.pop("confirmation_pending", None)
        correction_field = confirmation_result.get("correction_field")
        spoken = confirmation_result.get("spoken_text", "")

        if correction_field and correction_field in _FIELD_TO_QUESTION:
            # Re-ask the specific question
            re_q_num = _FIELD_TO_QUESTION[correction_field]
            context.intake_question_index = re_q_num
            re_q_text = self._get_question_text(re_q_num, language)
            spoken = f"{spoken} {re_q_text}".strip() if spoken else re_q_text
        elif not spoken:
            # Fallback: ask which part to correct
            spoken = _CORRECTION_PROMPTS.get(
                language,
                _CORRECTION_PROMPTS["hi-IN"],
            )

        return AgentResponse(
            text=spoken,
            updated_profile=profile,
            metadata={"intake_correction": True},
        )

    async def _extract_confirmation(
        self,
        user_input: str,
        language: str,
    ) -> dict[str, Any]:
        """Determine whether the user confirmed or denied the summary.

        Returns dict with:
          confirmed: bool
          spoken_text: str (acknowledgement or follow-up)
          correction_field: str | None (which field to re-ask if denied)
        """
        system = prompts.render(
            "intake_system",
            question_number="confirmation",
            current_question="Confirmation of intake summary — user should say yes or no.",
            profile_summary="(confirming previously collected data)",
            language=language,
        )
        try:
            result = await self._call_llm_json(
                system,
                user_input,
                reasoning_effort="low",
                max_tokens=1024,
            )
            # Ensure the confirmed key exists
            if "confirmed" not in result:
                # Heuristic fallback: check for affirmative words
                lower = user_input.lower().strip()
                affirm = {"haan", "ha", "ji", "sahi", "theek", "yes", "correct", "aama", "hya"}
                result["confirmed"] = any(w in lower for w in affirm)
            return result
        except Exception as exc:
            logger.warning("Confirmation extraction failed", extra={"error": str(exc)})
            # Conservative: treat as confirmed to avoid infinite loops
            return {"confirmed": True, "spoken_text": ""}

    async def _extract_freeform(
        self,
        user_input: str,
        language: str,
    ) -> dict[str, Any]:
        """Extract any profile fields from the initial free-form statement."""
        system = prompts.render(
            "intake_system",
            question_number="0",
            current_question=(
                "(Initial free-form statement — extract whatever information the user volunteers.)"
            ),
            profile_summary="No information yet.",
            language=language,
        )
        try:
            return await self._call_llm_json(
                system,
                user_input,
                reasoning_effort="low",
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Free-form extraction failed", extra={"error": str(exc)})
            return {"extracted_fields": {}, "field_confidence": {}}

    async def _extract_answer(
        self,
        user_input: str,
        question_number: int,
        profile: UserProfile,
        language: str,
    ) -> dict[str, Any]:
        """Extract structured fields from the user's answer to a specific question."""
        q_def = _QUESTIONS.get(question_number, {})
        current_question = q_def.get(language, q_def.get("hi-IN", ""))

        profile_summary = self._summarize_profile(profile)

        system = prompts.render(
            "intake_system",
            question_number=str(question_number),
            current_question=current_question,
            profile_summary=profile_summary,
            language=language,
        )
        try:
            return await self._call_llm_json(
                system,
                user_input,
                reasoning_effort="low",
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning(
                "Answer extraction failed",
                extra={"error": str(exc), "question": question_number},
            )
            return {
                "extracted_fields": {},
                "field_confidence": {},
                "question_complete": True,
                "spoken_text": "",
            }

    # ------------------------------------------------------------------
    # Profile mutation helpers
    # ------------------------------------------------------------------

    def _apply_extracted(
        self,
        profile: UserProfile,
        extracted: dict[str, Any],
    ) -> UserProfile:
        """Apply LLM-extracted fields to the profile with confidence tracking."""
        fields = extracted.get("extracted_fields", {})
        confidence = extracted.get("field_confidence", {})

        if not fields:
            return profile

        # State / district
        if fields.get("state"):
            profile.state = fields["state"]
            profile.confidence_flags["state"] = confidence.get("state", 0.5)
        if fields.get("district"):
            profile.district = fields["district"]
            profile.confidence_flags["district"] = confidence.get("district", 0.5)

        # Family size
        if fields.get("family_size") is not None:
            try:
                profile.family_size = int(fields["family_size"])
                profile.confidence_flags["family_size"] = confidence.get("family_size", 0.5)
            except (ValueError, TypeError):
                pass

        # Age
        if fields.get("age") is not None:
            try:
                profile.age = int(fields["age"])
                profile.confidence_flags["age"] = confidence.get("age", 0.5)
            except (ValueError, TypeError):
                pass

        # Income bracket
        raw_income = str(fields.get("income_bracket", "")).lower().strip()
        if raw_income and raw_income in _INCOME_MAP:
            profile.income_bracket = _INCOME_MAP[raw_income]
            profile.confidence_flags["income_bracket"] = confidence.get("income_bracket", 0.5)

        # Occupation type
        raw_occ = str(fields.get("occupation_type", "")).lower().strip()
        if raw_occ and raw_occ in _OCCUPATION_MAP:
            profile.occupation_type = _OCCUPATION_MAP[raw_occ]
            profile.confidence_flags["occupation_type"] = confidence.get("occupation_type", 0.5)

        # Existing coverage
        raw_cov = str(fields.get("existing_coverage", "")).lower().strip()
        if raw_cov and raw_cov in _COVERAGE_MAP:
            profile.existing_coverage = _COVERAGE_MAP[raw_cov]
            profile.confidence_flags["existing_coverage"] = confidence.get(
                "existing_coverage", 0.5
            )

        # Health need (free-text)
        if fields.get("health_need"):
            profile.health_need = fields["health_need"]
            profile.confidence_flags["health_need"] = confidence.get("health_need", 0.5)

        # BPL / ration card (booleans)
        if fields.get("bpl_card") is not None:
            profile.bpl_card = _to_bool(fields["bpl_card"])
            profile.confidence_flags["bpl_card"] = confidence.get("bpl_card", 0.5)
        if fields.get("ration_card") is not None:
            profile.ration_card = _to_bool(fields["ration_card"])
            profile.confidence_flags["ration_card"] = confidence.get("ration_card", 0.5)

        # SECC category
        if fields.get("secc_category"):
            profile.secc_category = fields["secc_category"]
            profile.confidence_flags["secc_category"] = confidence.get("secc_category", 0.5)

        return profile

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_question_text(question_number: int, language: str) -> str:
        """Retrieve the localized question text for a given question number."""
        q_def = _QUESTIONS.get(question_number, {})
        return q_def.get(language, q_def.get("hi-IN", ""))

    @staticmethod
    def _summarize_profile(profile: UserProfile) -> str:
        """Human-readable one-liner of what we know so far."""
        parts: list[str] = []
        if profile.state:
            loc = profile.state
            if profile.district:
                loc += f" ({profile.district})"
            parts.append(f"Location: {loc}")
        if profile.family_size is not None:
            parts.append(f"Family size: {profile.family_size}")
        if profile.income_bracket != IncomeCategory.UNKNOWN:
            parts.append(f"Income: {profile.income_bracket.value}")
        if profile.occupation_type != OccupationType.UNKNOWN:
            parts.append(f"Occupation: {profile.occupation_type.value}")
        if profile.existing_coverage != CoverageType.UNKNOWN:
            parts.append(f"Coverage: {profile.existing_coverage.value}")
        if profile.health_need:
            parts.append(f"Health need: {profile.health_need}")
        return " | ".join(parts) if parts else "No information yet."

    @staticmethod
    def _build_confirmation(profile: UserProfile, language: str) -> str:
        """Build a single confirmation summary from the collected profile.

        PRD Section 3.2: after ALL intake questions, present a summary
        and ask "Sahi hai?" before proceeding to PROCESSING.

        Template (hi-IN):
          "Aapne bataya ki aap [state] mein rehte hain, ghar mein [N] log hain,
           aur aap [occupation] karte hain. Sahi hai?"
        """
        # -- Hindi (default) ------------------------------------------------
        if language not in ("ta-IN", "bn-IN"):
            parts: list[str] = []
            if profile.state:
                parts.append(f"aap {profile.state} mein rehte hain")
            if profile.family_size is not None:
                parts.append(f"ghar mein {profile.family_size} log hain")
            if profile.occupation_type != OccupationType.UNKNOWN:
                occ_labels_hi = {
                    OccupationType.DAILY_WAGE: "daily mazdoori karte hain",
                    OccupationType.SALARIED_GOVT: "sarkari naukri karte hain",
                    OccupationType.SALARIED_PVT: "private naukri karte hain",
                    OccupationType.SELF_EMPLOYED: "apna kaam karte hain",
                    OccupationType.FARMER: "kheti karte hain",
                }
                parts.append(occ_labels_hi.get(profile.occupation_type, "kaam karte hain"))
            if not parts:
                return "Sahi hai?"
            joined = ", ".join(parts[:-1])
            if len(parts) > 1:
                joined += f", aur aap {parts[-1]}" if len(parts) > 2 else f" aur aap {parts[-1]}"
                return f"Aapne bataya ki {joined}. Sahi hai?"
            return f"Aapne bataya ki aap {parts[0]}. Sahi hai?"

        # -- Tamil ----------------------------------------------------------
        if language == "ta-IN":
            parts_ta: list[str] = []
            if profile.state:
                parts_ta.append(f"neengal {profile.state}-il vasikireerkal")
            if profile.family_size is not None:
                parts_ta.append(f"veetil {profile.family_size} per irukkiraargal")
            if profile.occupation_type != OccupationType.UNKNOWN:
                occ_labels_ta = {
                    OccupationType.DAILY_WAGE: "dhina kooli velai seygiReerkal",
                    OccupationType.SALARIED_GOVT: "arasanga velai seygiReerkal",
                    OccupationType.SALARIED_PVT: "private velai seygiReerkal",
                    OccupationType.SELF_EMPLOYED: "sondha thozhil seygiReerkal",
                    OccupationType.FARMER: "vivasaayam seygiReerkal",
                }
                parts_ta.append(occ_labels_ta.get(profile.occupation_type, "velai seygiReerkal"))
            if not parts_ta:
                return "Sari-yaa?"
            summary = ", ".join(parts_ta)
            return f"Neengal sonnadhu: {summary}. Sari-yaa?"

        # -- Bengali --------------------------------------------------------
        parts_bn: list[str] = []
        if profile.state:
            parts_bn.append(f"apni {profile.state}-e thaken")
        if profile.family_size is not None:
            parts_bn.append(f"barite {profile.family_size} jon achen")
        if profile.occupation_type != OccupationType.UNKNOWN:
            occ_labels_bn = {
                OccupationType.DAILY_WAGE: "doinik mazdoori koren",
                OccupationType.SALARIED_GOVT: "sorkari chakri koren",
                OccupationType.SALARIED_PVT: "private chakri koren",
                OccupationType.SELF_EMPLOYED: "nijer kaaj koren",
                OccupationType.FARMER: "chash koren",
            }
            parts_bn.append(occ_labels_bn.get(profile.occupation_type, "kaaj koren"))
        if not parts_bn:
            return "Thik ache?"
        summary = ", ".join(parts_bn)
        return f"Apni bollen je {summary}. Thik ache?"

    @staticmethod
    def _build_acknowledgement(extracted: dict[str, Any], language: str) -> str:
        """Build a short acknowledgement from the LLM extraction.

        If the LLM provided a spoken_text field, use that as the ack.
        Otherwise return empty string (the next question will stand alone).
        """
        spoken = extracted.get("spoken_text", "")
        if spoken:
            return spoken
        return ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_bool(value: Any) -> bool | None:
    """Coerce various truthy representations to bool."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    if s in ("true", "yes", "haan", "ha", "1"):
        return True
    if s in ("false", "no", "nahi", "nah", "0"):
        return False
    return None
