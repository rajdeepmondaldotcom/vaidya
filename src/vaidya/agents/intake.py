"""Intake Agent: progressive 5-question profile elicitation over voice."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.constants import MAX_INTAKE_QUESTIONS
from vaidya.i18n import get_msg, get_msg_template
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
from vaidya.utils.states import state_code_to_name, state_name_to_code

logger = logging.getLogger(__name__)

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

_FIELD_EXAMPLES: dict[int | str, str] = {
    0: '{"state": "value_or_null", "district": "value_or_null", "family_size": "integer_or_null", "occupation_type": "value_or_null", "income_bracket": "value_or_null", "existing_coverage": "value_or_null", "health_need": "value_or_null"}',
    1: '{"state": "value_or_null", "district": "value_or_null"}',
    2: '{"family_size": "integer_or_null"}',
    3: '{"occupation_type": "daily_wage|salaried_govt|salaried_pvt|self_employed|farmer|null", "income_bracket": "below_1l|1l_to_2.5l|2.5l_to_5l|above_5l|null"}',
    4: '{"existing_coverage": "none|employer|govt_scheme|private|null"}',
    5: '{"health_need": "description_or_null"}',
    "confirmation": '{"confirmed": "true_or_false", "correction_field": "field_name_or_null"}',
}

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


_TRUE_WORDS = frozenset({"true", "yes", "haan", "ha", "1"})
_FALSE_WORDS = frozenset({"false", "no", "nahi", "nah", "0"})

_CONFIRMATION_WORDS = frozenset(
    {"haan", "ha", "ji", "sahi", "theek", "yes", "correct", "aama", "hya"}
)

_MAX_REPAIRS_PER_QUESTION = 2

# Fast path: a short, unambiguous answer the deterministic heuristics fully cover
# skips the LLM extraction entirely (an instant turn). Long or ambiguous answers
# fall back to the LLM. A mis-read is caught by the confirmation step.
_FAST_PATH_MAX_WORDS = 12

_QUESTION_REQUIRED_FIELDS: dict[int, tuple[str, ...]] = {
    1: ("state",),
    2: ("family_size",),
    3: ("occupation_type", "income_bracket"),
    4: ("existing_coverage",),
    5: ("health_need",),
}


def _to_bool(value: Any) -> bool | None:
    """Coerce various truthy representations to bool."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    if s in _TRUE_WORDS:
        return True
    if s in _FALSE_WORDS:
        return False
    return None


def _coerce_int(value: Any) -> int | None:
    """Attempt int coercion, returning None on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _coerce_enum(value: Any, mapping: dict[str, Any]) -> Any | None:
    """Look up a string value in a mapping, returning None if not found."""
    raw = str(value).lower().strip()
    return mapping.get(raw) if raw else None


_FIELD_MAPPINGS: list[tuple[str, str, Callable[[Any], Any]]] = [
    ("state", "state", lambda v: v),
    ("district", "district", lambda v: v),
    ("family_size", "family_size", _coerce_int),
    ("age", "age", _coerce_int),
    ("income_bracket", "income_bracket", lambda v: _coerce_enum(v, _INCOME_MAP)),
    ("occupation_type", "occupation_type", lambda v: _coerce_enum(v, _OCCUPATION_MAP)),
    ("existing_coverage", "existing_coverage", lambda v: _coerce_enum(v, _COVERAGE_MAP)),
    ("health_need", "health_need", lambda v: v),
    ("bpl_card", "bpl_card", _to_bool),
    ("ration_card", "ration_card", _to_bool),
    ("secc_category", "secc_category", lambda v: v),
]

_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "ek": 1,
    "do": 2,
    "two": 2,
    "teen": 3,
    "three": 3,
    "char": 4,
    "chaar": 4,
    "four": 4,
    "paanch": 5,
    "panch": 5,
    "five": 5,
    "six": 6,
    "che": 6,
    "saat": 7,
    "seven": 7,
    "aath": 8,
    "eight": 8,
    "nau": 9,
    "nine": 9,
    "dus": 10,
    "ten": 10,
}

_STATE_WINDOW_WORDS = 4


def _heuristic_confirmation(user_input: str) -> bool:
    """Word-boundary matching fallback for confirmation detection."""
    words = set(re.findall(r"\b\w+\b", user_input.lower().strip()))
    return bool(words & _CONFIRMATION_WORDS)


def _collect_confirmation_parts(profile: UserProfile, language: str) -> list[str]:
    """Gather localized confirmation fragments from the profile."""
    parts: list[str] = []
    if profile.state:
        parts.append(get_msg_template("intake", "confirm_state", language, state=profile.state))
    if profile.family_size is not None:
        parts.append(
            get_msg_template("intake", "confirm_family", language, count=profile.family_size)
        )
    if profile.occupation_type != OccupationType.UNKNOWN:
        parts.append(_format_occupation_label(profile.occupation_type, language))
    if profile.income_bracket != IncomeCategory.UNKNOWN:
        parts.append(
            get_msg_template(
                "intake",
                "confirm_income",
                language,
                income=_format_enum_value(profile.income_bracket.value),
            )
        )
    if profile.existing_coverage != CoverageType.UNKNOWN:
        parts.append(
            get_msg_template(
                "intake",
                "confirm_coverage",
                language,
                coverage=_format_enum_value(profile.existing_coverage.value),
            )
        )
    if profile.health_need:
        parts.append(
            get_msg_template(
                "intake",
                "confirm_health_need",
                language,
                need=profile.health_need,
            )
        )
    return parts


def _format_occupation_label(occupation: OccupationType, language: str) -> str:
    """Resolve a localized occupation label, falling back to the default."""
    occ_key = f"occ_{occupation.value}"
    occ_label = get_msg("intake", occ_key, language)
    if occ_label == occ_key:
        return get_msg("intake", "occ_default", language)
    return occ_label


def _join_with_conjunction(parts: list[str], conjunction: str) -> str:
    """Join parts with commas and a final conjunction."""
    joined = ", ".join(parts[:-1])
    separator = ", " if len(parts) > 2 else " "
    return f"{joined}{separator}{conjunction} {parts[-1]}"


def _format_enum_value(value: str) -> str:
    """Make enum values speakable without adding a large label table."""
    return (
        value.replace("_", " ")
        .replace("1l", "1 lakh")
        .replace("2.5l", "2.5 lakh")
        .replace(
            "5l",
            "5 lakh",
        )
    )


class IntakeAgent(BaseAgent):
    """Asks 5 progressive questions to build the user profile.

    Question order is deliberate: easy rapport-builder first (location),
    sensitive income question third (after trust is built), and the
    optional health-need question last (shows the system cares).
    """

    def __init__(self, client: SarvamClient, model: str, reasoning_effort: str = "low") -> None:
        super().__init__(client=client, model=model, agent_name="intake")
        self._reasoning_effort = reasoning_effort
        # LLM-first by default. The deterministic heuristic fast path (skip the
        # LLM for clear answers) is an opt-in scale/cost optimization set from
        # config in app.py — off by default so intake showcases sarvam-30b's
        # native multilingual extraction and never depends on brittle keyword
        # matching for correctness.
        self._fast_path_enabled = False

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Process one turn of the intake conversation.

        Returns an AgentResponse containing the next spoken question
        and any profile updates extracted from the user's answer.

        Intake runs entirely in the user's language: every spoken string is
        either pulled from the localized i18n catalogue (questions,
        confirmation, corrections) via ``get_msg(..., context.language)`` or is
        the LLM's ``spoken_text`` ack, which the prompt requires in the user's
        language. So every response is marked ``already_localized=True`` and the
        ConversationManager skips the (redundant) en-IN translate-out hop -- the
        hop where the placeholder-leak / blank-turn bugs lived.
        """
        response = await self._process_turn(context, user_input)
        response.already_localized = True
        return response

    async def safe_process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Override to flag the error fallback as already-localized too.

        ``BaseAgent.safe_process`` returns ``_fallback_response`` on failure,
        which is a ``get_msg(..., context.language)`` string -- already in the
        user's language. Flag it so the manager never translates it back out.
        """
        response = await super().safe_process(context, user_input)
        response.already_localized = True
        return response

    async def _process_turn(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Route by intake state: confirmation, initial freeform, or Q&A."""
        profile = context.user_profile.model_copy(deep=True)
        language = context.language

        if context.metadata.get("confirmation_pending"):
            return await self._handle_confirmation_response(context, profile, user_input, language)

        q_index = context.intake_question_index

        if q_index == 0 and user_input.strip():
            return await self._handle_initial_freeform(context, profile, user_input, language)
        elif q_index == 0:
            # Empty input at start - ask the first question
            context.intake_question_index = self._next_missing_question(context, profile, 1) or 1
            self._mark_question_asked(context, context.intake_question_index)
            return AgentResponse(
                text=self._get_question_text(context.intake_question_index, language),
                updated_profile=profile,
                metadata={"intake_q": context.intake_question_index},
            )

        if 1 <= q_index <= MAX_INTAKE_QUESTIONS:
            return await self._handle_question_answer(
                context,
                profile,
                user_input,
                q_index,
                language,
            )

        return AgentResponse(
            text="",
            updated_profile=profile,
            metadata={"intake_complete": True},
        )

    async def _handle_initial_freeform(
        self,
        context: ConversationContext,
        profile: UserProfile,
        user_input: str,
        language: str,
    ) -> AgentResponse:
        """Extract from first free-form statement, then ask the next missing field."""
        extracted = await self._extract_freeform(user_input, language)
        extracted = self._merge_heuristic_fields(extracted, user_input, 0)
        profile = self._apply_extracted(profile, extracted)

        next_index = self._next_missing_question(context, profile, 1)
        if next_index is None:
            return self._enter_confirmation(context, profile, extracted, language)

        context.intake_question_index = next_index
        self._mark_question_asked(context, next_index)
        next_q = self._get_question_text(next_index, language)

        return AgentResponse(
            text=next_q,
            updated_profile=profile,
            metadata={
                "intake_q": context.intake_question_index,
                "fields_complete": not profile.missing_fields,
            },
        )

    def _try_fast_extract(
        self,
        user_input: str,
        q_index: int,
        language: str,
    ) -> dict[str, Any] | None:
        """Deterministic fast path: skip the LLM when the heuristics fully and
        unambiguously cover this question's required field(s).

        Returns an ``extracted``-shaped dict (so the normal apply/advance flow is
        unchanged) on a confident match, or ``None`` to fall back to the LLM.
        Only the single-field factual questions (1-4) qualify; the free-form
        initial turn (0) and the health-need question (5) always use the LLM. A
        mis-read here is caught by the end-of-intake confirmation step, which
        recaps the profile and lets the caller correct it.
        """
        if not self._fast_path_enabled:
            return None
        required = _QUESTION_REQUIRED_FIELDS.get(q_index)
        if not required or q_index not in (1, 2, 3, 4):
            return None
        # Long answers tend to carry context/qualifiers the heuristics miss.
        if len(user_input.split()) > _FAST_PATH_MAX_WORDS:
            return None
        # Negation flips meaning on the entity questions (state/occupation), where
        # a naive keyword grab could pick the negated value — let the LLM handle
        # those. (Coverage/family negation is handled by their own heuristics.)
        if q_index in (1, 3) and self._NEGATION_RE.search(user_input.lower()):
            return None
        heuristic = self._heuristic_fields(user_input, q_index)
        if not all(field in heuristic for field in required):
            return None
        confidence = {f: (0.9 if f == "income_bracket" else 0.85) for f in heuristic}
        ack = get_msg("intake", "ack", language)
        return {
            "extracted_fields": heuristic,
            "field_confidence": confidence,
            "question_complete": True,
            "spoken_text": ack if ack != "ack" else "",
        }

    async def _handle_question_answer(
        self,
        context: ConversationContext,
        profile: UserProfile,
        user_input: str,
        q_index: int,
        language: str,
    ) -> AgentResponse:
        """Extract the answer, handle distress/followup, advance or confirm.

        Tries a deterministic fast path first (skips the LLM for short, clear
        answers the heuristics fully cover — an instant turn); falls back to the
        LLM for anything ambiguous.
        """
        extracted = self._try_fast_extract(user_input, q_index, language)
        if extracted is None:
            extracted = await self._extract_answer(user_input, q_index, profile, language)
            if extracted.get("distress_detected"):
                return self._handle_distress_response(context, profile, extracted, language)
            extracted = self._merge_heuristic_fields(extracted, user_input, q_index)

        profile = self._apply_extracted(profile, extracted)

        if self._needs_repair(context, profile, extracted, q_index):
            repair_response = self._repair_or_skip_question(
                context,
                profile,
                extracted,
                q_index,
                language,
            )
            if repair_response is not None:
                return repair_response

        return self._advance_to_next_question(context, profile, extracted, q_index, language)

    def _handle_distress_response(
        self,
        context: ConversationContext,
        profile: UserProfile,
        extracted: dict[str, Any],
        language: str,
    ) -> AgentResponse:
        """Fast-track to confirmation when emotional distress is detected."""
        context.emotional_distress_detected = True
        profile = self._apply_extracted(profile, extracted)
        context.intake_question_index = MAX_INTAKE_QUESTIONS + 1
        empathy = extracted.get("spoken_text", "")
        confirmation = self._build_confirmation(profile, language)
        context.metadata["confirmation_pending"] = True
        spoken = f"{empathy} {confirmation}".strip() if empathy else confirmation
        return AgentResponse(
            text=spoken,
            updated_profile=profile,
            metadata={"intake_distress_detected": True, "confirmation_pending": True},
        )

    def _advance_to_next_question(
        self,
        context: ConversationContext,
        profile: UserProfile,
        extracted: dict[str, Any],
        q_index: int,
        language: str,
    ) -> AgentResponse:
        """Update question index and either enter confirmation or ask the next question."""
        self._mark_question_answered(context, q_index)
        next_index = self._next_missing_question(context, profile, q_index + 1)

        if next_index is None:
            return self._enter_confirmation(context, profile, extracted, language)

        context.intake_question_index = next_index
        self._mark_question_asked(context, next_index)
        next_q = self._get_question_text(next_index, language)
        ack = self._build_acknowledgement(extracted, language)
        spoken_text = f"{ack} {next_q}".strip() if ack else next_q
        return AgentResponse(
            text=spoken_text,
            updated_profile=profile,
            metadata={
                "intake_q": context.intake_question_index,
                "fields_complete": not profile.missing_fields,
            },
        )

    def _enter_confirmation(
        self,
        context: ConversationContext,
        profile: UserProfile,
        extracted: dict[str, Any],
        language: str,
    ) -> AgentResponse:
        """Transition to confirmation pass after all questions are done."""
        context.intake_question_index = MAX_INTAKE_QUESTIONS + 1
        ack = self._build_acknowledgement(extracted, language)
        confirmation = self._build_confirmation(profile, language)
        context.metadata["confirmation_pending"] = True
        spoken = f"{ack} {confirmation}".strip() if ack else confirmation
        return AgentResponse(
            text=spoken,
            updated_profile=profile,
            metadata={"confirmation_pending": True},
        )

    # Explicit "this is WRONG / change it" intent only. A bare "no" must
    # NOT match: intake can desync (an off-script follow-up shifts question
    # alignment) so a question-answer like "No, we don't have insurance" or
    # "No specific illness" can land on the confirmation step — treating that
    # "no" as a correction trapped the caller in a re-ask loop. Require a
    # real correction word so confirmation stays biased to proceed.
    _CORRECTION_RE = re.compile(
        r"\b(wrong|incorrect|mistake|change it|change kar|edit|galat|ghalat|"
        r"bhul|badlo|badal do|theek nahi|sahi nahi|not correct|not right)\b"
        r"|ভুল|ভূল|গলত|ঠিক নয়|ঠিক নেই",
        re.IGNORECASE,
    )

    async def _handle_confirmation_response(
        self,
        context: ConversationContext,
        profile: UserProfile,
        user_input: str,
        language: str,
    ) -> AgentResponse:
        """Route the confirmation response, BIASED toward proceeding.

        A false "not confirmed" traps the caller in a correction loop
        forever; a false "confirmed" merely advances to the results, which
        are already conservative ("you MAY qualify, confirm at CSC"). So we
        only branch to correction on an EXPLICIT correction request and
        otherwise proceed — the flaky LLM confirmation verdict is no longer
        on the critical path.
        """
        context.metadata.pop("confirmation_pending", None)
        text = (user_input or "").strip()

        wants_correction = bool(self._CORRECTION_RE.search(text)) and not _heuristic_confirmation(
            text
        )
        if not wants_correction:
            return self._handle_confirmation_yes(profile, {"spoken_text": ""})

        confirmation_result = await self._extract_confirmation(user_input, language)
        return self._handle_confirmation_no(context, profile, confirmation_result, language)

    @staticmethod
    def _handle_confirmation_yes(
        profile: UserProfile,
        confirmation_result: dict[str, Any],
    ) -> AgentResponse:
        """Mark intake complete after user confirms the summary."""
        return AgentResponse(
            text=confirmation_result.get("spoken_text", ""),
            updated_profile=profile,
            metadata={"intake_complete": True},
        )

    def _handle_confirmation_no(
        self,
        context: ConversationContext,
        profile: UserProfile,
        confirmation_result: dict[str, Any],
        language: str,
    ) -> AgentResponse:
        """Re-ask the specific question or list correctable fields."""
        correction_field = confirmation_result.get("correction_field")
        spoken = confirmation_result.get("spoken_text", "")

        if correction_field and correction_field in _FIELD_TO_QUESTION:
            re_q_num = _FIELD_TO_QUESTION[correction_field]
            context.intake_question_index = re_q_num
            re_q_text = self._get_question_text(re_q_num, language)
            spoken = f"{spoken} {re_q_text}".strip() if spoken else re_q_text
        else:
            correction_prompt = get_msg("intake", "correction", language)
            spoken = f"{spoken} {correction_prompt}".strip() if spoken else correction_prompt

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

        Returns dict with confirmed (bool), spoken_text (str),
        and correction_field (str | None).
        """
        system = prompts.render(
            "intake_system",
            question_number="confirmation",
            current_question="Confirmation of intake summary -- user should say yes or no.",
            profile_summary="(confirming previously collected data)",
            language=language,
            expected_fields_json=_FIELD_EXAMPLES["confirmation"],
        )
        try:
            result = await self._call_llm_json(
                system,
                user_input,
                reasoning_effort=self._reasoning_effort,
                max_tokens=4096,
            )
            if "confirmed" not in result:
                result["confirmed"] = _heuristic_confirmation(user_input)
            return result
        except Exception as exc:
            logger.debug(
                "LLM confirmation parse failed, using heuristic",
                extra={"error": str(exc)},
            )
            return {
                "confirmed": _heuristic_confirmation(user_input),
                "spoken_text": "",
            }

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
                "(Initial free-form statement -- extract whatever information the user volunteers.)"
            ),
            profile_summary="No information yet.",
            language=language,
            expected_fields_json=_FIELD_EXAMPLES[0],
        )
        try:
            return await self._call_llm_json(
                system,
                user_input,
                reasoning_effort=self._reasoning_effort,
                max_tokens=4096,
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
        current_question = get_msg("intake", f"q{question_number}", language)

        profile_summary = self._summarize_profile(profile)

        system = prompts.render(
            "intake_system",
            question_number=str(question_number),
            current_question=current_question,
            profile_summary=profile_summary,
            language=language,
            expected_fields_json=_FIELD_EXAMPLES.get(question_number, _FIELD_EXAMPLES[1]),
        )
        try:
            return await self._call_llm_json(
                system,
                user_input,
                reasoning_effort=self._reasoning_effort,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.warning(
                "Answer extraction failed",
                extra={"error": str(exc), "question": question_number},
                exc_info=True,
            )
            return {
                "extracted_fields": {},
                "field_confidence": {},
                "question_complete": True,
                "spoken_text": "",
            }

    def _apply_extracted(
        self,
        profile: UserProfile,
        extracted: dict[str, Any],
    ) -> UserProfile:
        """Apply LLM-extracted fields to the profile with confidence tracking."""
        # The LLM sometimes returns these as a scalar/list instead of an
        # object (e.g. field_confidence: 0.9). Coerce to dict so the later
        # .get() calls can't crash the whole turn into an error fallback.
        fields = extracted.get("extracted_fields", {})
        confidence = extracted.get("field_confidence", {})
        if not isinstance(fields, dict):
            fields = {}
        if not isinstance(confidence, dict):
            confidence = {}

        if not fields:
            extracted = extracted or {}
            if not any(
                k in extracted
                for k in (
                    "state",
                    "district",
                    "family_size",
                    "age",
                    "income",
                    "occupation",
                    "coverage",
                    "health_need",
                )
            ):
                logger.debug(
                    "LLM extraction returned no recognized fields",
                    extra={"raw_keys": list(extracted.keys())},
                )
            return profile

        for field_name, attr_name, transform_fn in _FIELD_MAPPINGS:
            raw_value = fields.get(field_name)
            if raw_value is None:
                if field_name not in fields:
                    continue
                if field_name not in ("bpl_card", "ration_card"):
                    continue

            transformed = transform_fn(raw_value)
            if transformed is None:
                continue

            setattr(profile, attr_name, transformed)
            profile.confidence_flags[field_name] = confidence.get(field_name, 0.5)

        return profile

    def _merge_heuristic_fields(
        self,
        extracted: dict[str, Any],
        user_input: str,
        question_number: int,
    ) -> dict[str, Any]:
        """Fill common intake fields with deterministic extraction as LLM backup."""
        heuristic_fields = self._heuristic_fields(user_input, question_number)
        if not heuristic_fields:
            return extracted

        merged = dict(extracted or {})
        fields = merged.get("extracted_fields")
        fields = {} if not isinstance(fields, dict) else dict(fields)

        confidence = merged.get("field_confidence")
        confidence = {} if not isinstance(confidence, dict) else dict(confidence)

        for field, value in heuristic_fields.items():
            # income_bracket is computed deterministically from a stated
            # rupee/lakh figure — more reliable than the LLM, which often
            # mis-brackets boundary incomes (a stated "2 lakh" came back as
            # "2.5-5 lakh"). Override the LLM for it; fill-only otherwise.
            if field == "income_bracket" or fields.get(field) in (None, "", "null"):
                fields[field] = value
                confidence[field] = (
                    0.9 if field == "income_bracket" else confidence.get(field, 0.85)
                )

        merged["extracted_fields"] = fields
        merged["field_confidence"] = confidence
        if heuristic_fields:
            merged.setdefault("question_complete", True)
        return merged

    @classmethod
    def _heuristic_fields(cls, user_input: str, question_number: int) -> dict[str, Any]:
        text = user_input.strip()
        lower = text.lower()
        fields: dict[str, Any] = {}

        if question_number in (0, 1):
            state = cls._extract_state(lower)
            if state:
                fields["state"] = state

        if question_number in (0, 2):
            family_size = cls._extract_family_size(lower)
            if family_size is not None:
                fields["family_size"] = family_size

        if question_number in (0, 3):
            occupation = cls._extract_occupation(lower)
            if occupation:
                fields["occupation_type"] = occupation
            income = cls._extract_income(lower)
            if income:
                fields["income_bracket"] = income

        if question_number in (0, 4):
            coverage = cls._extract_coverage(lower)
            if coverage:
                fields["existing_coverage"] = coverage

        if "bpl" in lower:
            fields["bpl_card"] = True
        if "ration" in lower or "nfsa" in lower:
            fields["ration_card"] = True
        return fields

    @staticmethod
    def _extract_state(lower_text: str) -> str | None:
        words = re.findall(r"[\w&]+", lower_text)
        for window in range(_STATE_WINDOW_WORDS, 0, -1):
            for start in range(0, len(words) - window + 1):
                candidate = " ".join(words[start : start + window])
                code = state_name_to_code(candidate)
                if code:
                    return state_code_to_name(code)
        return None

    @staticmethod
    def _extract_family_size(lower_text: str) -> int | None:
        family_markers = ("family", "parivaar", "pariwar", "ghar", "log", "members")
        if not any(marker in lower_text for marker in family_markers):
            return None

        match = re.search(r"\b([1-9][0-9]?)\b", lower_text)
        if match:
            return int(match.group(1))

        for word, value in _NUMBER_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", lower_text):
                return value
        return None

    @staticmethod
    def _extract_occupation(lower_text: str) -> str | None:
        if any(
            marker in lower_text
            for marker in ("daily wage", "daily-wage", "mazdoori", "majdoori", "construction")
        ):
            return "daily_wage"
        if any(marker in lower_text for marker in ("sarkari", "government job", "govt job")):
            return "salaried_govt"
        if any(
            marker in lower_text for marker in ("company", "private job", "salary", "salaried")
        ):
            return "salaried_pvt"
        if any(marker in lower_text for marker in ("farmer", "kisan", "farming", "खेती")):
            return "farmer"
        if any(marker in lower_text for marker in ("business", "shop", "self employed")):
            return "self_employed"
        return None

    @staticmethod
    def _extract_income(lower_text: str) -> str | None:
        if "income tax" in lower_text or "tax bharta" in lower_text:
            return "above_5l"

        # "X lakh" amounts — the most common phrasing for annual income.
        annual_lakh = IntakeAgent._extract_annual_lakh_rupees(lower_text)
        if annual_lakh is not None:
            return IntakeAgent._bracket_for_annual(annual_lakh)

        monthly_amount = IntakeAgent._extract_monthly_rupees(lower_text)
        if monthly_amount is None:
            if "below 1 lakh" in lower_text or "bpl" in lower_text:
                return "below_1l"
            return None

        return IntakeAgent._bracket_for_annual(monthly_amount * 12)

    @staticmethod
    def _bracket_for_annual(annual_rupees: float) -> str:
        if annual_rupees < 100_000:
            return "below_1l"
        if annual_rupees <= 250_000:
            return "1l_to_2.5l"
        if annual_rupees <= 500_000:
            return "2.5l_to_5l"
        return "above_5l"

    @staticmethod
    def _extract_annual_lakh_rupees(lower_text: str) -> float | None:
        """Parse "2 lakh" / "two lakh" / "2.5 lakhs" amounts.

        Lakh figures are treated as annual income unless a monthly marker
        appears ("mahina", "per month"), in which case they are annualized.
        """
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|लाख|লাখ)\b", lower_text)
        amount: float | None = float(match.group(1)) if match else None

        if amount is None:
            for word, value in _NUMBER_WORDS.items():
                if re.search(rf"\b{re.escape(word)}\s+(?:lakhs?|lacs?|लाख|লাখ)\b", lower_text):
                    amount = float(value)
                    break

        if amount is None:
            return None

        annual = amount * 100_000
        if any(marker in lower_text for marker in ("mahina", "mahine", "per month", "monthly")):
            annual *= 12
        return annual

    @staticmethod
    def _extract_monthly_rupees(lower_text: str) -> int | None:
        hazaar_match = re.search(
            r"\b(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*(?:hazaar|hazar|thousand)\b", lower_text
        )
        if hazaar_match:
            value = int(hazaar_match.group(2) or hazaar_match.group(1))
            return value * 1000

        amount_match = re.search(r"\b(\d{4,6})\b", lower_text)
        if amount_match:
            return int(amount_match.group(1))
        return None

    _NEGATION_RE = re.compile(
        r"\b(no|not|none|never|don'?t|doesn'?t|without|nahi+n?|nei|kono|naai|illa|"
        r"নেই|নাই|নয়|নেহি)\b"
    )

    @staticmethod
    def _extract_coverage(lower_text: str) -> str | None:
        insurance_words = ("insurance", "coverage", "bima", "बीमा", "card", "scheme")
        # Word-boundary negation so "No, we don't have insurance" /
        # "kono insurance nahi" / "কোনো বীমা নেই" all read as none — a
        # plain "no " substring missed "no," and dropped the whole answer.
        if any(word in lower_text for word in insurance_words) and (
            IntakeAgent._NEGATION_RE.search(lower_text)
        ):
            return "none"
        if any(marker in lower_text for marker in ("company insurance", "employer insurance")):
            return "employer"
        if "private insurance" in lower_text:
            return "private"
        if any(marker in lower_text for marker in ("sarkari insurance", "government scheme")):
            return "govt_scheme"
        return None

    def _needs_repair(
        self,
        context: ConversationContext,
        profile: UserProfile,
        extracted: dict[str, Any],
        question_number: int,
    ) -> bool:
        """Return True when a caller answer needs one bounded clarification.

        Profile state is ground truth: once the question's required fields
        are on the profile, never re-ask — the LLM's hesitancy flags
        (``needs_followup``, low confidence) routinely fire on answers that
        extracted fine, and re-asking an answered question reads as broken.
        The end-of-intake confirmation step catches wrong extractions.
        """
        if self._question_complete(context, profile, question_number):
            return False

        # Otherwise repair — except on Q5 (the last question), where we only
        # clarify when the LLM explicitly flagged needs_followup.
        return question_number != 5 or bool(extracted.get("needs_followup"))

    def _repair_or_skip_question(
        self,
        context: ConversationContext,
        profile: UserProfile,
        extracted: dict[str, Any],
        question_number: int,
        language: str,
    ) -> AgentResponse | None:
        """Ask a clearer follow-up twice, then skip and keep the call moving."""
        counts = self._repair_counts(context)
        key = str(question_number)
        repair_count = counts.get(key, 0)

        if repair_count >= _MAX_REPAIRS_PER_QUESTION:
            self._mark_question_skipped(context, question_number)
            return None

        counts[key] = repair_count + 1
        # Always re-ask the CANONICAL question (a clearer variant), never the
        # LLM's free-form follow-up. Those drift off-script (e.g. "what's your
        # rent?"), which desyncs the 5-question flow and lands later answers
        # on the wrong question — and a stray "no" then derails confirmation.
        followup = self._get_question_text(question_number, language, fallback=True)

        return AgentResponse(
            text=followup,
            updated_profile=profile,
            metadata={
                "intake_q": question_number,
                "followup": True,
                "ux_action": "repair",
                "repair_type": "intake_low_confidence",
                "repair_count": counts[key],
                "tts_profile": "repair",
            },
        )

    def _next_missing_question(
        self,
        context: ConversationContext,
        profile: UserProfile,
        start_at: int,
    ) -> int | None:
        for question_number in range(start_at, MAX_INTAKE_QUESTIONS + 1):
            if not self._question_complete(context, profile, question_number):
                return question_number
        return None

    def _question_complete(
        self,
        context: ConversationContext,
        profile: UserProfile,
        question_number: int,
    ) -> bool:
        if question_number in self._metadata_ints(context, "intake_skipped_questions"):
            return True
        if question_number == 1:
            return profile.state is not None
        if question_number == 2:
            return profile.family_size is not None
        if question_number == 3:
            return (
                profile.occupation_type != OccupationType.UNKNOWN
                and profile.income_bracket != IncomeCategory.UNKNOWN
            )
        if question_number == 4:
            return profile.existing_coverage != CoverageType.UNKNOWN
        if question_number == 5:
            answered = self._metadata_ints(context, "intake_answered_questions")
            asked = self._metadata_ints(context, "intake_asked_questions")
            return (
                profile.health_need is not None
                or question_number in answered
                or question_number in asked
            )
        return True

    def _mark_question_asked(self, context: ConversationContext, question_number: int) -> None:
        self._add_metadata_int(context, "intake_asked_questions", question_number)

    def _mark_question_answered(self, context: ConversationContext, question_number: int) -> None:
        self._add_metadata_int(context, "intake_answered_questions", question_number)

    def _mark_question_skipped(self, context: ConversationContext, question_number: int) -> None:
        self._add_metadata_int(context, "intake_skipped_questions", question_number)

    @staticmethod
    def _repair_counts(context: ConversationContext) -> dict[str, int]:
        raw = context.metadata.get("intake_repair_counts")
        if not isinstance(raw, dict):
            raw = {}
            context.metadata["intake_repair_counts"] = raw
        clean: dict[str, int] = {}
        for key, value in raw.items():
            try:
                clean[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        context.metadata["intake_repair_counts"] = clean
        return clean

    @staticmethod
    def _metadata_ints(context: ConversationContext, key: str) -> set[int]:
        raw = context.metadata.get(key)
        if not isinstance(raw, list):
            return set()
        values: set[int] = set()
        for value in raw:
            try:
                values.add(int(value))
            except (TypeError, ValueError):
                continue
        return values

    def _add_metadata_int(
        self,
        context: ConversationContext,
        key: str,
        value: int,
    ) -> None:
        values = self._metadata_ints(context, key)
        values.add(value)
        context.metadata[key] = sorted(values)

    @staticmethod
    def _get_question_text(question_number: int, language: str, fallback: bool = False) -> str:
        """Retrieve the localized question text for a given question number."""
        if fallback:
            key = f"q{question_number}_fallback"
            text = get_msg("intake", key, language)
            if text != key:
                return text
        return get_msg("intake", f"q{question_number}", language)

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
        """Build a confirmation summary from the collected profile (PRD Section 3.2)."""
        prefix = get_msg("intake", "confirm_prefix", language)
        suffix = get_msg("intake", "confirm_suffix", language)
        conjunction = get_msg("intake", "confirm_conjunction", language)

        parts = _collect_confirmation_parts(profile, language)

        if not parts:
            return suffix
        if len(parts) == 1:
            return f"{prefix} {parts[0]}. {suffix}"

        joined = _join_with_conjunction(parts, conjunction)
        return f"{prefix} {joined}. {suffix}"

    @staticmethod
    def _build_acknowledgement(extracted: dict[str, Any], language: str) -> str:
        """A short, deterministic acknowledgement before the next question.

        Using the fixed i18n ack ("Theek hai") instead of the LLM's free-form
        ``spoken_text`` keeps each intake turn (ack + i18n question) byte-identical
        and therefore CACHEABLE — so the voice edge renders it in the caller's
        language/voice via the shared TTS cache, instead of the unique LLM text
        missing the cache and falling back to the streaming TTS, which speaks it
        in the wrong (un-switched) voice and comes out garbled. It is also
        snappier (no per-turn ack-generation latency) and reliably short.
        """
        ack = get_msg("intake", "ack", language)
        return "" if ack == "ack" else ack
