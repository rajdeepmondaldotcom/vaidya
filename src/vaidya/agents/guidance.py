"""Guidance Agent: delivers eligible-scheme results and next steps via spoken voice."""

from __future__ import annotations

import logging
import time
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.constants import LOW_CONFIDENCE_THRESHOLD
from vaidya.i18n import get_msg, get_msg_template
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    ConvergenceResult,
    EligibilityVerdict,
    GuidanceOutput,
    Jurisdiction,
    SchemeMatch,
    SchemeRecord,
    SpokenPart,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import SARVAM_30B
from vaidya.schemes.registry import get_schemes
from vaidya.utils.states import state_name_to_code

logger = logging.getLogger(__name__)

_SMS_MAX_LENGTH = 160
# Schemes spoken aloud on a call. Reading every eligible scheme (often 14-18,
# many of them universal national programmes) is a multi-minute monologue; speak
# the most relevant few and SMS the complete list.
_MAX_SPOKEN_SCHEMES = 5

# Maps a caller's stated condition to the words that identify the scheme(s) which
# directly address it (in scheme names / covered procedures), so a TB patient
# hears NTEP, an eye patient hears the blindness-control programme, a heart/BP/
# diabetes patient hears the NCD programme, etc. Keys are matched as substrings
# of the (lower-cased) health need, including common transliterations.
_CONDITION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tb": ("tuberculosis",),
    "tuberculosis": ("tuberculosis",),
    "kshay": ("tuberculosis",),
    "jokkha": ("tuberculosis",),
    "cataract": ("cataract", "blindness", "visual"),
    "eye": ("cataract", "blindness", "visual", "ophthal"),
    "aankh": ("cataract", "blindness", "visual"),
    "kannu": ("cataract", "blindness", "visual"),
    "blind": ("blindness", "visual"),
    "vision": ("blindness", "visual"),
    "heart": ("cardio", "cardiac", "non-communicable", "hypertension"),
    "cardiac": ("cardio", "cardiac"),
    "dil": ("cardio", "cardiac", "non-communicable", "hypertension"),
    "bp": ("hypertension", "non-communicable"),
    "hypertension": ("hypertension", "non-communicable"),
    "diabet": ("diabetes", "non-communicable"),
    "sugar": ("diabetes", "non-communicable"),
    "cancer": ("cancer", "oncology"),
    "tumour": ("cancer", "oncology"),
    "mental": ("mental",),
    "depress": ("mental",),
    "kidney": ("dialysis", "renal", "nephro"),
    "dialysis": ("dialysis", "renal"),
    "pregnan": ("maternal", "matru", "janani", "pregnan"),
    "matern": ("maternal", "matru", "janani"),
    "baby": ("child", "shishu", "bal swasthya", "newborn"),
    "child": ("child", "shishu", "bal swasthya", "newborn"),
    "accident": ("accident", "suraksha"),
}


def _scheme_matches_need(record: SchemeRecord | None, health_need: str) -> bool:
    """True when *record* directly addresses the caller's stated *health_need*.

    Matches the need (and its transliteration/synonym expansions) against the
    scheme's name, aliases, and covered procedures. Best-effort: a miss simply
    means the scheme isn't condition-boosted, never a wrong answer.
    """
    if record is None or not health_need:
        return False
    need = health_need.lower()
    targets: set[str] = set()
    for key, expansions in _CONDITION_KEYWORDS.items():
        if key in need:
            targets.update(expansions)
    # Also try the caller's own words (already-English needs like "tuberculosis").
    targets.update(tok for tok in need.split() if len(tok) > 3)
    if not targets:
        return False
    haystack = " ".join(
        [record.canonical_name, *record.aliases, *record.covered_procedures]
    ).lower()
    return any(t in haystack for t in targets)


def _relevance_score(match: SchemeMatch, record: SchemeRecord | None, profile: object) -> float:
    """Rank schemes for the SPOKEN top-N: the scheme the caller actually enrols in
    (their state scheme) first, then PM-JAY, then ones addressing their condition,
    then by financial cover — so the handful spoken aloud is the most useful, not
    the universal low-value programmes that merely happen to score high confidence.
    """
    score = float(getattr(match, "confidence", 0.0))  # 0..1 base / tiebreaker
    if record is None:
        return score
    state = getattr(profile, "state", None)
    state_code = state_name_to_code(state) if state else None
    if (
        record.jurisdiction == Jurisdiction.STATE
        and state_code
        and record.state_code == state_code
    ):
        score += 100.0  # the caller's own state scheme — their enrolment vehicle
    if "PMJAY" in match.scheme_id.upper():
        score += 80.0  # national flagship cover
    if _scheme_matches_need(record, getattr(profile, "health_need", "") or ""):
        score += 60.0  # directly addresses the stated condition
    score += min(record.coverage_amount_inr, 2_500_000) / 100_000.0  # financial cover, up to ~25
    return score


def _extract_spoken_parts(raw: dict[str, Any]) -> list[SpokenPart]:
    """Validate and extract SpokenPart objects from raw LLM output."""
    spoken_parts = raw.get("spoken_parts", [])
    if not isinstance(spoken_parts, list):
        return []
    return [
        SpokenPart(type=str(part["type"]), text=str(part["text"]))
        for part in spoken_parts
        if isinstance(part, dict) and "type" in part and "text" in part
    ]


class GuidanceAgent(BaseAgent):
    """Generates TTS-ready spoken guidance from converged eligibility results.

    The guidance agent takes the ``convergence_result`` attached to the
    conversation context and produces ONE combined spoken results message
    that names EVERY eligible scheme in a single turn:

    1. **Intro**   -- "You may qualify for N schemes:".
    2. **Schemes** -- one concise advisory line per scheme (name + benefit).
    3. **Offer**   -- offer fuller detail on any one + an SMS of the full list.

    There is no longer a one-scheme-at-a-time "ek aur suno" gate -- the caller
    hears all their schemes at once and can then ask for detail on any of them.
    """

    def __init__(
        self,
        client: SarvamClient,
        model: str = SARVAM_30B,
        reasoning_effort: str = "low",
    ) -> None:
        super().__init__(client=client, model=model, agent_name="guidance")
        self._reasoning_effort = reasoning_effort

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Produce ONE combined spoken results message for ALL eligible schemes.

        Every eligible scheme on ``context.convergence_result`` is named in a
        single turn -- there is no per-scheme drip-feed.
        """
        start = time.perf_counter()

        convergence = context.convergence_result
        if convergence is None:
            logger.warning(
                "Guidance agent called without convergence result",
                extra={"call_id": context.call_id},
            )
            return self._no_match_response(context.language)

        eligible = convergence.all_eligible or []
        if not eligible:
            return self._no_match_response(context.language)

        # Speak only the most relevant handful; the SMS carries the full list.
        # The free-form LLM path read every eligible scheme (sometimes twice) and
        # risked the wrong TTS voice. Deterministic template parts are capped,
        # never duplicated, and render in the caller's voice via the TTS cache.
        # Rank by RELEVANCE (state scheme -> PM-JAY -> condition match -> cover),
        # not raw confidence, so the spoken few are the schemes that actually
        # matter to this caller rather than the universal low-value programmes.
        by_id = {s.scheme_id: s for s in get_schemes()}
        profile = context.user_profile
        ranked = sorted(
            eligible,
            key=lambda m: _relevance_score(m, by_id.get(m.scheme_id), profile),
            reverse=True,
        )
        spoken_schemes = ranked[:_MAX_SPOKEN_SCHEMES]
        guidance_output = GuidanceOutput(
            spoken_parts=self._build_fallback_parts(
                spoken_schemes, context.language, total_count=len(eligible)
            ),
            sms_summary=self._build_fallback_sms(eligible, context.language),
            has_more_schemes=len(eligible) > len(spoken_schemes),
            caveat_needed=len(convergence.disagreements) > 0,
            processing_time_ms=0.0,
        )
        elapsed = (time.perf_counter() - start) * 1000
        guidance_output.processing_time_ms = round(elapsed, 1)

        context.guidance_output = guidance_output
        spoken_text = " ".join(part.text for part in guidance_output.spoken_parts if part.text)

        return AgentResponse(
            text=spoken_text,
            guidance_output=guidance_output,
            phase_transition=ConversationPhase.RESULTS,
            metadata={
                "schemes_delivered": len(eligible),
                "sms_summary": guidance_output.sms_summary,
            },
        )

    async def _generate_guidance(
        self,
        eligible: list[SchemeMatch],
        convergence: ConvergenceResult,
        context: ConversationContext,
    ) -> GuidanceOutput:
        """Call the LLM with the guidance prompt and parse the structured output."""
        language = context.language
        system_prompt = self._build_guidance_prompt(eligible, convergence, context)

        user_message = (
            f"Generate spoken guidance for {len(eligible)} eligible scheme(s). "
            f"Language: {language}."
        )

        result = await self._call_llm_json(
            system_prompt,
            user_message,
            reasoning_effort=self._reasoning_effort,
            max_tokens=4096,
        )

        return self._parse_guidance_output(result, eligible, convergence, language)

    def _build_guidance_prompt(
        self,
        eligible: list[SchemeMatch],
        convergence: ConvergenceResult,
        context: ConversationContext,
    ) -> str:
        """Assemble the system prompt from schemes, caveats, and profile."""
        profile = context.user_profile
        profile_text = (
            f"State: {profile.state or 'unknown'}, "
            f"Income: {profile.income_bracket.value}, "
            f"Family size: {profile.family_size or 'unknown'}, "
            f"Occupation: {profile.occupation_type.value}, "
            f"Health need: {profile.health_need or 'not specified'}"
        )

        return prompts.render(
            "guidance_system",
            eligible_schemes=self._format_schemes_for_prompt(eligible),
            user_profile=profile_text,
            language=context.language,
            caveats=self._build_caveats(convergence),
        )

    def _format_schemes_for_prompt(self, schemes: list[SchemeMatch]) -> str:
        """Build a numbered text summary of eligible schemes for the prompt.

        Deliberately omits the internal scheme_id and the raw
        matched_criteria field names ("income_bracket", "existing_coverage")
        — the model echoed those into the spoken result, and after
        translation the underscores were read aloud as "underscore". The
        guidance only needs the human name + benefit to speak.
        """
        lines: list[str] = []
        for idx, s in enumerate(schemes, 1):
            confidence_tag = ""
            if s.confidence < LOW_CONFIDENCE_THRESHOLD:
                confidence_tag = " [NEEDS VERIFICATION]"
            lines.append(
                f"{idx}. {s.scheme_name}\n"
                f"   Coverage: {s.coverage_summary}\n"
                f"   Confidence: {s.confidence:.0%}{confidence_tag}"
            )
        return "\n".join(lines)

    def _build_caveats(self, convergence: ConvergenceResult) -> str:
        """Assemble caveats from disagreements and conservative matches."""
        parts: list[str] = []

        if convergence.disagreements:
            for d in convergence.disagreements:
                if d.final_verdict == EligibilityVerdict.UNCERTAIN:
                    parts.append(
                        f"Scheme {d.scheme_name}: uncertain on '{d.disagreement_field}'. "
                        f"Verify at Jan Seva Kendra."
                    )

        if convergence.conservative_eligible:
            ids = [s.scheme_name for s in convergence.conservative_eligible]
            parts.append(
                f"Conservative matches ({', '.join(ids)}): "
                "some criteria could not be fully verified from the call."
            )

        return "; ".join(parts) if parts else "None"

    def _parse_guidance_output(
        self,
        raw: dict[str, Any],
        eligible: list[SchemeMatch],
        convergence: ConvergenceResult,
        language: str = "hi-IN",
    ) -> GuidanceOutput:
        """Parse LLM JSON into a GuidanceOutput, with safe fallbacks."""
        if raw.get("_parse_error"):
            logger.warning(
                "LLM returned a parse error, using fallback guidance",
                extra={"error": raw.get("_parse_error")},
            )
            return GuidanceOutput(
                spoken_parts=self._build_fallback_parts(eligible, language),
                sms_summary=self._build_fallback_sms(eligible, language),
                has_more_schemes=len(eligible) > 1,
                caveat_needed=len(convergence.disagreements) > 0,
                processing_time_ms=0.0,
            )

        cleaned_parts = _extract_spoken_parts(raw)
        if not cleaned_parts:
            cleaned_parts = self._build_fallback_parts(eligible, language)

        sms_summary = str(raw.get("sms_summary", ""))
        if not sms_summary:
            sms_summary = self._build_fallback_sms(eligible, language)
        if len(sms_summary) > _SMS_MAX_LENGTH:
            sms_summary = sms_summary[: _SMS_MAX_LENGTH - 3] + "..."

        return GuidanceOutput(
            spoken_parts=cleaned_parts,
            sms_summary=sms_summary,
            has_more_schemes=bool(raw.get("has_more_schemes", len(eligible) > 1)),
            caveat_needed=bool(raw.get("caveat_needed", len(convergence.disagreements) > 0)),
            processing_time_ms=0.0,
        )

    def _build_fallback_parts(
        self,
        eligible: list[SchemeMatch],
        language: str = "hi-IN",
        total_count: int | None = None,
    ) -> list[SpokenPart]:
        """Deterministic combined spoken parts naming EVERY eligible scheme.

        Produces a single results turn:
        1. an intro announcing the count of schemes,
        2. one concise advisory line per scheme (name + one-line benefit),
        3. a closing offer of fuller detail on any one + an SMS of the list.

        Used both as the LLM fallback and whenever the LLM returns no usable
        spoken parts, so the caller always hears all their schemes at once.
        """
        # ``total_count`` is the full eligible count even when only the top few
        # SchemeMatch objects are passed in to be spoken; the intro announces the
        # real total and the closing offer points to the SMS for the rest.
        count = total_count if total_count is not None else len(eligible)
        if count == 1:
            intro = get_msg_template(
                "guidance",
                "fallback_headline",
                language,
                scheme_name=eligible[0].scheme_name,
            )
        else:
            intro = get_msg_template("guidance", "results_intro", language, count=count)

        parts: list[SpokenPart] = [SpokenPart(type="intro", text=intro)]
        for scheme in eligible:
            line = get_msg_template(
                "guidance",
                "results_scheme_line",
                language,
                scheme_name=scheme.scheme_name,
                benefit=scheme.coverage_summary,
            )
            parts.append(SpokenPart(type="scheme", text=line))

        parts.append(
            SpokenPart(type="offer", text=get_msg("guidance", "results_offer_detail", language))
        )
        return parts

    def _build_fallback_sms(self, eligible: list[SchemeMatch], language: str = "hi-IN") -> str:
        """Deterministic SMS listing every eligible scheme (truncated to 160 chars)."""
        names = ", ".join(s.scheme_name for s in eligible)
        sms = get_msg_template("guidance", "fallback_sms", language, names=names)
        if len(sms) > _SMS_MAX_LENGTH:
            sms = sms[: _SMS_MAX_LENGTH - 3] + "..."
        return sms

    def _no_match_response(self, language: str) -> AgentResponse:
        """Return an empathetic no-match message in the user's language."""
        text = get_msg("guidance", "no_match", language)
        return AgentResponse(
            text=text,
            guidance_output=GuidanceOutput(
                spoken_parts=[SpokenPart(type="no_match", text=text)],
                sms_summary=get_msg("guidance", "no_match_sms", language),
                has_more_schemes=False,
                caveat_needed=False,
                processing_time_ms=0.0,
            ),
            metadata={"no_match": True},
        )
