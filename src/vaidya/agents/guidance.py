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
    SchemeMatch,
    SpokenPart,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import SARVAM_30B

logger = logging.getLogger(__name__)

_SMS_MAX_LENGTH = 160


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
    conversation context and produces a three-part spoken output:

    1. **Headline** -- one punchy sentence announcing the good news.
    2. **Benefit**  -- what the scheme gives in plain rupee terms.
    3. **Action**   -- documents to collect, where to go, timeline.

    For multiple schemes it uses the *"ek aur suno"* pattern: deliver one
    scheme at a time, then offer to read out the next.
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
        """Produce spoken guidance from the convergence result on *context*.

        When *user_input* starts with ``__deliver_scheme_index:N`` the agent
        generates guidance for only the Nth scheme (0-based) instead of all.
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

        if user_input.startswith("__deliver_scheme_index:"):
            try:
                idx = int(user_input.split(":")[1])
                if 0 <= idx < len(eligible):
                    eligible = [eligible[idx]]
            except (ValueError, IndexError):
                pass

        guidance_output = await self._generate_guidance(
            eligible=eligible,
            convergence=convergence,
            context=context,
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
        """Build a numbered text summary of eligible schemes for the prompt."""
        lines: list[str] = []
        for idx, s in enumerate(schemes, 1):
            confidence_tag = ""
            if s.confidence < LOW_CONFIDENCE_THRESHOLD:
                confidence_tag = " [NEEDS VERIFICATION]"
            lines.append(
                f"{idx}. {s.scheme_name} (ID: {s.scheme_id})\n"
                f"   Coverage: {s.coverage_summary}\n"
                f"   Confidence: {s.confidence:.0%}{confidence_tag}\n"
                f"   Matched: {', '.join(s.matched_criteria)}"
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
    ) -> list[SpokenPart]:
        """Deterministic spoken parts when LLM output is unparseable."""
        scheme = eligible[0]
        headline = get_msg_template(
            "guidance",
            "fallback_headline",
            language,
            scheme_name=scheme.scheme_name,
        )
        return [
            SpokenPart(type="headline", text=headline),
            SpokenPart(type="benefit", text=scheme.coverage_summary),
            SpokenPart(type="action", text=get_msg("guidance", "fallback_action", language)),
        ]

    def _build_fallback_sms(self, eligible: list[SchemeMatch], language: str = "hi-IN") -> str:
        """Deterministic SMS when LLM output is unusable."""
        names = ", ".join(s.scheme_name for s in eligible[:2])
        sms = get_msg_template("guidance", "fallback_sms", language, names=names)
        return sms[:_SMS_MAX_LENGTH]

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
