"""Guidance Agent: delivers eligible-scheme results and next steps via spoken voice."""

from __future__ import annotations

import logging
import time
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    ConvergenceResult,
    EligibilityVerdict,
    GuidanceOutput,
    SchemeMatch,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient
from vaidya.sarvam.models import SARVAM_30B

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# No-match fallback messages per language
# ---------------------------------------------------------------------------

_NO_MATCH: dict[str, str] = {
    "hi-IN": (
        "Abhi jo bataya usse koi yojana match nahi ho rahi. "
        "Najdeeki Jan Seva Kendra mein sab check ho sakta hai."
    ),
    "ta-IN": (
        "Neengal solliya thagavalil edhum porundhavilai. "
        "Arugil ulla Jan Seva Kendra-vil muzhumaiyaaga paarkkalaam."
    ),
    "bn-IN": (
        "Apni ja bolelen tar sathe kono yojana match hocche na. "
        "Kachhakachhi Jan Seva Kendra-te sob check hote pare."
    ),
    "en-IN": (
        "Based on what you shared, no scheme matched right now. "
        "Your nearest Jan Seva Kendra can do a full check."
    ),
}


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
    ) -> None:
        super().__init__(client=client, model=model, agent_name="guidance")

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Produce spoken guidance from the convergence result on *context*."""
        start = time.perf_counter()

        convergence = context.convergence_result
        if convergence is None:
            logger.warning(
                "Guidance agent called without convergence result",
                extra={"call_id": context.call_id},
            )
            return self._no_match_response(context.language)

        eligible = convergence.all_eligible
        if not eligible:
            return self._no_match_response(context.language)

        try:
            guidance_output = await self._generate_guidance(
                eligible=eligible,
                convergence=convergence,
                context=context,
            )
            elapsed = (time.perf_counter() - start) * 1000
            guidance_output.processing_time_ms = round(elapsed, 1)

            # Persist on context for downstream use
            context.guidance_output = guidance_output

            # Assemble flat spoken text for TTS
            spoken_text = " ".join(
                part["text"] for part in guidance_output.spoken_parts if part.get("text")
            )

            return AgentResponse(
                text=spoken_text,
                guidance_output=guidance_output,
                phase_transition=ConversationPhase.RESULTS,
                metadata={
                    "schemes_delivered": len(eligible),
                    "sms_summary": guidance_output.sms_summary,
                },
            )
        except Exception as exc:
            logger.error(
                "Guidance generation failed",
                extra={"call_id": context.call_id, "error": str(exc)},
            )
            return self._fallback_response(context.language)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_guidance(
        self,
        eligible: list[SchemeMatch],
        convergence: ConvergenceResult,
        context: ConversationContext,
    ) -> GuidanceOutput:
        """Call the LLM with the guidance prompt and parse the structured output."""
        language = context.language

        # Build a compact text block for eligible schemes
        scheme_text = self._format_schemes_for_prompt(eligible)

        # Build caveat notes
        caveats = self._build_caveats(convergence)

        # Build user profile summary
        profile = context.user_profile
        profile_text = (
            f"State: {profile.state or 'unknown'}, "
            f"Income: {profile.income_bracket.value}, "
            f"Family size: {profile.family_size or 'unknown'}, "
            f"Occupation: {profile.occupation_type.value}, "
            f"Health need: {profile.health_need or 'not specified'}"
        )

        system_prompt = prompts.render(
            "guidance_system",
            eligible_schemes=scheme_text,
            user_profile=profile_text,
            language=language,
            caveats=caveats,
        )

        user_message = (
            f"Generate spoken guidance for {len(eligible)} eligible scheme(s). "
            f"Language: {language}."
        )

        result = await self._call_llm_json(
            system_prompt,
            user_message,
            reasoning_effort="low",
            max_tokens=2048,
        )

        return self._parse_guidance_output(result, eligible, convergence)

    def _format_schemes_for_prompt(self, schemes: list[SchemeMatch]) -> str:
        """Build a numbered text summary of eligible schemes for the prompt."""
        lines: list[str] = []
        for idx, s in enumerate(schemes, 1):
            confidence_tag = ""
            if s.confidence < 0.7:
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
    ) -> GuidanceOutput:
        """Parse LLM JSON into a GuidanceOutput, with safe fallbacks."""
        spoken_parts = raw.get("spoken_parts", [])
        if not isinstance(spoken_parts, list):
            spoken_parts = []

        # Validate each part has required keys
        cleaned_parts: list[dict[str, str]] = []
        for part in spoken_parts:
            if isinstance(part, dict) and "type" in part and "text" in part:
                cleaned_parts.append({"type": str(part["type"]), "text": str(part["text"])})

        # If LLM returned nothing usable, build a minimal fallback
        if not cleaned_parts:
            cleaned_parts = self._build_fallback_parts(eligible)

        sms_summary = str(raw.get("sms_summary", ""))
        if not sms_summary:
            sms_summary = self._build_fallback_sms(eligible)

        # Enforce 160-char SMS limit
        if len(sms_summary) > 160:
            sms_summary = sms_summary[:157] + "..."

        has_more = bool(raw.get("has_more_schemes", len(eligible) > 1))
        caveat_needed = bool(raw.get("caveat_needed", len(convergence.disagreements) > 0))

        return GuidanceOutput(
            spoken_parts=cleaned_parts,
            sms_summary=sms_summary,
            has_more_schemes=has_more,
            caveat_needed=caveat_needed,
            processing_time_ms=0.0,  # filled by caller
        )

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _build_fallback_parts(self, eligible: list[SchemeMatch]) -> list[dict[str, str]]:
        """Deterministic spoken parts when LLM output is unparseable."""
        scheme = eligible[0]
        return [
            {
                "type": "headline",
                "text": f"Achi khabar hai. Aapko {scheme.scheme_name} mil sakti hai.",
            },
            {"type": "benefit", "text": scheme.coverage_summary},
            {"type": "action", "text": "Apne najdeeki Jan Seva Kendra mein jaayein."},
        ]

    def _build_fallback_sms(self, eligible: list[SchemeMatch]) -> str:
        """Deterministic SMS when LLM output is unusable."""
        names = ", ".join(s.scheme_name for s in eligible[:2])
        return f"Vaidya: {names} mil sakti hai. Jan Seva Kendra mein jaayein."[:160]

    def _no_match_response(self, language: str) -> AgentResponse:
        """Return an empathetic no-match message in the user's language."""
        text = _NO_MATCH.get(language, _NO_MATCH["hi-IN"])
        return AgentResponse(
            text=text,
            guidance_output=GuidanceOutput(
                spoken_parts=[{"type": "no_match", "text": text}],
                sms_summary="Vaidya: Koi yojana match nahi hui. Jan Seva Kendra jaayein.",
                has_more_schemes=False,
                caveat_needed=False,
                processing_time_ms=0.0,
            ),
            metadata={"no_match": True},
        )
