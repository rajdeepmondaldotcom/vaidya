"""Reviewer Agent: independent eligibility validation from raw transcript."""

from __future__ import annotations

import logging
import time
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    EligibilityVerdict,
    Jurisdiction,
    ReviewerResult,
    SchemeMatch,
    SchemeRecord,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)

# Maximum number of schemes to include in the reviewer prompt
_MAX_SCHEMES_PER_CALL = 20


class ReviewerAgent(BaseAgent):
    """Independently validates eligibility by reading the full transcript.

    This agent exists specifically to catch what the structured
    Eligibility Agent misses: exclusion criteria mentioned in passing
    (e.g. "company ka insurance to hai"), corrections the user made
    mid-conversation, code-mixed asides, and contradictions between
    early and late statements.

    Key design difference from EligibilityAgent: the Reviewer works
    from the raw conversation narrative, not the structured UserProfile.
    Its reasoning path is transcript-evidence-based rather than
    field-by-field matching.
    """

    def __init__(
        self,
        client: SarvamClient,
        model: str,
        schemes: list[SchemeRecord],
    ) -> None:
        super().__init__(client=client, model=model, agent_name="reviewer")
        self._schemes = schemes

    async def process(
        self,
        context: ConversationContext,
        user_input: str,
    ) -> AgentResponse:
        """Analyze the full transcript and produce an independent review.

        The *user_input* argument is unused (the orchestrator passes
        an empty string); all data comes from context.full_transcript_text.
        """
        start_ms = time.perf_counter()

        try:
            result = await self._review(context)
        except Exception as exc:
            logger.error(
                "Reviewer agent failed",
                extra={"error": str(exc), "call_id": context.call_id},
                exc_info=True,
            )
            return self._fallback_response(context.language)

        elapsed_ms = (time.perf_counter() - start_ms) * 1000
        result.processing_time_ms = round(elapsed_ms, 1)

        return AgentResponse(
            text="",  # Reviewer produces data, not spoken text
            reviewer_result=result,
            metadata={
                "reviewer_matches": len(result.matches),
                "transcript_evidence_count": len(result.transcript_evidence),
                "processing_time_ms": result.processing_time_ms,
            },
        )

    # ------------------------------------------------------------------
    # Core review logic
    # ------------------------------------------------------------------

    async def _review(
        self,
        context: ConversationContext,
    ) -> ReviewerResult:
        """Build prompt from transcript + schemes, call LLM, parse result."""
        transcript_text = context.full_transcript_text

        if not transcript_text.strip():
            logger.warning(
                "Reviewer called with empty transcript",
                extra={"call_id": context.call_id},
            )
            return ReviewerResult(
                matches=[],
                processing_time_ms=0,
                model_used=self._model,
                transcript_evidence=[],
            )

        # Filter schemes by state if we can glean it from the profile
        # (the reviewer should be state-aware even though it reads the transcript)
        candidate_schemes = self._filter_schemes(context.user_profile.state)
        schemes_payload = self._serialize_schemes(candidate_schemes)

        system = prompts.render(
            "reviewer_system",
            transcript=transcript_text,
            schemes=_json_str(schemes_payload),
        )

        raw = await self._call_llm_json(
            system,
            "Review the transcript now.",
            reasoning_effort="high",
            max_tokens=4096,
        )
        return self._parse_result(raw, candidate_schemes)

    # ------------------------------------------------------------------
    # Scheme filtering (shared logic with eligibility, but intentionally
    # duplicated — the reviewer is an independent agent)
    # ------------------------------------------------------------------

    def _filter_schemes(self, user_state: str | None) -> list[SchemeRecord]:
        """Return schemes relevant to the user's state.

        Central schemes are always included. State schemes are included
        only when the state matches or the state is unknown.
        """
        if not user_state:
            return self._schemes

        user_state_lower = user_state.lower().strip()
        filtered: list[SchemeRecord] = []

        for scheme in self._schemes:
            if scheme.jurisdiction == Jurisdiction.CENTRAL:
                filtered.append(scheme)
                continue

            if not scheme.geographic_restrictions:
                filtered.append(scheme)
                continue

            if any(user_state_lower in r.lower() for r in scheme.geographic_restrictions):
                filtered.append(scheme)

        return filtered

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_schemes(schemes: list[SchemeRecord]) -> list[dict[str, Any]]:
        """Compact scheme representation for the reviewer prompt.

        The reviewer needs enough detail to check exclusion rules but
        does not need the full enrollment / document data.
        """
        out: list[dict[str, Any]] = []
        for s in schemes[:_MAX_SCHEMES_PER_CALL]:
            out.append(
                {
                    "scheme_id": s.scheme_id,
                    "canonical_name": s.canonical_name,
                    "jurisdiction": s.jurisdiction.value,
                    "state_code": s.state_code,
                    "income_thresholds": [t.model_dump(mode="json") for t in s.income_thresholds],
                    "occupation_included": s.occupation_included,
                    "occupation_excluded": s.occupation_excluded,
                    "exclusion_rules": [r.model_dump(mode="json") for r in s.exclusion_rules],
                    "age_criteria": (
                        s.age_criteria.model_dump(mode="json") if s.age_criteria else None
                    ),
                    "family_criteria": s.family_criteria.model_dump(mode="json"),
                    "geographic_restrictions": s.geographic_restrictions,
                    "coverage_amount_inr": s.coverage_amount_inr,
                    "coverage_type": s.coverage_type.value,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_result(
        self,
        raw: dict[str, Any],
        candidate_schemes: list[SchemeRecord],
    ) -> ReviewerResult:
        """Parse LLM JSON into a validated ReviewerResult."""
        # Handle cases where LLM returns a list instead of dict
        if isinstance(raw, list):
            raw = {"matches": raw}
        if not isinstance(raw, dict) or raw.get("_parse_error"):
            logger.warning("LLM returned unparseable JSON for reviewer")
            return ReviewerResult(
                matches=[],
                processing_time_ms=0,
                model_used=self._model,
                transcript_evidence=[],
            )

        # Build a scheme_id -> name lookup for populating scheme_name
        name_lookup: dict[str, str] = {s.scheme_id: s.canonical_name for s in candidate_schemes}

        matches: list[SchemeMatch] = []
        all_evidence: list[str] = []

        for item in raw.get("matches", []):
            try:
                verdict_str = str(item.get("verdict", "uncertain")).lower()
                verdict = _parse_verdict(verdict_str)
                scheme_id = str(item.get("scheme_id", ""))

                # Collect transcript evidence from this match
                item_evidence = item.get("transcript_evidence", [])
                if isinstance(item_evidence, list):
                    all_evidence.extend(str(e) for e in item_evidence)

                # Collect fields the reviewer found that the profile missed
                missed_by_profile = item.get("missed_by_profile", [])
                reasoning = str(item.get("reasoning_trace", ""))
                if missed_by_profile:
                    missed_str = "; ".join(str(m) for m in missed_by_profile)
                    reasoning += f" [Missed by structured profile: {missed_str}]"

                match = SchemeMatch(
                    scheme_id=scheme_id,
                    scheme_name=name_lookup.get(scheme_id, scheme_id),
                    verdict=verdict,
                    confidence=float(item.get("confidence", 0.0)),
                    reasoning_trace=reasoning,
                    matched_criteria=item.get("matched_criteria", []),
                    failed_criteria=item.get("failed_criteria", []),
                    coverage_summary=str(item.get("coverage_summary", "")),
                )
                matches.append(match)
            except Exception as exc:
                logger.warning(
                    "Skipping malformed reviewer match",
                    extra={"error": str(exc), "item": str(item)[:200]},
                )

        return ReviewerResult(
            matches=matches,
            processing_time_ms=0,  # filled by caller
            model_used=self._model,
            transcript_evidence=all_evidence,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_verdict(raw: str) -> EligibilityVerdict:
    """Map raw string to EligibilityVerdict, defaulting to UNCERTAIN."""
    mapping = {
        "eligible": EligibilityVerdict.ELIGIBLE,
        "ineligible": EligibilityVerdict.INELIGIBLE,
        "uncertain": EligibilityVerdict.UNCERTAIN,
    }
    return mapping.get(raw.strip().lower(), EligibilityVerdict.UNCERTAIN)


def _json_str(obj: Any) -> str:
    """Compact JSON string for prompt embedding."""
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
