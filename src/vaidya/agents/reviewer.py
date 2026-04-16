"""Reviewer Agent: independent eligibility validation from raw transcript."""

from __future__ import annotations

import logging
import time
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.scheme_utils import (
    filter_schemes_by_state,
    json_compact,
    parse_verdict,
    serialize_for_prompt,
)
from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    ReviewerResult,
    SchemeMatch,
    SchemeRecord,
)
from vaidya.prompts import registry as prompts
from vaidya.sarvam.client import SarvamClient

logger = logging.getLogger(__name__)


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
        reasoning_effort: str = "high",
    ) -> None:
        super().__init__(client=client, model=model, agent_name="reviewer")
        self._schemes = schemes
        self._reasoning_effort = reasoning_effort

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
        result = await self._review(context)
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
            schemes=json_compact(schemes_payload),
        )

        raw = await self._call_llm_json(
            system,
            "Review the transcript now.",
            reasoning_effort=self._reasoning_effort,
            max_tokens=4096,
        )
        return self._parse_result(raw, candidate_schemes)

    # ------------------------------------------------------------------
    # Scheme filtering
    # ------------------------------------------------------------------

    def _filter_schemes(self, user_state: str | None) -> list[SchemeRecord]:
        """Return schemes relevant to the user's state."""
        return filter_schemes_by_state(self._schemes, user_state)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_schemes(schemes: list[SchemeRecord]) -> list[dict[str, Any]]:
        """Compact scheme representation for the reviewer prompt."""
        return serialize_for_prompt(schemes, include_procedures=False)

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
                verdict = parse_verdict(verdict_str)
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
