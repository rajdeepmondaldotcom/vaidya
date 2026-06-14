"""Reviewer Agent: independent eligibility validation from raw transcript."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from vaidya.agents.base import BaseAgent
from vaidya.agents.constants import MAX_PARALLEL_SCHEME_BATCHES, MAX_SCHEMES_PER_LLM_CALL
from vaidya.agents.scheme_utils import (
    batch_schemes,
    filter_schemes_by_state,
    json_compact,
    missing_candidate_ids,
    normalize_matches_for_candidates,
    parse_verdict,
    serialize_for_prompt,
    uncertain_matches_for_missing,
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
        batch_size: int = MAX_SCHEMES_PER_LLM_CALL,
        max_parallel_batches: int = MAX_PARALLEL_SCHEME_BATCHES,
    ) -> None:
        super().__init__(client=client, model=model, agent_name="reviewer")
        self._schemes = schemes
        self._reasoning_effort = reasoning_effort
        self._batch_size = max(1, batch_size)
        self._max_parallel_batches = max(1, max_parallel_batches)

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
                "reviewer_batch_count": context.metadata.get("reviewer_batch_count", 0),
                "reviewer_missing_scheme_ids": context.metadata.get(
                    "reviewer_missing_scheme_ids",
                    [],
                ),
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

        candidate_schemes = self._filter_schemes(context.user_profile.state)
        batches = batch_schemes(candidate_schemes, self._batch_size)
        context.metadata["reviewer_candidate_count"] = len(candidate_schemes)
        context.metadata["reviewer_batch_count"] = len(batches)
        context.metadata["reviewer_missing_scheme_ids"] = []

        batch_results = await self._review_batches(
            context,
            transcript_text,
            batches,
        )
        matches_by_id: dict[str, SchemeMatch] = {}
        transcript_evidence: list[str] = []
        for result in batch_results:
            for match in result.matches:
                matches_by_id.setdefault(match.scheme_id, match)
            transcript_evidence.extend(result.transcript_evidence)

        matches = [
            matches_by_id[scheme.scheme_id]
            for scheme in candidate_schemes
            if scheme.scheme_id in matches_by_id
        ]
        return ReviewerResult(
            matches=matches,
            processing_time_ms=0,
            model_used=self._model,
            transcript_evidence=transcript_evidence,
        )

    async def _review_batches(
        self,
        context: ConversationContext,
        transcript_text: str,
        batches: list[list[SchemeRecord]],
    ) -> list[ReviewerResult]:
        """Review scheme batches with bounded concurrency."""
        semaphore = asyncio.Semaphore(self._max_parallel_batches)

        async def _run(batch: list[SchemeRecord], batch_index: int) -> ReviewerResult:
            async with semaphore:
                return await self._review_batch(context, transcript_text, batch, batch_index)

        return await asyncio.gather(*[_run(batch, idx) for idx, batch in enumerate(batches)])

    async def _review_batch(
        self,
        context: ConversationContext,
        transcript_text: str,
        batch: list[SchemeRecord],
        batch_index: int,
    ) -> ReviewerResult:
        """Review one batch and retry once if scheme IDs are omitted."""
        result = await self._review_batch_once(transcript_text, batch)
        result = self._normalize_batch_result(result, batch)
        missing = missing_candidate_ids(result.matches, batch)

        if missing:
            logger.warning(
                "Reviewer batch omitted schemes; retrying once",
                extra={
                    "batch_index": batch_index,
                    "missing_scheme_ids": missing,
                    "call_id": context.call_id,
                },
            )
            retry = await self._review_batch_once(transcript_text, batch)
            retry = self._normalize_batch_result(retry, batch)
            result = self._merge_batch_retry(result, retry, batch)
            result.transcript_evidence.extend(retry.transcript_evidence)
            missing = missing_candidate_ids(result.matches, batch)

        if missing:
            context.metadata.setdefault("reviewer_missing_scheme_ids", []).extend(missing)
            result.matches.extend(uncertain_matches_for_missing(missing, batch, source="reviewer"))

        result.matches = normalize_matches_for_candidates(result.matches, batch)
        return result

    async def _review_batch_once(
        self,
        transcript_text: str,
        candidate_schemes: list[SchemeRecord],
    ) -> ReviewerResult:
        """Build one reviewer prompt, call the LLM, and parse the batch result."""
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
            # 8192, not the free-tier 4096: on the paid tier a 105b reasoning
            # pass can spend the 4096 budget on its reasoning trace and truncate
            # the JSON answer, which surfaces as "Failed to parse LLM JSON" and a
            # full ~26s retry. The extra headroom lets the answer complete on the
            # first attempt; max_tokens is a cap, so it never slows a short reply.
            max_tokens=8192,
        )
        return self._parse_result(raw, candidate_schemes)

    @staticmethod
    def _normalize_batch_result(
        result: ReviewerResult,
        batch: list[SchemeRecord],
    ) -> ReviewerResult:
        result.matches = normalize_matches_for_candidates(result.matches, batch)
        return result

    @staticmethod
    def _merge_batch_retry(
        first: ReviewerResult,
        retry: ReviewerResult,
        batch: list[SchemeRecord],
    ) -> ReviewerResult:
        matches_by_id: dict[str, SchemeMatch] = {}
        for match in first.matches + retry.matches:
            matches_by_id.setdefault(match.scheme_id, match)
        first.matches = [
            matches_by_id[scheme.scheme_id]
            for scheme in batch
            if scheme.scheme_id in matches_by_id
        ]
        return first

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

        raw_matches = raw.get("matches", [])
        if not isinstance(raw_matches, list):
            raw_matches = []
        for item in raw_matches:
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
