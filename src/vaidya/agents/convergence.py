"""ConvergenceChecker: pure-Python decision matrix for merging eligibility and reviewer verdicts.

No LLM calls. The logic is deterministic:

- Both agree eligible   -> output eligible (high confidence)
- Both agree ineligible -> skip (add to agreed_ineligible)
- Disagree              -> identify the divergent field, check transcript for evidence
  - Resolvable from transcript -> trust the reviewer (it catches transcript issues)
  - Not resolvable             -> conservative "uncertain" + caveat
- All disagreements are logged with both reasoning traces
"""

from __future__ import annotations

import logging

from vaidya.agents.constants import (
    DISAGREEMENT_CONFIDENCE_PENALTY,
    SINGLE_AGENT_CONFIDENCE_PENALTY,
)
from vaidya.agents.field_keywords import FIELD_KEYWORDS
from vaidya.models.conversation import ConversationContext
from vaidya.models.scheme import (
    ConvergenceResult,
    DisagreementRecord,
    EligibilityResult,
    EligibilityVerdict,
    ReviewerResult,
    SchemeMatch,
)

logger = logging.getLogger(__name__)


class ConvergenceChecker:
    """Merge eligibility and reviewer outputs into a single ConvergenceResult.

    This class is intentionally stateless -- each call to ``check()`` is
    independent. All disagreements are surfaced in the result so the audit
    trail can record them.
    """

    def check(
        self,
        eligibility: EligibilityResult,
        reviewer: ReviewerResult,
        context: ConversationContext,
    ) -> ConvergenceResult:
        """Compare eligibility and reviewer verdicts for every evaluated scheme.

        Returns a :class:`ConvergenceResult` with four buckets:
        ``agreed_eligible``, ``agreed_ineligible``, ``disagreements``, and
        ``conservative_eligible``.
        """
        elig_map: dict[str, SchemeMatch] = {m.scheme_id: m for m in eligibility.matches}
        rev_map: dict[str, SchemeMatch] = {m.scheme_id: m for m in reviewer.matches}

        all_scheme_ids = set(elig_map.keys()) | set(rev_map.keys())

        agreed_eligible: list[SchemeMatch] = []
        agreed_ineligible: list[str] = []
        disagreements: list[DisagreementRecord] = []
        conservative_eligible: list[SchemeMatch] = []

        for sid in sorted(all_scheme_ids):
            self._classify_scheme(
                sid,
                elig_map.get(sid),
                rev_map.get(sid),
                context,
                agreed_eligible,
                agreed_ineligible,
                disagreements,
                conservative_eligible,
            )

        logger.info(
            "Convergence complete",
            extra={
                "call_id": context.call_id,
                "agreed_eligible": len(agreed_eligible),
                "agreed_ineligible": len(agreed_ineligible),
                "disagreements": len(disagreements),
                "conservative_eligible": len(conservative_eligible),
            },
        )

        return ConvergenceResult(
            agreed_eligible=agreed_eligible,
            agreed_ineligible=agreed_ineligible,
            disagreements=disagreements,
            conservative_eligible=conservative_eligible,
        )

    def _classify_scheme(
        self,
        sid: str,
        e_match: SchemeMatch | None,
        r_match: SchemeMatch | None,
        context: ConversationContext,
        agreed_eligible: list[SchemeMatch],
        agreed_ineligible: list[str],
        disagreements: list[DisagreementRecord],
        conservative_eligible: list[SchemeMatch],
    ) -> None:
        """Classify a single scheme into the appropriate convergence bucket."""
        if e_match is None and r_match is not None:
            self._handle_single_verdict(
                r_match,
                "reviewer",
                agreed_eligible,
                agreed_ineligible,
                conservative_eligible,
            )
            return
        if r_match is None and e_match is not None:
            self._handle_single_verdict(
                e_match,
                "eligibility",
                agreed_eligible,
                agreed_ineligible,
                conservative_eligible,
            )
            return

        if e_match is None or r_match is None:
            return

        if e_match.verdict == r_match.verdict:
            self._handle_agreement(
                sid,
                e_match,
                r_match,
                agreed_eligible,
                agreed_ineligible,
                conservative_eligible,
            )
        else:
            self._handle_disagreement(
                e_match,
                r_match,
                context,
                disagreements,
                conservative_eligible,
            )

    def _handle_agreement(
        self,
        sid: str,
        e_match: SchemeMatch,
        r_match: SchemeMatch,
        agreed_eligible: list[SchemeMatch],
        agreed_ineligible: list[str],
        conservative_eligible: list[SchemeMatch],
    ) -> None:
        """Route same-verdict schemes to the correct bucket."""
        if e_match.verdict == EligibilityVerdict.ELIGIBLE:
            best = e_match if e_match.confidence >= r_match.confidence else r_match
            agreed_eligible.append(best)
        elif e_match.verdict == EligibilityVerdict.INELIGIBLE:
            agreed_ineligible.append(sid)
        else:
            conservative_eligible.append(self._as_uncertain_match(e_match, r_match))

    def _handle_disagreement(
        self,
        e_match: SchemeMatch,
        r_match: SchemeMatch,
        context: ConversationContext,
        disagreements: list[DisagreementRecord],
        conservative_eligible: list[SchemeMatch],
    ) -> None:
        """Resolve a disagreement and route the result.

        Surface BOTH eligible and uncertain outcomes conservatively — an
        unresolved disagreement becomes UNCERTAIN, and dropping those (the
        old behaviour) turned a genuinely-maybe-eligible caller away with
        "no scheme matched". Guidance frames uncertain schemes as "you may
        qualify, confirm at the Jan Seva Kendra", which is the correct
        conservative answer. Only a disagreement that resolves to a clear
        INELIGIBLE (reviewer found disqualifying transcript evidence) is
        dropped.
        """
        record = self._resolve_disagreement(e_match, r_match, context)
        disagreements.append(record)
        # An unresolved disagreement becomes UNCERTAIN. Surface those as
        # conservative ("you may qualify, confirm at CSC") UNLESS one agent
        # found an explicit ineligibility (a real disqualifier) — then stay
        # conservative the other way and drop it. This keeps the
        # eligible-vs-uncertain case (where neither found a disqualifier)
        # from being silently turned away as "no scheme matched".
        neither_found_disqualifier = (
            e_match.verdict != EligibilityVerdict.INELIGIBLE
            and r_match.verdict != EligibilityVerdict.INELIGIBLE
        )
        if record.final_verdict == EligibilityVerdict.ELIGIBLE or (
            record.final_verdict == EligibilityVerdict.UNCERTAIN and neither_found_disqualifier
        ):
            conservative_eligible.append(self._merge_into_match(e_match, r_match, record))

    def _resolve_disagreement(
        self,
        e: SchemeMatch,
        r: SchemeMatch,
        context: ConversationContext,
    ) -> DisagreementRecord:
        """Identify the divergent field, check transcript, and build the record."""
        divergent_field = self._identify_divergent_field(e, r)
        resolved_from_transcript = self._check_transcript_for_field(
            divergent_field,
            context.full_transcript_text,
        )
        final_verdict, caveat = self._determine_verdict(
            divergent_field,
            resolved_from_transcript,
            r.verdict,
        )

        logger.info(
            "Disagreement resolved",
            extra={
                "scheme_id": e.scheme_id,
                "divergent_field": divergent_field,
                "resolved_from_transcript": resolved_from_transcript,
                "final_verdict": final_verdict.value,
            },
        )

        return self._build_disagreement_record(
            e,
            r,
            divergent_field,
            resolved_from_transcript,
            final_verdict,
            caveat,
        )

    @staticmethod
    def _determine_verdict(
        divergent_field: str,
        resolved_from_transcript: bool,
        reviewer_verdict: EligibilityVerdict,
    ) -> tuple[EligibilityVerdict, str]:
        """Return (final_verdict, caveat) based on transcript resolution."""
        if resolved_from_transcript:
            return reviewer_verdict, (
                f"Reviewer and eligibility disagreed on '{divergent_field}'. "
                f"Transcript evidence found -- trusting reviewer verdict."
            )
        return EligibilityVerdict.UNCERTAIN, (
            f"Agents disagreed on '{divergent_field}'. "
            f"No clear transcript evidence. Verify at Jan Seva Kendra."
        )

    @staticmethod
    def _build_disagreement_record(
        e: SchemeMatch,
        r: SchemeMatch,
        divergent_field: str,
        resolved_from_transcript: bool,
        final_verdict: EligibilityVerdict,
        caveat: str,
    ) -> DisagreementRecord:
        """Construct the DisagreementRecord from resolution data."""
        return DisagreementRecord(
            scheme_id=e.scheme_id,
            scheme_name=e.scheme_name,
            eligibility_verdict=e.verdict,
            reviewer_verdict=r.verdict,
            eligibility_reasoning=e.reasoning_trace,
            reviewer_reasoning=r.reasoning_trace,
            disagreement_field=divergent_field,
            resolved_from_transcript=resolved_from_transcript,
            final_verdict=final_verdict,
            caveat=caveat,
        )

    def _identify_divergent_field(self, e: SchemeMatch, r: SchemeMatch) -> str:
        """Find the field that differs between the two agents' failed_criteria."""
        e_failed = {f.lower().strip() for f in e.failed_criteria}
        r_failed = {f.lower().strip() for f in r.failed_criteria}

        divergent_fields = (e_failed - r_failed) | (r_failed - e_failed)
        if divergent_fields:
            return sorted(divergent_fields)[0]

        combined_reasoning = (e.reasoning_trace + " " + r.reasoning_trace).lower()
        for field_name, keywords in FIELD_KEYWORDS.items():
            if any(kw in combined_reasoning for kw in keywords):
                return field_name

        return "unknown_field"

    def _check_transcript_for_field(self, field: str, transcript: str) -> bool:
        """Check whether the transcript mentions the disputed field via keywords."""
        if not transcript:
            return False

        lower_transcript = transcript.lower()
        keywords = FIELD_KEYWORDS.get(field)

        if keywords is None:
            return field.lower() in lower_transcript
        return any(kw in lower_transcript for kw in keywords)

    @staticmethod
    def _handle_single_verdict(
        match: SchemeMatch,
        source: str,
        agreed_eligible: list[SchemeMatch],
        agreed_ineligible: list[str],
        conservative_eligible: list[SchemeMatch],
    ) -> None:
        """Route a scheme evaluated by only one agent."""
        if match.verdict == EligibilityVerdict.ELIGIBLE:
            lowered = match.model_copy(
                update={"confidence": match.confidence * SINGLE_AGENT_CONFIDENCE_PENALTY},
            )
            conservative_eligible.append(lowered)
        elif match.verdict == EligibilityVerdict.INELIGIBLE:
            agreed_ineligible.append(match.scheme_id)
        else:
            conservative_eligible.append(match)

    @staticmethod
    def _as_uncertain_match(e: SchemeMatch, r: SchemeMatch) -> SchemeMatch:
        """Merge two uncertain matches into a single conservative match."""
        return SchemeMatch(
            scheme_id=e.scheme_id,
            scheme_name=e.scheme_name,
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=min(e.confidence, r.confidence),
            reasoning_trace=(
                f"Both agents uncertain. E: {e.reasoning_trace} | R: {r.reasoning_trace}"
            ),
            matched_criteria=list(set(e.matched_criteria + r.matched_criteria)),
            failed_criteria=list(set(e.failed_criteria + r.failed_criteria)),
            coverage_summary=e.coverage_summary or r.coverage_summary,
        )

    @staticmethod
    def _merge_into_match(
        e: SchemeMatch,
        r: SchemeMatch,
        record: DisagreementRecord,
    ) -> SchemeMatch:
        """Build a conservative SchemeMatch from a resolved disagreement."""
        source = r if record.final_verdict == r.verdict else e
        return SchemeMatch(
            scheme_id=source.scheme_id,
            scheme_name=source.scheme_name,
            verdict=record.final_verdict,
            confidence=source.confidence * DISAGREEMENT_CONFIDENCE_PENALTY,
            reasoning_trace=(
                f"Convergence resolved: {record.caveat}. "
                f"Original: E={e.verdict.value}, R={r.verdict.value}"
            ),
            matched_criteria=source.matched_criteria,
            failed_criteria=source.failed_criteria,
            coverage_summary=source.coverage_summary,
        )
