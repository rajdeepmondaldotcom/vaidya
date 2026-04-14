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

# ---------------------------------------------------------------------------
# Field keywords used for heuristic transcript search
# ---------------------------------------------------------------------------

_FIELD_KEYWORDS: dict[str, list[str]] = {
    "income": [
        "income",
        "kamai",
        "amdani",
        "salary",
        "tankha",
        "paisa",
        "rupee",
        "lakh",
        "hazaar",
        "maheena",
        "month",
        "saal",
        "year",
        "bpl",
        "gareebi",
        "below poverty",
    ],
    "age": [
        "age",
        "umar",
        "saal",
        "years old",
        "baras",
        "budha",
        "baccha",
        "child",
        "senior",
        "bujurg",
    ],
    "state": [
        "state",
        "rajya",
        "pradesh",
        "district",
        "zila",
        "gaon",
        "village",
        "city",
        "shehar",
    ],
    "family_size": [
        "family",
        "parivar",
        "ghar",
        "member",
        "bacche",
        "children",
        "log",
        "sadasya",
    ],
    "occupation": [
        "kaam",
        "naukri",
        "job",
        "occupation",
        "majdoor",
        "kisan",
        "farmer",
        "daily wage",
        "dihadi",
        "salaried",
        "self employed",
        "rozgaar",
        "vyapar",
        "business",
    ],
    "bpl_card": [
        "bpl",
        "ration card",
        "rashan",
        "below poverty",
        "gareebi rekha",
        "antyodaya",
        "priority household",
    ],
    "coverage": [
        "insurance",
        "bima",
        "coverage",
        "cashless",
        "hospital",
        "ayushman",
        "scheme",
        "yojana",
        "card",
    ],
    "secc": [
        "secc",
        "deprivation",
        "kachha",
        "kutcha",
        "sc",
        "st",
        "scheduled caste",
        "scheduled tribe",
        "obc",
        "category",
    ],
    "documents": [
        "aadhaar",
        "aadhar",
        "pan",
        "document",
        "dastavez",
        "kaagaz",
        "certificate",
        "pramaan patra",
    ],
}


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
        # Build fast lookup maps keyed by scheme_id
        elig_map: dict[str, SchemeMatch] = {m.scheme_id: m for m in eligibility.matches}
        rev_map: dict[str, SchemeMatch] = {m.scheme_id: m for m in reviewer.matches}

        all_scheme_ids = set(elig_map.keys()) | set(rev_map.keys())

        agreed_eligible: list[SchemeMatch] = []
        agreed_ineligible: list[str] = []
        disagreements: list[DisagreementRecord] = []
        conservative_eligible: list[SchemeMatch] = []

        for sid in sorted(all_scheme_ids):
            e_match = elig_map.get(sid)
            r_match = rev_map.get(sid)

            # Only one agent evaluated this scheme
            if e_match is None and r_match is not None:
                self._handle_single_verdict(
                    r_match,
                    "reviewer",
                    agreed_eligible,
                    agreed_ineligible,
                    conservative_eligible,
                )
                continue
            if r_match is None and e_match is not None:
                self._handle_single_verdict(
                    e_match,
                    "eligibility",
                    agreed_eligible,
                    agreed_ineligible,
                    conservative_eligible,
                )
                continue

            # Both agents evaluated
            assert e_match is not None and r_match is not None  # type guard

            if e_match.verdict == r_match.verdict:
                # Agreement
                if e_match.verdict == EligibilityVerdict.ELIGIBLE:
                    # Merge with the higher-confidence match
                    best = e_match if e_match.confidence >= r_match.confidence else r_match
                    agreed_eligible.append(best)
                elif e_match.verdict == EligibilityVerdict.INELIGIBLE:
                    agreed_ineligible.append(sid)
                else:
                    # Both say uncertain -- treat as conservative eligible
                    conservative_eligible.append(
                        self._as_uncertain_match(e_match, r_match),
                    )
            else:
                # Disagreement -- attempt resolution
                record = self._resolve_disagreement(e_match, r_match, context)
                disagreements.append(record)

                if record.final_verdict == EligibilityVerdict.ELIGIBLE:
                    conservative_eligible.append(
                        self._merge_into_match(e_match, r_match, record),
                    )
                # Ineligible and uncertain are not surfaced as eligible

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

    # ------------------------------------------------------------------
    # Disagreement resolution
    # ------------------------------------------------------------------

    def _resolve_disagreement(
        self,
        e: SchemeMatch,
        r: SchemeMatch,
        context: ConversationContext,
    ) -> DisagreementRecord:
        """Attempt to resolve a disagreement between the two agents.

        Strategy:
        1. Identify the field that caused divergence.
        2. Check the transcript for evidence about that field.
        3. If found -> trust the reviewer (it is designed to catch transcript-level issues).
        4. If not found -> conservative "uncertain" with a caveat.
        """
        divergent_field = self._identify_divergent_field(e, r)
        transcript_text = context.full_transcript_text

        resolved_from_transcript = self._check_transcript_for_field(
            divergent_field,
            transcript_text,
        )

        if resolved_from_transcript:
            # Trust the reviewer -- it validates against transcript evidence
            final_verdict = r.verdict
            caveat = (
                f"Reviewer and eligibility disagreed on '{divergent_field}'. "
                f"Transcript evidence found -- trusting reviewer verdict."
            )
        else:
            # Cannot resolve -- be conservative
            final_verdict = EligibilityVerdict.UNCERTAIN
            caveat = (
                f"Agents disagreed on '{divergent_field}'. "
                f"No clear transcript evidence. Verify at Jan Seva Kendra."
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
        """Compare failed_criteria between agents to find the disputed field.

        Heuristic: the field that appears in one agent's failed_criteria but
        not the other's is the likely divergence point.
        """
        e_failed = set(f.lower().strip() for f in e.failed_criteria)
        r_failed = set(f.lower().strip() for f in r.failed_criteria)

        # Fields that only one agent flagged as failed
        only_in_elig = e_failed - r_failed
        only_in_rev = r_failed - e_failed

        divergent_fields = only_in_elig | only_in_rev

        if divergent_fields:
            # Return the first one alphabetically for determinism
            return sorted(divergent_fields)[0]

        # If failed_criteria don't differ, fall back to reasoning comparison
        # Look for field keywords in the reasoning traces
        combined_reasoning = (e.reasoning_trace + " " + r.reasoning_trace).lower()
        for field_name, keywords in _FIELD_KEYWORDS.items():
            if any(kw in combined_reasoning for kw in keywords):
                return field_name

        return "unknown_field"

    def _check_transcript_for_field(self, field: str, transcript: str) -> bool:
        """Keyword-based heuristic: does the transcript mention the disputed field?

        A match means the user *did* say something about the field, so the
        reviewer (which reads the transcript) had more information than the
        eligibility agent (which works from the structured profile).
        """
        if not transcript:
            return False

        lower_transcript = transcript.lower()
        keywords = _FIELD_KEYWORDS.get(field)

        if keywords is None:
            # Unknown field -- try to match the field name itself
            return field.lower() in lower_transcript

        # Require at least one keyword match
        return any(kw in lower_transcript for kw in keywords)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            # Single-source eligible -- treat as conservative (lower trust)
            lowered = match.model_copy(
                update={"confidence": match.confidence * 0.8},
            )
            conservative_eligible.append(lowered)
        elif match.verdict == EligibilityVerdict.INELIGIBLE:
            agreed_ineligible.append(match.scheme_id)
        else:
            # Uncertain from single source -- still surface as conservative
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
        # Use the match from whichever agent's verdict won
        source = r if record.final_verdict == r.verdict else e
        return SchemeMatch(
            scheme_id=source.scheme_id,
            scheme_name=source.scheme_name,
            verdict=record.final_verdict,
            confidence=source.confidence * 0.7,  # penalise for disagreement
            reasoning_trace=(
                f"Convergence resolved: {record.caveat}. "
                f"Original: E={e.verdict.value}, R={r.verdict.value}"
            ),
            matched_criteria=source.matched_criteria,
            failed_criteria=source.failed_criteria,
            coverage_summary=source.coverage_summary,
        )
