"""Tests for ConvergenceChecker: all 9 cells of the decision matrix.

Matrix axes:
  Eligibility agent verdict: ELIGIBLE | INELIGIBLE | UNCERTAIN
  Reviewer agent verdict:    ELIGIBLE | INELIGIBLE | UNCERTAIN

Cells:
  1. (E, E)  -> agreed eligible (high confidence winner)
  2. (I, I)  -> agreed ineligible
  3. (E, I)  -> disagreement -> resolution via transcript
  4. (I, E)  -> disagreement -> resolution via transcript
  5. (E, U)  -> disagreement -> resolution via transcript
  6. (U, E)  -> disagreement -> resolution via transcript
  7. (I, U)  -> disagreement -> resolution via transcript
  8. (U, I)  -> disagreement -> resolution via transcript
  9. (U, U)  -> conservative_eligible (min confidence, merged reasoning)

All cells where verdicts differ (3-8) go through _resolve_disagreement.
Resolution trusts the reviewer when transcript evidence is found, else UNCERTAIN.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    make_eligibility_result,
    make_reviewer_result,
    make_scheme_match,
)
from vaidya.agents.convergence import ConvergenceChecker
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import EligibilityVerdict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def checker() -> ConvergenceChecker:
    return ConvergenceChecker()


@pytest.fixture()
def ctx_with_income_keywords() -> ConversationContext:
    """Context with transcript mentioning income/occupation keywords."""
    context = ConversationContext(
        call_id="conv-test-001",
        phone_number_hash="hash_test",
        language="hi-IN",
        phase=ConversationPhase.PROCESSING,
    )
    context.add_turn(
        role="user",
        text="Meri income 80 hazaar saal ki hai, daily wage kaam karta hoon",
        raw_text="Meri income 80 hazaar saal ki hai, daily wage kaam karta hoon",
    )
    return context


@pytest.fixture()
def ctx_empty_transcript() -> ConversationContext:
    """Context with no transcript turns -- no evidence for resolution."""
    return ConversationContext(
        call_id="conv-test-002",
        phone_number_hash="hash_test",
        language="hi-IN",
        phase=ConversationPhase.PROCESSING,
    )


# -----------------------------------------------------------------------
# Cell 1: Both ELIGIBLE -> agreed_eligible
# -----------------------------------------------------------------------


class TestBothEligible:
    """Cell 1: both agents say ELIGIBLE -> agreed_eligible bucket."""

    def test_agreed_eligible(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.9)
        r = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.85)

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.agreed_eligible) == 1
        assert result.agreed_eligible[0].scheme_id == "S1"
        assert result.agreed_eligible[0].verdict == EligibilityVerdict.ELIGIBLE
        assert len(result.agreed_ineligible) == 0
        assert len(result.disagreements) == 0
        assert len(result.conservative_eligible) == 0

    def test_higher_confidence_wins(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When both agree eligible, the match with higher confidence is kept."""
        e = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.95)
        r = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.80)

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert result.agreed_eligible[0].confidence == 0.95

    def test_reviewer_wins_when_higher_confidence(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Reviewer's match is used when it has higher confidence."""
        e = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.70)
        r = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE, confidence=0.95)

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert result.agreed_eligible[0].confidence == 0.95


# -----------------------------------------------------------------------
# Cell 2: Both INELIGIBLE -> agreed_ineligible
# -----------------------------------------------------------------------


class TestBothIneligible:
    """Cell 2: both agents say INELIGIBLE -> agreed_ineligible bucket."""

    def test_agreed_ineligible(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S2",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income_too_high"],
        )
        r = make_scheme_match(
            scheme_id="S2",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income_too_high"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert "S2" in result.agreed_ineligible
        assert len(result.agreed_eligible) == 0
        assert len(result.conservative_eligible) == 0
        assert len(result.disagreements) == 0

    def test_only_scheme_id_stored(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """agreed_ineligible stores scheme_ids as strings, not full matches."""
        e = make_scheme_match(scheme_id="SKIP-ME", verdict=EligibilityVerdict.INELIGIBLE)
        r = make_scheme_match(scheme_id="SKIP-ME", verdict=EligibilityVerdict.INELIGIBLE)

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert result.agreed_ineligible == ["SKIP-ME"]


# -----------------------------------------------------------------------
# Cell 3: Eligibility=ELIGIBLE, Reviewer=INELIGIBLE -> disagreement
# -----------------------------------------------------------------------


class TestEligibleVsIneligible:
    """Cell 3: eligibility says ELIGIBLE, reviewer says INELIGIBLE."""

    def test_disagreement_recorded(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S3",
            verdict=EligibilityVerdict.ELIGIBLE,
            reasoning_trace="Income below threshold, state matches",
        )
        r = make_scheme_match(
            scheme_id="S3",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
            reasoning_trace="Transcript suggests income above 2.5 lakh",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.scheme_id == "S3"
        assert d.eligibility_verdict == EligibilityVerdict.ELIGIBLE
        assert d.reviewer_verdict == EligibilityVerdict.INELIGIBLE

    def test_transcript_evidence_trusts_reviewer(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When transcript has income keywords, reviewer verdict wins."""
        e = make_scheme_match(scheme_id="S3", verdict=EligibilityVerdict.ELIGIBLE)
        r = make_scheme_match(
            scheme_id="S3",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,  # mentions "income" and "daily wage"
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is True
        # Reviewer says INELIGIBLE, so that wins
        assert d.final_verdict == EligibilityVerdict.INELIGIBLE
        assert "trusting reviewer" in d.caveat.lower()

    def test_no_transcript_resolves_uncertain(
        self,
        checker: ConvergenceChecker,
        ctx_empty_transcript: ConversationContext,
    ):
        """Without transcript evidence, disagreement resolves to UNCERTAIN."""
        e = make_scheme_match(scheme_id="S3", verdict=EligibilityVerdict.ELIGIBLE)
        r = make_scheme_match(
            scheme_id="S3",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_empty_transcript,
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is False
        assert d.final_verdict == EligibilityVerdict.UNCERTAIN
        assert "Jan Seva Kendra" in d.caveat

    def test_ineligible_resolved_not_in_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When resolved to INELIGIBLE, scheme does NOT appear in conservative_eligible."""
        e = make_scheme_match(scheme_id="S3", verdict=EligibilityVerdict.ELIGIBLE)
        r = make_scheme_match(
            scheme_id="S3",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.conservative_eligible) == 0
        assert len(result.agreed_eligible) == 0


# -----------------------------------------------------------------------
# Cell 4: Eligibility=INELIGIBLE, Reviewer=ELIGIBLE -> disagreement
# -----------------------------------------------------------------------


class TestIneligibleVsEligible:
    """Cell 4: eligibility says INELIGIBLE, reviewer says ELIGIBLE."""

    def test_disagreement_reversed_direction(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["occupation"],
            reasoning_trace="Occupation type salaried_govt not eligible",
        )
        r = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.ELIGIBLE,
            reasoning_trace="Transcript says daily wage, not govt salaried",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.eligibility_verdict == EligibilityVerdict.INELIGIBLE
        assert d.reviewer_verdict == EligibilityVerdict.ELIGIBLE

    def test_resolved_eligible_goes_to_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When reviewer says ELIGIBLE and transcript confirms, scheme goes conservative."""
        e = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["occupation"],
        )
        r = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.ELIGIBLE,
            reasoning_trace="Transcript says daily wage kaam",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,  # transcript mentions kaam
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is True
        assert d.final_verdict == EligibilityVerdict.ELIGIBLE
        # Resolved-eligible disagreements land in conservative_eligible
        assert any(m.scheme_id == "S4" for m in result.conservative_eligible)

    def test_resolved_eligible_gets_confidence_penalty(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Conservative matches from disagreements get a 0.7x confidence penalty."""
        e = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["occupation"],
        )
        r = make_scheme_match(
            scheme_id="S4",
            verdict=EligibilityVerdict.ELIGIBLE,
            confidence=1.0,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        match = next(m for m in result.conservative_eligible if m.scheme_id == "S4")
        assert match.confidence == pytest.approx(0.7)  # 1.0 * 0.7


# -----------------------------------------------------------------------
# Cell 5: Eligibility=ELIGIBLE, Reviewer=UNCERTAIN -> disagreement
# -----------------------------------------------------------------------


class TestEligibleVsUncertain:
    """Cell 5: eligible + uncertain -> disagreement -> resolution."""

    def test_creates_disagreement(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(scheme_id="S5", verdict=EligibilityVerdict.ELIGIBLE)
        r = make_scheme_match(
            scheme_id="S5",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["age"],
            reasoning_trace="Age unclear from transcript",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.eligibility_verdict == EligibilityVerdict.ELIGIBLE
        assert d.reviewer_verdict == EligibilityVerdict.UNCERTAIN
        assert d.caveat  # always has a caveat

    def test_no_transcript_evidence_resolves_uncertain(
        self,
        checker: ConvergenceChecker,
        ctx_empty_transcript: ConversationContext,
    ):
        """Without transcript evidence, ELIGIBLE vs UNCERTAIN resolves to UNCERTAIN."""
        e = make_scheme_match(scheme_id="S5", verdict=EligibilityVerdict.ELIGIBLE)
        r = make_scheme_match(
            scheme_id="S5",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["age"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_empty_transcript,
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is False
        assert d.final_verdict == EligibilityVerdict.UNCERTAIN


# -----------------------------------------------------------------------
# Cell 6: Eligibility=UNCERTAIN, Reviewer=ELIGIBLE -> disagreement
# -----------------------------------------------------------------------


class TestUncertainVsEligible:
    """Cell 6: uncertain + eligible -> disagreement -> resolution."""

    def test_creates_disagreement(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S6",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["state"],
            reasoning_trace="State not confirmed in profile",
        )
        r = make_scheme_match(
            scheme_id="S6",
            verdict=EligibilityVerdict.ELIGIBLE,
            reasoning_trace="Transcript confirms Rajasthan state",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.eligibility_verdict == EligibilityVerdict.UNCERTAIN
        assert d.reviewer_verdict == EligibilityVerdict.ELIGIBLE

    def test_transcript_evidence_trusts_reviewer_eligible(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """With transcript evidence, reviewer's ELIGIBLE verdict wins -> conservative."""
        e = make_scheme_match(
            scheme_id="S6",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["income"],
        )
        r = make_scheme_match(
            scheme_id="S6",
            verdict=EligibilityVerdict.ELIGIBLE,
            confidence=0.85,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is True
        assert d.final_verdict == EligibilityVerdict.ELIGIBLE
        # Goes to conservative_eligible
        assert any(m.scheme_id == "S6" for m in result.conservative_eligible)


# -----------------------------------------------------------------------
# Cell 7: Eligibility=INELIGIBLE, Reviewer=UNCERTAIN -> disagreement
# -----------------------------------------------------------------------


class TestIneligibleVsUncertain:
    """Cell 7: ineligible + uncertain -> disagreement -> resolution."""

    def test_creates_disagreement(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
            reasoning_trace="Income above 5 lakh",
        )
        r = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.UNCERTAIN,
            reasoning_trace="Income unclear from transcript",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.eligibility_verdict == EligibilityVerdict.INELIGIBLE
        assert d.reviewer_verdict == EligibilityVerdict.UNCERTAIN

    def test_with_transcript_trusts_reviewer_uncertain(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """With transcript evidence, reviewer's UNCERTAIN verdict wins -> not eligible."""
        e = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )
        r = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.UNCERTAIN,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        d = result.disagreements[0]
        assert d.resolved_from_transcript is True
        # Reviewer says UNCERTAIN, so that is the final verdict
        assert d.final_verdict == EligibilityVerdict.UNCERTAIN
        # UNCERTAIN final verdict from disagreement does NOT go to conservative_eligible
        assert "S7" not in [m.scheme_id for m in result.conservative_eligible]

    def test_not_surfaced_as_eligible(
        self,
        checker: ConvergenceChecker,
        ctx_empty_transcript: ConversationContext,
    ):
        """Without evidence, resolved UNCERTAIN -- not in agreed or conservative eligible."""
        e = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )
        r = make_scheme_match(
            scheme_id="S7",
            verdict=EligibilityVerdict.UNCERTAIN,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_empty_transcript,
        )

        assert "S7" not in [m.scheme_id for m in result.agreed_eligible]
        assert "S7" not in [m.scheme_id for m in result.conservative_eligible]


# -----------------------------------------------------------------------
# Cell 8: Eligibility=UNCERTAIN, Reviewer=INELIGIBLE -> disagreement
# -----------------------------------------------------------------------


class TestUncertainVsIneligible:
    """Cell 8: uncertain + ineligible -> disagreement -> resolution."""

    def test_creates_disagreement(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S8",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["bpl_card"],
            reasoning_trace="BPL status unclear",
        )
        r = make_scheme_match(
            scheme_id="S8",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["bpl_card"],
            reasoning_trace="No BPL card mentioned in transcript",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.disagreements) == 1
        d = result.disagreements[0]
        assert d.eligibility_verdict == EligibilityVerdict.UNCERTAIN
        assert d.reviewer_verdict == EligibilityVerdict.INELIGIBLE

    def test_resolved_ineligible_not_in_eligible_lists(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Reviewer says INELIGIBLE, transcript confirms -> INELIGIBLE, not eligible."""
        e = make_scheme_match(
            scheme_id="S8",
            verdict=EligibilityVerdict.UNCERTAIN,
            failed_criteria=["income"],
        )
        r = make_scheme_match(
            scheme_id="S8",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,  # transcript mentions income keywords
        )

        d = result.disagreements[0]
        # Transcript has evidence, trusts reviewer -> INELIGIBLE
        assert d.final_verdict == EligibilityVerdict.INELIGIBLE
        assert "S8" not in [m.scheme_id for m in result.agreed_eligible]
        assert "S8" not in [m.scheme_id for m in result.conservative_eligible]


# -----------------------------------------------------------------------
# Cell 9: Both UNCERTAIN -> conservative_eligible
# -----------------------------------------------------------------------


class TestBothUncertain:
    """Cell 9: both agents say UNCERTAIN -> conservative_eligible bucket."""

    def test_goes_to_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.4,
            reasoning_trace="Income data ambiguous",
        )
        r = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.35,
            reasoning_trace="Transcript unclear about income",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        # Both uncertain -> conservative_eligible, NOT disagreement
        assert len(result.disagreements) == 0
        assert len(result.conservative_eligible) == 1
        match = result.conservative_eligible[0]
        assert match.scheme_id == "S9"
        assert match.verdict == EligibilityVerdict.UNCERTAIN

    def test_uses_minimum_confidence(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When both are uncertain, confidence is the minimum of the two."""
        e = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.4,
        )
        r = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.35,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert result.conservative_eligible[0].confidence == pytest.approx(0.35)

    def test_merges_reasoning_traces(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Merged match includes both agents' reasoning traces."""
        e = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            reasoning_trace="E-trace",
        )
        r = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            reasoning_trace="R-trace",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        trace = result.conservative_eligible[0].reasoning_trace
        assert "E-trace" in trace
        assert "R-trace" in trace
        assert "Both agents uncertain" in trace

    def test_not_in_agreed_eligible_or_ineligible(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        e = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.3,
        )
        r = make_scheme_match(
            scheme_id="S9",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.3,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert len(result.agreed_eligible) == 0
        assert len(result.agreed_ineligible) == 0


# -----------------------------------------------------------------------
# Multi-scheme and edge-case tests
# -----------------------------------------------------------------------


class TestMultiSchemeConvergence:
    """Tests with multiple schemes landing in different buckets."""

    def test_mixed_outcomes_all_buckets(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Three schemes: one agreed-eligible, one agreed-ineligible, one both-uncertain."""
        ea = make_scheme_match(scheme_id="A", verdict=EligibilityVerdict.ELIGIBLE)
        ra = make_scheme_match(scheme_id="A", verdict=EligibilityVerdict.ELIGIBLE)

        eb = make_scheme_match(scheme_id="B", verdict=EligibilityVerdict.INELIGIBLE)
        rb = make_scheme_match(scheme_id="B", verdict=EligibilityVerdict.INELIGIBLE)

        ec = make_scheme_match(scheme_id="C", verdict=EligibilityVerdict.UNCERTAIN, confidence=0.3)
        rc = make_scheme_match(scheme_id="C", verdict=EligibilityVerdict.UNCERTAIN, confidence=0.2)

        result = checker.check(
            make_eligibility_result([ea, eb, ec]),
            make_reviewer_result([ra, rb, rc]),
            ctx_with_income_keywords,
        )

        assert len(result.agreed_eligible) == 1
        assert result.agreed_eligible[0].scheme_id == "A"
        assert "B" in result.agreed_ineligible
        assert len(result.conservative_eligible) == 1
        assert result.conservative_eligible[0].scheme_id == "C"

    def test_all_eligible_combines_agreed_and_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """ConvergenceResult.all_eligible includes both agreed and conservative lists."""
        # S1: both eligible -> agreed
        e1 = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE)
        r1 = make_scheme_match(scheme_id="S1", verdict=EligibilityVerdict.ELIGIBLE)

        # S2: both uncertain -> conservative
        e2 = make_scheme_match(
            scheme_id="S2", verdict=EligibilityVerdict.UNCERTAIN, confidence=0.5
        )
        r2 = make_scheme_match(
            scheme_id="S2", verdict=EligibilityVerdict.UNCERTAIN, confidence=0.4
        )

        result = checker.check(
            make_eligibility_result([e1, e2]),
            make_reviewer_result([r1, r2]),
            ctx_with_income_keywords,
        )

        all_ids = {m.scheme_id for m in result.all_eligible}
        assert "S1" in all_ids
        assert "S2" in all_ids
        assert len(result.all_eligible) == 2


class TestSingleAgentEvaluation:
    """Schemes evaluated by only one agent."""

    def test_single_agent_eligible_goes_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """A scheme evaluated by only one agent (eligible) -> conservative at 80% confidence."""
        e = make_scheme_match(
            scheme_id="SOLO", verdict=EligibilityVerdict.ELIGIBLE, confidence=1.0
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([]),  # reviewer did not evaluate this scheme
            ctx_with_income_keywords,
        )

        assert len(result.agreed_eligible) == 0
        assert len(result.conservative_eligible) == 1
        assert result.conservative_eligible[0].scheme_id == "SOLO"
        assert result.conservative_eligible[0].confidence == pytest.approx(0.8)

    def test_single_agent_ineligible_goes_to_agreed_ineligible(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """A scheme evaluated by only one agent (ineligible) -> agreed_ineligible."""
        r = make_scheme_match(
            scheme_id="SOLO-I",
            verdict=EligibilityVerdict.INELIGIBLE,
        )

        result = checker.check(
            make_eligibility_result([]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        assert "SOLO-I" in result.agreed_ineligible

    def test_single_agent_uncertain_goes_conservative(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """A scheme evaluated by only one agent (uncertain) -> conservative."""
        e = make_scheme_match(
            scheme_id="SOLO-U",
            verdict=EligibilityVerdict.UNCERTAIN,
            confidence=0.5,
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([]),
            ctx_with_income_keywords,
        )

        assert len(result.conservative_eligible) == 1
        assert result.conservative_eligible[0].scheme_id == "SOLO-U"


class TestDisagreementFieldIdentification:
    """Test the divergent field detection heuristic."""

    def test_divergent_field_from_failed_criteria(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """Divergent field is identified from differing failed_criteria."""
        e = make_scheme_match(
            scheme_id="DF1",
            verdict=EligibilityVerdict.ELIGIBLE,
            failed_criteria=[],
        )
        r = make_scheme_match(
            scheme_id="DF1",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["income"],
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        d = result.disagreements[0]
        assert d.disagreement_field == "income"

    def test_divergent_field_falls_back_to_reasoning(
        self,
        checker: ConvergenceChecker,
        ctx_with_income_keywords: ConversationContext,
    ):
        """When failed_criteria match, field is found from reasoning keywords."""
        e = make_scheme_match(
            scheme_id="DF2",
            verdict=EligibilityVerdict.ELIGIBLE,
            failed_criteria=["something"],
            reasoning_trace="User meets age requirement of 70+",
        )
        r = make_scheme_match(
            scheme_id="DF2",
            verdict=EligibilityVerdict.INELIGIBLE,
            failed_criteria=["something"],
            reasoning_trace="User age is only 45, below 70 threshold",
        )

        result = checker.check(
            make_eligibility_result([e]),
            make_reviewer_result([r]),
            ctx_with_income_keywords,
        )

        d = result.disagreements[0]
        # "age" keyword found in reasoning -> field is "age"
        assert d.disagreement_field == "age"
