"""Tests for Pydantic data model validation and computed properties.

Covers:
- UserProfile: required_fields_complete, missing_fields, defaults, partial fill
- ConversationContext: add_turn, full_transcript_text, updated_at tracking
- ConversationPhase: enum values and string representation
- SchemeMatch: different verdicts
- ConvergenceResult: all_eligible combines agreed + conservative
- EligibilityResult: eligible_schemes computed field
"""

from __future__ import annotations

import pytest

from vaidya.models.api import AgentResponse
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    ConvergenceResult,
    EligibilityResult,
    EligibilityVerdict,
    SchemeMatch,
)
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match(
    scheme_id: str = "T1",
    verdict: EligibilityVerdict = EligibilityVerdict.ELIGIBLE,
    confidence: float = 0.9,
) -> SchemeMatch:
    return SchemeMatch(
        scheme_id=scheme_id,
        scheme_name=f"Test {scheme_id}",
        verdict=verdict,
        confidence=confidence,
        reasoning_trace="test",
        matched_criteria=["income"],
        failed_criteria=[],
        coverage_summary="5L",
    )


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class TestUserProfileComplete:
    """Tests for required_fields_complete computed property."""

    def test_complete_when_all_required_set(self) -> None:
        profile = UserProfile(
            state="RJ",
            family_size=5,
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
        )
        assert profile.required_fields_complete is True

    def test_incomplete_when_fresh_defaults(self) -> None:
        """A brand-new profile with all defaults should be incomplete."""
        profile = UserProfile()
        assert profile.required_fields_complete is False

    def test_incomplete_when_state_missing(self) -> None:
        profile = UserProfile(
            family_size=5,
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
        )
        assert profile.required_fields_complete is False

    def test_incomplete_when_family_size_missing(self) -> None:
        profile = UserProfile(
            state="RJ",
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
        )
        assert profile.required_fields_complete is False

    def test_incomplete_when_income_unknown(self) -> None:
        profile = UserProfile(
            state="RJ",
            family_size=5,
            income_bracket=IncomeCategory.UNKNOWN,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
        )
        assert profile.required_fields_complete is False

    def test_incomplete_when_occupation_unknown(self) -> None:
        profile = UserProfile(
            state="RJ",
            family_size=5,
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.UNKNOWN,
            existing_coverage=CoverageType.NONE,
        )
        assert profile.required_fields_complete is False

    def test_incomplete_when_coverage_unknown(self) -> None:
        profile = UserProfile(
            state="RJ",
            family_size=5,
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.UNKNOWN,
        )
        assert profile.required_fields_complete is False

    def test_optional_fields_dont_affect_completeness(self) -> None:
        """age, district, health_need, bpl_card, etc. are optional."""
        profile = UserProfile(
            state="RJ",
            family_size=5,
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
            # Omitting all optional fields
        )
        assert profile.required_fields_complete is True


class TestUserProfileMissingFields:
    """Tests for missing_fields computed property."""

    def test_empty_profile_has_5_missing(self) -> None:
        profile = UserProfile()
        missing = profile.missing_fields
        assert len(missing) == 5
        expected = {
            "state",
            "family_size",
            "income_bracket",
            "occupation_type",
            "existing_coverage",
        }
        assert set(missing) == expected

    def test_no_missing_when_complete(self) -> None:
        profile = UserProfile(
            state="WB",
            family_size=3,
            income_bracket=IncomeCategory.L1_TO_2_5L,
            occupation_type=OccupationType.FARMER,
            existing_coverage=CoverageType.GOVT_SCHEME,
        )
        assert profile.missing_fields == []

    def test_partial_fill_reports_remaining(self) -> None:
        profile = UserProfile(state="WB", family_size=3)
        missing = profile.missing_fields
        assert "state" not in missing
        assert "family_size" not in missing
        assert "income_bracket" in missing
        assert "occupation_type" in missing
        assert "existing_coverage" in missing

    def test_single_field_set(self) -> None:
        profile = UserProfile(state="MH")
        missing = profile.missing_fields
        assert "state" not in missing
        assert len(missing) == 4


class TestUserProfileDefaults:
    """Tests for default values on UserProfile."""

    def test_default_income_is_unknown(self) -> None:
        profile = UserProfile()
        assert profile.income_bracket == IncomeCategory.UNKNOWN

    def test_default_occupation_is_unknown(self) -> None:
        profile = UserProfile()
        assert profile.occupation_type == OccupationType.UNKNOWN

    def test_default_coverage_is_unknown(self) -> None:
        profile = UserProfile()
        assert profile.existing_coverage == CoverageType.UNKNOWN

    def test_default_state_is_none(self) -> None:
        profile = UserProfile()
        assert profile.state is None

    def test_default_family_size_is_none(self) -> None:
        profile = UserProfile()
        assert profile.family_size is None

    def test_default_confidence_flags_empty(self) -> None:
        profile = UserProfile()
        assert profile.confidence_flags == {}


class TestUserProfileEnums:
    """Verify enum values exist for all categories."""

    def test_income_categories(self) -> None:
        assert len(list(IncomeCategory)) >= 5
        assert IncomeCategory.BELOW_1L.value == "below_1l"
        assert IncomeCategory.ABOVE_5L.value == "above_5l"

    def test_occupation_types(self) -> None:
        assert len(list(OccupationType)) >= 6
        assert OccupationType.DAILY_WAGE.value == "daily_wage"
        assert OccupationType.FARMER.value == "farmer"

    def test_coverage_types(self) -> None:
        assert len(list(CoverageType)) >= 5
        assert CoverageType.NONE.value == "none"
        assert CoverageType.EMPLOYER.value == "employer"


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------


class TestConversationContextAddTurn:
    """Tests for add_turn method."""

    def test_add_single_turn(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        turn = ctx.add_turn("user", "Hello", "Hello", "hi-IN")
        assert len(ctx.transcript) == 1
        assert ctx.transcript[0].role == "user"
        assert ctx.transcript[0].text == "Hello"
        assert turn is ctx.transcript[0]

    def test_add_multiple_turns(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        ctx.add_turn("user", "Q1", "Q1")
        ctx.add_turn("assistant", "A1", "A1")
        ctx.add_turn("user", "Q2", "Q2")
        assert len(ctx.transcript) == 3

    def test_add_turn_updates_updated_at(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        original_time = ctx.updated_at
        ctx.add_turn("user", "Hello", "Hello")
        assert ctx.updated_at >= original_time

    def test_add_turn_uses_context_language_as_default(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        turn = ctx.add_turn("user", "Hello", "Hello")
        assert turn.language == "hi-IN"

    def test_add_turn_with_explicit_language(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        turn = ctx.add_turn("user", "Hello", "Hello", language="en-IN")
        assert turn.language == "en-IN"

    def test_add_turn_with_stt_confidence(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        turn = ctx.add_turn("user", "Hi", "Hi", stt_confidence=0.87)
        assert turn.stt_confidence == pytest.approx(0.87)

    def test_add_turn_preserves_raw_text(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        turn = ctx.add_turn("user", "masked text", "original PII text")
        assert turn.text == "masked text"
        assert turn.raw_text == "original PII text"


class TestConversationContextTranscript:
    """Tests for full_transcript_text property."""

    def test_full_transcript_text(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        ctx.add_turn("user", "Question", "Question")
        ctx.add_turn("assistant", "Answer", "Answer")
        text = ctx.full_transcript_text
        assert "[user] Question" in text
        assert "[assistant] Answer" in text

    def test_empty_transcript_text(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        assert ctx.full_transcript_text == ""

    def test_transcript_text_order(self) -> None:
        ctx = ConversationContext(
            call_id="test",
            phone_number_hash="abc",
            language="hi-IN",
            phase=ConversationPhase.INTAKE,
        )
        ctx.add_turn("user", "First", "First")
        ctx.add_turn("assistant", "Second", "Second")
        ctx.add_turn("user", "Third", "Third")
        text = ctx.full_transcript_text
        lines = text.split("\n")
        assert lines[0] == "[user] First"
        assert lines[1] == "[assistant] Second"
        assert lines[2] == "[user] Third"


# ---------------------------------------------------------------------------
# ConversationPhase enum
# ---------------------------------------------------------------------------


class TestConversationPhase:
    def test_all_7_phases_exist(self) -> None:
        phases = list(ConversationPhase)
        assert len(phases) == 7

    def test_expected_phases_present(self) -> None:
        expected = {
            "welcome",
            "open_elicitation",
            "intake",
            "processing",
            "results",
            "guidance",
            "closure",
        }
        actual = {p.value for p in ConversationPhase}
        assert actual == expected

    def test_phase_is_string_enum(self) -> None:
        assert ConversationPhase.WELCOME.value == "welcome"
        assert ConversationPhase.INTAKE.value == "intake"
        assert ConversationPhase.CLOSURE.value == "closure"

    def test_phase_comparison_with_string(self) -> None:
        """ConversationPhase is a str enum, so it should compare with strings."""
        assert ConversationPhase.WELCOME == "welcome"
        assert ConversationPhase.PROCESSING == "processing"


# ---------------------------------------------------------------------------
# SchemeMatch
# ---------------------------------------------------------------------------


class TestSchemeMatch:
    def test_eligible_verdict(self) -> None:
        m = _match(verdict=EligibilityVerdict.ELIGIBLE)
        assert m.verdict == EligibilityVerdict.ELIGIBLE

    def test_ineligible_verdict(self) -> None:
        m = _match(verdict=EligibilityVerdict.INELIGIBLE)
        assert m.verdict == EligibilityVerdict.INELIGIBLE

    def test_uncertain_verdict(self) -> None:
        m = _match(verdict=EligibilityVerdict.UNCERTAIN)
        assert m.verdict == EligibilityVerdict.UNCERTAIN

    def test_scheme_match_fields(self) -> None:
        m = SchemeMatch(
            scheme_id="X",
            scheme_name="Test X",
            verdict=EligibilityVerdict.ELIGIBLE,
            confidence=0.88,
            reasoning_trace="matches all",
            matched_criteria=["income", "state", "age"],
            failed_criteria=["bpl_card"],
            coverage_summary="Rs 5 lakh",
        )
        assert m.scheme_id == "X"
        assert m.confidence == 0.88
        assert len(m.matched_criteria) == 3
        assert "bpl_card" in m.failed_criteria


# ---------------------------------------------------------------------------
# EligibilityResult
# ---------------------------------------------------------------------------


class TestEligibilityResult:
    def test_eligible_schemes_filter(self) -> None:
        """eligible_schemes should return only ELIGIBLE matches."""
        matches = [
            _match("A", EligibilityVerdict.ELIGIBLE),
            _match("B", EligibilityVerdict.INELIGIBLE),
            _match("C", EligibilityVerdict.UNCERTAIN),
            _match("D", EligibilityVerdict.ELIGIBLE),
        ]
        result = EligibilityResult(
            matches=matches,
            processing_time_ms=100.0,
            model_used="test",
            schemes_evaluated=4,
        )
        eligible = result.eligible_schemes
        assert len(eligible) == 2
        assert {m.scheme_id for m in eligible} == {"A", "D"}

    def test_eligible_schemes_empty(self) -> None:
        result = EligibilityResult(
            matches=[_match("A", EligibilityVerdict.INELIGIBLE)],
            processing_time_ms=50.0,
            model_used="test",
            schemes_evaluated=1,
        )
        assert result.eligible_schemes == []


# ---------------------------------------------------------------------------
# ConvergenceResult
# ---------------------------------------------------------------------------


class TestConvergenceResult:
    def test_all_eligible_combines_agreed_and_conservative(self) -> None:
        agreed = _match("S1", EligibilityVerdict.ELIGIBLE)
        conservative = _match("S2", EligibilityVerdict.UNCERTAIN, 0.5)
        result = ConvergenceResult(
            agreed_eligible=[agreed],
            agreed_ineligible=[],
            disagreements=[],
            conservative_eligible=[conservative],
        )
        all_elig = result.all_eligible
        assert len(all_elig) == 2
        ids = {m.scheme_id for m in all_elig}
        assert "S1" in ids
        assert "S2" in ids

    def test_all_eligible_empty_when_no_matches(self) -> None:
        result = ConvergenceResult(
            agreed_eligible=[],
            agreed_ineligible=["X"],
            disagreements=[],
            conservative_eligible=[],
        )
        assert result.all_eligible == []

    def test_all_eligible_only_agreed(self) -> None:
        agreed = _match("S1", EligibilityVerdict.ELIGIBLE)
        result = ConvergenceResult(
            agreed_eligible=[agreed],
            agreed_ineligible=[],
            disagreements=[],
            conservative_eligible=[],
        )
        assert len(result.all_eligible) == 1
        assert result.all_eligible[0].scheme_id == "S1"

    def test_all_eligible_only_conservative(self) -> None:
        conservative = _match("S2", EligibilityVerdict.UNCERTAIN, 0.5)
        result = ConvergenceResult(
            agreed_eligible=[],
            agreed_ineligible=[],
            disagreements=[],
            conservative_eligible=[conservative],
        )
        assert len(result.all_eligible) == 1
        assert result.all_eligible[0].scheme_id == "S2"

    def test_all_eligible_preserves_order(self) -> None:
        """agreed comes first, then conservative."""
        a1 = _match("A1", EligibilityVerdict.ELIGIBLE)
        a2 = _match("A2", EligibilityVerdict.ELIGIBLE)
        c1 = _match("C1", EligibilityVerdict.UNCERTAIN, 0.5)
        result = ConvergenceResult(
            agreed_eligible=[a1, a2],
            agreed_ineligible=[],
            disagreements=[],
            conservative_eligible=[c1],
        )
        ids = [m.scheme_id for m in result.all_eligible]
        assert ids == ["A1", "A2", "C1"]

    def test_default_factory_creates_empty_lists(self) -> None:
        """Fields use default_factory=list, so missing args yield empty lists."""
        result = ConvergenceResult()
        assert result.agreed_eligible == []
        assert result.agreed_ineligible == []
        assert result.disagreements == []
        assert result.conservative_eligible == []
        assert result.all_eligible == []

    def test_default_factory_independent_instances(self) -> None:
        """Each instance gets its own list — no shared mutable default."""
        r1 = ConvergenceResult()
        r2 = ConvergenceResult()
        r1.agreed_ineligible.append("X")
        assert r2.agreed_ineligible == []


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponse:
    def test_minimal_response(self) -> None:
        resp = AgentResponse(text="Hello")
        assert resp.text == "Hello"
        assert resp.error is None
        assert resp.phase_transition is None

    def test_response_with_phase_transition(self) -> None:
        resp = AgentResponse(
            text="Proceeding to processing",
            phase_transition=ConversationPhase.PROCESSING,
        )
        assert resp.phase_transition == ConversationPhase.PROCESSING

    def test_response_with_error(self) -> None:
        resp = AgentResponse(text="", error="Something went wrong")
        assert resp.error == "Something went wrong"
