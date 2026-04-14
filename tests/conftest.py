"""Shared fixtures for Vaidya unit and integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    EligibilityResult,
    EligibilityVerdict,
    ReviewerResult,
    SchemeMatch,
)
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
    UserProfile,
)
from vaidya.schemes.registry import get_schemes

# ---------------------------------------------------------------------------
# Sarvam client mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sarvam_client() -> MagicMock:
    """A mock SarvamClient that returns canned JSON responses for chat_json().

    Default canned response simulates a successful intake extraction.
    Override ``client.chat_json.return_value`` in individual tests as needed.
    """
    client = MagicMock()
    client.chat = AsyncMock(return_value="Hello, how can I help you?")
    client.chat_json = AsyncMock(
        return_value={
            "state": "RJ",
            "family_size": 5,
            "income_bracket": "below_1l",
            "occupation_type": "daily_wage",
            "existing_coverage": "none",
            "health_need": "heart surgery",
            "confidence": 0.9,
        },
    )
    client.translate = AsyncMock(side_effect=lambda text, src, tgt: text)
    client.tts = AsyncMock(return_value=b"\x00\x01\x02")
    return client


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_user_profile() -> UserProfile:
    """A fully-filled UserProfile for a BPL daily-wage family in Rajasthan."""
    return UserProfile(
        state="RJ",
        district="Jaipur",
        family_size=5,
        income_bracket=IncomeCategory.BELOW_1L,
        occupation_type=OccupationType.DAILY_WAGE,
        existing_coverage=CoverageType.NONE,
        health_need="heart surgery",
        age=45,
        bpl_card=True,
        ration_card=True,
    )


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_context(sample_user_profile: UserProfile) -> ConversationContext:
    """A ConversationContext in INTAKE phase with the sample profile attached."""
    ctx = ConversationContext(
        call_id="test-call-001",
        phone_number_hash="hash_9876543210",
        language="hi-IN",
        phase=ConversationPhase.INTAKE,
        user_profile=sample_user_profile,
    )
    ctx.add_turn(
        role="assistant",
        text="Namaste! Aap kahan se bol rahe hain?",
        raw_text="Namaste! Aap kahan se bol rahe hain?",
        language="hi-IN",
    )
    ctx.add_turn(
        role="user",
        text="Main Jaipur, Rajasthan se hoon. Meri family mein 5 log hain.",
        raw_text="Main Jaipur, Rajasthan se hoon. Meri family mein 5 log hain.",
        language="hi-IN",
        stt_confidence=0.92,
    )
    return ctx


# ---------------------------------------------------------------------------
# Schemes (loaded from actual JSON data)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_schemes():
    """All scheme records loaded from src/vaidya/schemes/data/*.json."""
    return get_schemes()


# ---------------------------------------------------------------------------
# Helpers for building synthetic SchemeMatch / EligibilityResult objects
# ---------------------------------------------------------------------------


def make_scheme_match(
    scheme_id: str = "TEST-001",
    scheme_name: str = "Test Scheme",
    verdict: EligibilityVerdict = EligibilityVerdict.ELIGIBLE,
    confidence: float = 0.9,
    reasoning_trace: str = "Matches income and state criteria",
    matched_criteria: list[str] | None = None,
    failed_criteria: list[str] | None = None,
    coverage_summary: str = "Rs 5 lakh per family per year",
) -> SchemeMatch:
    """Build a SchemeMatch with sensible defaults for tests."""
    return SchemeMatch(
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        verdict=verdict,
        confidence=confidence,
        reasoning_trace=reasoning_trace,
        matched_criteria=matched_criteria if matched_criteria is not None else ["income", "state"],
        failed_criteria=failed_criteria if failed_criteria is not None else [],
        coverage_summary=coverage_summary,
    )


def make_eligibility_result(
    matches: list[SchemeMatch] | None = None,
) -> EligibilityResult:
    """Build an EligibilityResult with sensible defaults."""
    matches = matches or []
    return EligibilityResult(
        matches=matches,
        processing_time_ms=150.0,
        model_used="saaras:v2",
        schemes_evaluated=len(matches),
    )


def make_reviewer_result(
    matches: list[SchemeMatch] | None = None,
) -> ReviewerResult:
    """Build a ReviewerResult with sensible defaults."""
    matches = matches or []
    return ReviewerResult(
        matches=matches,
        processing_time_ms=120.0,
        model_used="saaras:v2",
        transcript_evidence=["User mentioned daily wage income below 1 lakh"],
    )
