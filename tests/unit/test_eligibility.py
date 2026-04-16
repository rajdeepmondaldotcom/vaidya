"""Unit tests for the EligibilityAgent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaidya.agents.eligibility import EligibilityAgent
from vaidya.agents.scheme_utils import parse_verdict
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    EligibilityVerdict,
    FamilyCriteria,
    Jurisdiction,
    SchemeCoverageType,
    SchemeRecord,
)
from vaidya.models.user_profile import IncomeCategory, OccupationType, UserProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheme(
    scheme_id: str = "pmjay",
    name: str = "PM-JAY",
    jurisdiction: Jurisdiction = Jurisdiction.CENTRAL,
    state_code: str | None = None,
    geo_restrictions: list[str] | None = None,
) -> SchemeRecord:
    return SchemeRecord(
        scheme_id=scheme_id,
        canonical_name=name,
        aliases=[],
        local_names={},
        jurisdiction=jurisdiction,
        state_code=state_code,
        income_thresholds=[],
        secc_categories=[],
        occupation_included=[],
        occupation_excluded=[],
        exclusion_rules=[],
        family_criteria=FamilyCriteria(
            max_family_size=None,
            family_definition="household",
            head_of_family_required=False,
        ),
        geographic_restrictions=geo_restrictions or [],
        coverage_amount_inr=500000,
        coverage_type=SchemeCoverageType.PER_FAMILY_PER_YEAR,
        covered_procedures=["surgery"],
        excluded_procedures=[],
        required_documents=[],
        enrollment_channels=[],
        enrollment_steps=[],
        processing_time_days=30,
        version="1.0",
        effective_date="2024-01-01",
        last_verified="2024-01-01",
        source_url="https://example.com",
        confidence_level="verified",
        description_for_embedding="test",
        keywords=[],
    )


def _make_context(
    state: str | None = "Maharashtra",
    income: IncomeCategory = IncomeCategory.BELOW_1L,
    occupation: OccupationType = OccupationType.DAILY_WAGE,
    health_need: str | None = "heart surgery",
) -> ConversationContext:
    profile = UserProfile(
        state=state,
        income_bracket=income,
        occupation_type=occupation,
        health_need=health_need,
    )
    return ConversationContext(
        call_id="test-call-001",
        phone_number_hash="abc123",
        language="hi-IN",
        phase=ConversationPhase.PROCESSING,
        user_profile=profile,
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value="")
    client.chat_json = AsyncMock(return_value={})
    client.costs = MagicMock()
    return client


# ---------------------------------------------------------------------------
# _filter_schemes
# ---------------------------------------------------------------------------


class TestFilterSchemes:
    def test_central_schemes_always_included(self):
        central = _make_scheme("pmjay", jurisdiction=Jurisdiction.CENTRAL)
        state = _make_scheme(
            "aarogyasri", jurisdiction=Jurisdiction.STATE, geo_restrictions=["Telangana"]
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[central, state]
        )
        result = agent._filter_schemes("Maharashtra")
        assert central in result
        assert state not in result

    def test_state_scheme_included_when_state_matches(self):
        state_scheme = _make_scheme(
            "aarogyasri", jurisdiction=Jurisdiction.STATE, geo_restrictions=["Telangana"]
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[state_scheme]
        )
        result = agent._filter_schemes("Telangana")
        assert state_scheme in result

    def test_state_scheme_excluded_when_state_differs(self):
        state_scheme = _make_scheme(
            "aarogyasri", jurisdiction=Jurisdiction.STATE, geo_restrictions=["Telangana"]
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[state_scheme]
        )
        result = agent._filter_schemes("Kerala")
        assert state_scheme not in result

    def test_no_geo_restrictions_always_included(self):
        scheme = _make_scheme("open", jurisdiction=Jurisdiction.STATE, geo_restrictions=[])
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[scheme])
        result = agent._filter_schemes("AnyState")
        assert scheme in result

    def test_none_state_returns_all_schemes(self):
        central = _make_scheme("pmjay", jurisdiction=Jurisdiction.CENTRAL)
        state = _make_scheme(
            "aarogyasri", jurisdiction=Jurisdiction.STATE, geo_restrictions=["Telangana"]
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[central, state]
        )
        result = agent._filter_schemes(None)
        assert len(result) == 2

    def test_case_insensitive_state_match(self):
        scheme = _make_scheme(
            "aarogyasri", jurisdiction=Jurisdiction.STATE, geo_restrictions=["Telangana"]
        )
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[scheme])
        result = agent._filter_schemes("telangana")
        assert scheme in result


# ---------------------------------------------------------------------------
# _serialize_schemes
# ---------------------------------------------------------------------------


class TestSerializeSchemes:
    def test_output_contains_required_keys(self):
        scheme = _make_scheme()
        result = EligibilityAgent._serialize_schemes([scheme])
        assert len(result) == 1
        entry = result[0]
        for key in ("scheme_id", "canonical_name", "jurisdiction", "coverage_amount_inr"):
            assert key in entry

    def test_truncates_covered_procedures(self):
        scheme = _make_scheme()
        scheme.covered_procedures = [f"proc_{i}" for i in range(25)]
        result = EligibilityAgent._serialize_schemes([scheme])
        assert len(result[0]["covered_procedures"]) <= 10

    def test_max_schemes_cap(self):
        schemes = [_make_scheme(f"s{i}") for i in range(30)]
        result = EligibilityAgent._serialize_schemes(schemes)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# _parse_result
# ---------------------------------------------------------------------------


class TestParseResult:
    def _agent(self) -> EligibilityAgent:
        return EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[])

    def test_valid_json_with_matches(self):
        raw: dict[str, Any] = {
            "matches": [
                {
                    "scheme_id": "pmjay",
                    "scheme_name": "PM-JAY",
                    "verdict": "eligible",
                    "confidence": 0.95,
                    "reasoning_trace": "Income below threshold",
                    "matched_criteria": ["income"],
                    "failed_criteria": [],
                    "coverage_summary": "5 lakh per family",
                }
            ]
        }
        result = self._agent()._parse_result(raw, "sarvam-105b", 1)
        assert len(result.matches) == 1
        assert result.matches[0].verdict == EligibilityVerdict.ELIGIBLE
        assert result.matches[0].confidence == 0.95

    def test_malformed_json_returns_empty(self):
        raw: dict[str, Any] = {"_parse_error": True, "_raw": "garbage"}
        result = self._agent()._parse_result(raw, "sarvam-105b", 5)
        assert result.matches == []
        assert result.schemes_evaluated == 5

    def test_empty_response(self):
        result = self._agent()._parse_result({}, "sarvam-105b", 3)
        assert result.matches == []

    def test_list_response_wrapped(self):
        raw = [
            {
                "scheme_id": "pmjay",
                "verdict": "eligible",
                "confidence": 0.9,
            }
        ]
        result = self._agent()._parse_result(raw, "sarvam-105b", 1)
        assert len(result.matches) == 1
        assert result.matches[0].scheme_id == "pmjay"

    def test_malformed_match_item_skipped(self):
        raw: dict[str, Any] = {
            "matches": [
                {"scheme_id": "good", "verdict": "eligible", "confidence": 0.9},
                "not a dict",
            ]
        }
        result = self._agent()._parse_result(raw, "sarvam-105b", 2)
        assert len(result.matches) == 1


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_eligible(self):
        assert parse_verdict("eligible") == EligibilityVerdict.ELIGIBLE

    def test_ineligible(self):
        assert parse_verdict("ineligible") == EligibilityVerdict.INELIGIBLE

    def test_uncertain(self):
        assert parse_verdict("uncertain") == EligibilityVerdict.UNCERTAIN

    def test_unknown_string_defaults_to_uncertain(self):
        assert parse_verdict("maybe") == EligibilityVerdict.UNCERTAIN

    def test_case_insensitive(self):
        assert parse_verdict("ELIGIBLE") == EligibilityVerdict.ELIGIBLE
        assert parse_verdict("  Ineligible  ") == EligibilityVerdict.INELIGIBLE


# ---------------------------------------------------------------------------
# _build_retrieval_query
# ---------------------------------------------------------------------------


class TestBuildRetrievalQuery:
    def test_full_profile(self):
        ctx = _make_context(
            state="Maharashtra",
            income=IncomeCategory.BELOW_1L,
            occupation=OccupationType.DAILY_WAGE,
            health_need="heart surgery",
        )
        query = EligibilityAgent._build_retrieval_query(ctx)
        assert "Maharashtra" in query
        assert "heart surgery" in query
        assert "daily_wage" in query
        assert "below_1l" in query

    def test_empty_profile(self):
        ctx = _make_context(
            state=None,
            income=IncomeCategory.UNKNOWN,
            occupation=OccupationType.UNKNOWN,
            health_need=None,
        )
        query = EligibilityAgent._build_retrieval_query(ctx)
        assert query == "government healthcare scheme India"

    def test_partial_profile(self):
        ctx = _make_context(
            state="Tamil Nadu",
            income=IncomeCategory.UNKNOWN,
            occupation=OccupationType.UNKNOWN,
            health_need=None,
        )
        query = EligibilityAgent._build_retrieval_query(ctx)
        assert "Tamil Nadu" in query
        assert "unknown" not in query.lower() or "unknown" not in query


# ---------------------------------------------------------------------------
# process() with mocked LLM
# ---------------------------------------------------------------------------


class TestProcess:
    @pytest.mark.asyncio
    async def test_process_returns_agent_response(self):
        client = _mock_client()
        client.chat_json = AsyncMock(
            return_value={
                "matches": [
                    {
                        "scheme_id": "pmjay",
                        "scheme_name": "PM-JAY",
                        "verdict": "eligible",
                        "confidence": 0.9,
                        "reasoning_trace": "ok",
                        "matched_criteria": ["income"],
                        "failed_criteria": [],
                        "coverage_summary": "5L",
                    }
                ]
            }
        )
        scheme = _make_scheme()
        agent = EligibilityAgent(client=client, model="sarvam-105b", schemes=[scheme])
        ctx = _make_context()

        with patch("vaidya.agents.eligibility.prompts") as mock_prompts:
            mock_prompts.render.return_value = "system prompt"
            response = await agent.process(ctx, "")

        assert response.eligibility_result is not None
        assert len(response.eligibility_result.matches) == 1
