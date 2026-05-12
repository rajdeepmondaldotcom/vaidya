"""Unit tests for the EligibilityAgent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vaidya.agents.eligibility import EligibilityAgent
from vaidya.agents.scheme_utils import parse_verdict, serialize_for_prompt
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
            "aarogyasri",
            jurisdiction=Jurisdiction.STATE,
            state_code="TS",
            geo_restrictions=["Telangana"],
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[central, state]
        )
        result = agent._filter_schemes("Maharashtra")
        assert central in result
        assert state not in result

    def test_state_scheme_included_when_state_matches(self):
        state_scheme = _make_scheme(
            "aarogyasri",
            jurisdiction=Jurisdiction.STATE,
            state_code="TS",
            geo_restrictions=["Telangana"],
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[state_scheme]
        )
        result = agent._filter_schemes("Telangana")
        assert state_scheme in result

    def test_state_scheme_excluded_when_state_differs(self):
        state_scheme = _make_scheme(
            "aarogyasri",
            jurisdiction=Jurisdiction.STATE,
            state_code="TS",
            geo_restrictions=["Telangana"],
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[state_scheme]
        )
        result = agent._filter_schemes("Kerala")
        assert state_scheme not in result

    def test_empty_geo_restrictions_do_not_make_state_scheme_universal(self):
        scheme = _make_scheme(
            "open",
            jurisdiction=Jurisdiction.STATE,
            state_code="KA",
            geo_restrictions=[],
        )
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[scheme])
        result = agent._filter_schemes("Maharashtra")
        assert scheme not in result

    def test_none_state_returns_all_schemes(self):
        central = _make_scheme("pmjay", jurisdiction=Jurisdiction.CENTRAL)
        state = _make_scheme(
            "aarogyasri",
            jurisdiction=Jurisdiction.STATE,
            state_code="TS",
            geo_restrictions=["Telangana"],
        )
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[central, state]
        )
        result = agent._filter_schemes(None)
        assert len(result) == 2

    def test_unknown_state_returns_all_schemes(self):
        central = _make_scheme("pmjay", jurisdiction=Jurisdiction.CENTRAL)
        state = _make_scheme("aarogyasri", jurisdiction=Jurisdiction.STATE, state_code="TS")
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[central, state]
        )
        result = agent._filter_schemes("Unknown Place")
        assert len(result) == 2

    def test_case_insensitive_state_match(self):
        scheme = _make_scheme(
            "aarogyasri",
            jurisdiction=Jurisdiction.STATE,
            state_code="TS",
            geo_restrictions=["Telangana"],
        )
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[scheme])
        result = agent._filter_schemes("telangana")
        assert scheme in result

    def test_central_scheme_excluded_for_opt_out_state(self):
        pmjay = _make_scheme(
            "pmjay",
            jurisdiction=Jurisdiction.CENTRAL,
            geo_restrictions=["WB", "DL"],
        )
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[pmjay])
        assert pmjay not in agent._filter_schemes("Delhi")


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

    def test_no_default_max_schemes_cap(self):
        schemes = [_make_scheme(f"s{i}") for i in range(30)]
        result = EligibilityAgent._serialize_schemes(schemes)
        assert len(result) == 30

    def test_explicit_max_schemes_cap(self):
        schemes = [_make_scheme(f"s{i}") for i in range(30)]
        result = serialize_for_prompt(schemes, max_schemes=20)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------


class TestBatchedEvaluation:
    @pytest.mark.asyncio
    async def test_evaluates_all_candidates_across_batches(self):
        schemes = [_make_scheme(f"s{i}") for i in range(46)]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            payload = json.loads(system_prompt)
            return {
                "schemes_evaluated": len(payload),
                "matches": [
                    {
                        "scheme_id": item["scheme_id"],
                        "scheme_name": item["canonical_name"],
                        "verdict": "ineligible",
                        "confidence": 0.8,
                        "reasoning_trace": "batch evaluated",
                        "matched_criteria": [],
                        "failed_criteria": ["test"],
                        "coverage_summary": "test coverage",
                    }
                    for item in payload
                ],
            }

        with patch("vaidya.agents.eligibility.prompts.render") as mock_render:
            mock_render.side_effect = lambda _name, **kwargs: kwargs["schemes"]
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        assert result.schemes_evaluated == 46
        assert len(result.matches) == 46
        assert agent._call_llm_json.call_count == 3
        assert ctx.metadata["eligibility_batch_count"] == 3

    @pytest.mark.asyncio
    async def test_retry_fills_missing_scheme_ids(self):
        schemes = [_make_scheme(f"s{i}") for i in range(3)]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=3,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        responses = [
            {"matches": [{"scheme_id": "s0", "verdict": "ineligible", "confidence": 0.8}]},
            {
                "matches": [
                    {"scheme_id": "s1", "verdict": "ineligible", "confidence": 0.8},
                    {"scheme_id": "s2", "verdict": "ineligible", "confidence": 0.8},
                ]
            },
        ]

        with patch("vaidya.agents.eligibility.prompts.render", return_value="[]"):
            agent._call_llm_json = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        assert [m.scheme_id for m in result.matches] == ["s0", "s1", "s2"]
        assert ctx.metadata["eligibility_missing_scheme_ids"] == []

    @pytest.mark.asyncio
    async def test_persistent_missing_scheme_ids_become_uncertain(self):
        schemes = [_make_scheme("s0"), _make_scheme("s1")]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=2,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        with patch("vaidya.agents.eligibility.prompts.render", return_value="[]"):
            agent._call_llm_json = AsyncMock(return_value={"matches": []})  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        assert [m.scheme_id for m in result.matches] == ["s0", "s1"]
        assert all(m.verdict == EligibilityVerdict.UNCERTAIN for m in result.matches)
        assert ctx.metadata["eligibility_missing_scheme_ids"] == ["s0", "s1"]


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
