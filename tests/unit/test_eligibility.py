"""Unit tests for the EligibilityAgent."""

from __future__ import annotations

import asyncio
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
    SchemeMatch,
    SchemeRecord,
)
from vaidya.models.user_profile import (
    CoverageType,
    IncomeCategory,
    OccupationType,
    UserProfile,
)
from vaidya.prompts import registry as prompts_registry

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


# ---------------------------------------------------------------------------
# System-prompt render hoist
# ---------------------------------------------------------------------------


def _extract_schemes_payload(system_prompt: str) -> list[dict[str, Any]]:
    """Recover the serialized schemes list from a prompt.

    Works whether the prompt is the bare schemes JSON (render mocked to echo
    ``schemes``) or the fully rendered template (render wrapped). In the latter
    case the compact JSON array immediately follows the SCHEME DATA marker.
    """
    text = system_prompt.strip()
    marker = "SCHEME DATA (retrieved from knowledge base):"
    if marker in text:
        after = text.split(marker, 1)[1].lstrip("\n")
        text = after.split("\n", 1)[0]
    return json.loads(text)


def _ineligible_match_for_each(system_prompt: str) -> dict[str, Any]:
    """Fake LLM: echo back an ineligible verdict for every scheme in the prompt."""
    payload = _extract_schemes_payload(system_prompt)
    return {
        "matches": [
            {
                "scheme_id": item["scheme_id"],
                "scheme_name": item["canonical_name"],
                "verdict": "ineligible",
                "confidence": 0.8,
                "coverage_summary": "n/a",
            }
            for item in payload
        ]
    }


class TestSystemPromptRenderHoist:
    @pytest.mark.asyncio
    async def test_system_prompt_rendered_once_not_per_batch(self):
        # 46 schemes / batch_size 20 => 3 batches, but a single render call.
        schemes = [_make_scheme(f"s{i}") for i in range(46)]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        # Use the real template render so the placeholder substitution path is
        # exercised, but spy on the render function to count calls.
        with patch(
            "vaidya.agents.eligibility.prompts.render",
            wraps=prompts_registry.render,
        ) as spy_render:
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        # One render for the whole call; three LLM calls (one per batch).
        assert spy_render.call_count == 1
        assert agent._call_llm_json.call_count == 3
        # Every candidate still evaluated -- pruning/hoisting must not drop any.
        assert result.schemes_evaluated == 46
        assert len(result.matches) == 46

    @pytest.mark.asyncio
    async def test_each_batch_receives_its_own_schemes_payload(self):
        # The hoisted template must still splice the correct per-batch schemes.
        schemes = [_make_scheme(f"s{i}") for i in range(5)]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=2,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        seen_ids: list[set[str]] = []

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            payload = _extract_schemes_payload(system_prompt)
            seen_ids.append({item["scheme_id"] for item in payload})
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            wraps=prompts_registry.render,
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        # 5 schemes / batch 2 => batches of {s0,s1}, {s2,s3}, {s4}
        assert seen_ids == [{"s0", "s1"}, {"s2", "s3"}, {"s4"}]
        assert {m.scheme_id for m in result.matches} == {f"s{i}" for i in range(5)}


# ---------------------------------------------------------------------------
# RAG retrieval caching + top-k pruning
# ---------------------------------------------------------------------------


def _mock_store(hits: list[SchemeRecord]) -> MagicMock:
    """A KnowledgeStore stand-in whose search() returns a fixed hit list."""
    store = MagicMock()
    store.search = MagicMock(return_value=list(hits))
    return store


class TestRetrievalCacheAndPruning:
    @pytest.mark.asyncio
    async def test_store_search_not_called_again_on_unchanged_fingerprint(self):
        schemes = [_make_scheme(f"s{i}") for i in range(5)]
        store = _mock_store(schemes[:3])  # retrieval surfaces s0,s1,s2
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            store=store,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            side_effect=lambda _name, **kw: kw["schemes"],
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            # First eligibility turn -- retrieval runs once.
            await agent._evaluate(ctx, "sarvam-105b")
            assert store.search.call_count == 1
            # Second turn on the SAME (unchanged) profile -- cache hit, no search.
            await agent._evaluate(ctx, "sarvam-105b")
            assert store.search.call_count == 1

    @pytest.mark.asyncio
    async def test_store_search_recalled_when_fingerprint_changes(self):
        schemes = [_make_scheme(f"s{i}") for i in range(5)]
        store = _mock_store(schemes[:3])
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            store=store,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None, health_need="heart surgery")

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            side_effect=lambda _name, **kw: kw["schemes"],
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            await agent._evaluate(ctx, "sarvam-105b")
            assert store.search.call_count == 1
            # Mutate an eligibility-relevant field -> fingerprint changes.
            ctx.user_profile.health_need = "kidney dialysis"
            await agent._evaluate(ctx, "sarvam-105b")
            assert store.search.call_count == 2

    @pytest.mark.asyncio
    async def test_rag_prunes_to_topk_hits_not_full_applicable(self):
        # 10 applicable schemes, but retrieval only surfaces 3 -> evaluate 3.
        schemes = [_make_scheme(f"s{i}") for i in range(10)]
        store = _mock_store([schemes[2], schemes[5], schemes[7]])
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            store=store,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            side_effect=lambda _name, **kw: kw["schemes"],
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        # Pruned to the retrieval-ranked top-k, preserving retrieval order.
        assert result.schemes_evaluated == 3
        assert [m.scheme_id for m in result.matches] == ["s2", "s5", "s7"]

    @pytest.mark.asyncio
    async def test_empty_rag_result_falls_back_to_full_applicable(self):
        # Retrieval finds nothing applicable -> evaluate the FULL applicable set
        # so a weak/empty retrieval can never cause false negatives.
        schemes = [_make_scheme(f"s{i}") for i in range(4)]
        store = _mock_store([])
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            store=store,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            side_effect=lambda _name, **kw: kw["schemes"],
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        assert result.schemes_evaluated == 4
        # An empty retrieval must not pin the cache (so a later turn retries).
        assert "eligibility_rag_fingerprint" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_no_store_evaluates_full_applicable_without_search(self):
        # No knowledge store at all -> full applicable set, no pruning, no cache.
        schemes = [_make_scheme(f"s{i}") for i in range(4)]
        agent = EligibilityAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=20,
            max_parallel_batches=1,
        )
        ctx = _make_context(state=None)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _ineligible_match_for_each(system_prompt)

        with patch(
            "vaidya.agents.eligibility.prompts.render",
            side_effect=lambda _name, **kw: kw["schemes"],
        ):
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._evaluate(ctx, "sarvam-105b")

        assert result.schemes_evaluated == 4
        assert "eligibility_rag_fingerprint" not in ctx.metadata

    def test_profile_fingerprint_stable_and_sensitive(self):
        ctx = _make_context(
            state="Maharashtra",
            income=IncomeCategory.BELOW_1L,
            occupation=OccupationType.DAILY_WAGE,
            health_need="heart surgery",
        )
        fp1 = EligibilityAgent._profile_fingerprint(ctx)
        # Stable across calls when nothing changes.
        assert fp1 == EligibilityAgent._profile_fingerprint(ctx)
        # Sensitive to an eligibility-relevant change.
        ctx.user_profile.income_bracket = IncomeCategory.ABOVE_5L
        assert EligibilityAgent._profile_fingerprint(ctx) != fp1


# ---------------------------------------------------------------------------
# Speculative eligibility (kicked off during intake, consumed at PROCESSING)
# ---------------------------------------------------------------------------


def _evaluable_context(
    state: str = "Maharashtra",
    income: IncomeCategory = IncomeCategory.BELOW_1L,
    occupation: OccupationType = OccupationType.DAILY_WAGE,
    coverage: CoverageType = CoverageType.NONE,
    family_size: int = 4,
    call_id: str = "spec-call-001",
) -> ConversationContext:
    """A context whose profile is complete enough for eligibility to run.

    ``required_fields_complete`` must be True for ``start_speculative`` to act,
    so unlike :func:`_make_context` this fills family size + coverage too.
    """
    profile = UserProfile(
        state=state,
        family_size=family_size,
        income_bracket=income,
        occupation_type=occupation,
        existing_coverage=coverage,
        health_need="heart surgery",
    )
    return ConversationContext(
        call_id=call_id,
        phone_number_hash="abc123",
        language="hi-IN",
        phase=ConversationPhase.INTAKE,
        user_profile=profile,
    )


def _income_sensitive_llm(system_prompt: str, profile_income: str) -> dict[str, Any]:
    """Fake LLM verdict that depends on the profile income bracket.

    ELIGIBLE for every prompted scheme when income is ``below_1l``, otherwise
    INELIGIBLE. This lets a test prove that a post-speculation profile change
    yields a *genuinely different* recomputed result -- not a stale reuse.
    """
    payload = _extract_schemes_payload(system_prompt)
    eligible = profile_income == IncomeCategory.BELOW_1L.value
    return {
        "matches": [
            {
                "scheme_id": item["scheme_id"],
                "scheme_name": item["canonical_name"],
                "verdict": "eligible" if eligible else "ineligible",
                "confidence": 0.9,
                "coverage_summary": "n/a",
            }
            for item in payload
        ]
    }


def _patched_render() -> Any:
    """Patch prompt rendering to emit just the schemes payload (as other tests do)."""
    return patch(
        "vaidya.agents.eligibility.prompts.render",
        side_effect=lambda _name, **kw: kw["schemes"],
    )


def _eligible_ids(result: Any) -> list[str]:
    return sorted(m.scheme_id for m in result.matches if m.verdict == EligibilityVerdict.ELIGIBLE)


class _no_runtime_warnings:
    """Context manager that fails if any RuntimeWarning is emitted.

    Used to assert a failed speculative task never surfaces as a
    'coroutine/exception was never retrieved' RuntimeWarning.
    """

    def __enter__(self) -> _no_runtime_warnings:
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error", RuntimeWarning)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cm.__exit__(*exc)


class TestSpeculativeEligibility:
    """Speculative pass is a PURE optimisation: identical-or-discarded."""

    @pytest.mark.asyncio
    async def test_does_not_start_when_profile_incomplete(self):
        """No speculation until the profile is evaluable (all 5 fields known)."""
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[_make_scheme("s0")]
        )
        ctx = _make_context(state="Maharashtra")  # missing family_size + coverage
        assert ctx.user_profile.required_fields_complete is False
        assert agent.start_speculative(ctx) is False
        assert ctx.call_id not in agent._speculative

    @pytest.mark.asyncio
    async def test_unchanged_profile_reuses_speculative_result(self):
        """(a) Unchanged profile: PROCESSING reuses the speculative result and
        the eligible set is byte-identical to the synchronous path."""
        schemes = [_make_scheme(f"s{i}") for i in range(4)]

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _income_sensitive_llm(system_prompt, IncomeCategory.BELOW_1L.value)

        # Reference result via the plain synchronous path (no speculation).
        ref_agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=schemes)
        ref_ctx = _evaluable_context(call_id="ref")
        with _patched_render():
            ref_agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            ref_resp = await ref_agent.process(ref_ctx, "")
        assert ref_resp.metadata["eligibility_speculative_hit"] is False
        reference_eligible = _eligible_ids(ref_resp.eligibility_result)

        # Speculative path: start during intake, then process unchanged.
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=schemes)
        ctx = _evaluable_context(call_id="spec")
        with _patched_render():
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            assert agent.start_speculative(ctx) is True
            await asyncio.sleep(0)  # let the background task make progress
            resp = await agent.process(ctx, "")

        assert resp.metadata["eligibility_speculative_hit"] is True
        assert _eligible_ids(resp.eligibility_result) == reference_eligible
        # Single-use: the entry is consumed, not leaked.
        assert ctx.call_id not in agent._speculative

    @pytest.mark.asyncio
    async def test_idempotent_for_same_fingerprint(self):
        """Calling start_speculative twice on an unchanged profile reuses the
        in-flight task rather than spawning a second one."""
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[_make_scheme("s0")]
        )
        ctx = _evaluable_context()
        with _patched_render():
            agent._call_llm_json = AsyncMock(  # type: ignore[method-assign]
                side_effect=lambda sp, *a, **k: _ineligible_match_for_each(sp)
            )
            assert agent.start_speculative(ctx) is True
            first_task = agent._speculative[ctx.call_id].task
            assert agent.start_speculative(ctx) is True
            assert agent._speculative[ctx.call_id].task is first_task
            agent.cancel_speculative(ctx.call_id)

    @pytest.mark.asyncio
    async def test_profile_change_after_speculation_recomputes_no_stale(self):
        """(b) A profile change after speculation forces a recompute reflecting
        the NEW profile -- the stale speculative result must never leak."""
        schemes = [_make_scheme(f"s{i}") for i in range(3)]

        # The fake LLM reads the CURRENT profile income at call time, so the
        # speculative pass (BELOW_1L) and the post-change recompute (ABOVE_5L)
        # return opposite verdicts.
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=schemes)
        ctx = _evaluable_context(income=IncomeCategory.BELOW_1L)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _income_sensitive_llm(system_prompt, ctx.user_profile.income_bracket.value)

        with _patched_render():
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            assert agent.start_speculative(ctx) is True
            await asyncio.sleep(0)
            # Caller corrects income AFTER speculation began -> fingerprint changes.
            ctx.user_profile.income_bracket = IncomeCategory.ABOVE_5L
            resp = await agent.process(ctx, "")

        # Recomputed (not a hit) and reflects the NEW profile: ABOVE_5L => none eligible.
        assert resp.metadata["eligibility_speculative_hit"] is False
        assert _eligible_ids(resp.eligibility_result) == []
        assert ctx.call_id not in agent._speculative

    @pytest.mark.asyncio
    async def test_coverage_change_invalidates_even_though_rag_fp_unchanged(self):
        """REGRESSION: a change to a verdict-relevant field that the eligibility
        prompt reads but the narrow RAG fingerprint OMITS (here existing_coverage)
        must still invalidate the speculative result -- otherwise a stale verdict
        leaks. The reuse gate uses the full-profile fingerprint, not the RAG one.
        """
        schemes = [_make_scheme(f"s{i}") for i in range(3)]
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=schemes)
        # Start with NO existing coverage.
        ctx = _evaluable_context(coverage=CoverageType.NONE)

        # The narrow RAG fingerprint must NOT change when only coverage changes
        # (sanity: that is exactly why the speculative gate cannot use it).
        narrow_before = EligibilityAgent._profile_fingerprint(ctx)

        # Eligibility fake verdict depends on coverage: eligible only when NONE.
        def coverage_sensitive_llm(system_prompt: str) -> dict[str, Any]:
            payload = _extract_schemes_payload(system_prompt)
            eligible = ctx.user_profile.existing_coverage == CoverageType.NONE
            return {
                "matches": [
                    {
                        "scheme_id": item["scheme_id"],
                        "scheme_name": item["canonical_name"],
                        "verdict": "eligible" if eligible else "ineligible",
                        "confidence": 0.9,
                        "coverage_summary": "n/a",
                    }
                    for item in payload
                ]
            }

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return coverage_sensitive_llm(system_prompt)

        with _patched_render():
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            assert agent.start_speculative(ctx) is True
            await asyncio.sleep(0)
            # Caller now reveals employer coverage AFTER speculation began.
            ctx.user_profile.existing_coverage = CoverageType.EMPLOYER
            resp = await agent.process(ctx, "")

        # The narrow RAG fingerprint is unchanged...
        assert EligibilityAgent._profile_fingerprint(ctx) == narrow_before
        # ...yet the speculative result was correctly DISCARDED and recomputed
        # against the new coverage (=> ineligible), not the stale NONE result.
        assert resp.metadata["eligibility_speculative_hit"] is False
        assert _eligible_ids(resp.eligibility_result) == []
        assert ctx.call_id not in agent._speculative

    def test_eligibility_input_fingerprint_covers_full_profile(self):
        """The reuse fingerprint must change for ANY verdict-relevant field --
        including ones the narrow RAG fingerprint ignores."""
        ctx = _evaluable_context()
        base = EligibilityAgent._eligibility_input_fingerprint(ctx)
        # Fields the narrow RAG fingerprint omits but the prompt uses:
        for mutate in (
            lambda p: setattr(p, "existing_coverage", CoverageType.EMPLOYER),
            lambda p: setattr(p, "age", 70),
            lambda p: setattr(p, "bpl_card", True),
            lambda p: setattr(p, "ration_card", True),
            lambda p: setattr(p, "district", "Pune"),
            lambda p: setattr(p, "secc_category", "D1"),
        ):
            fresh = _evaluable_context()
            mutate(fresh.user_profile)
            assert EligibilityAgent._eligibility_input_fingerprint(fresh) != base

    @pytest.mark.asyncio
    async def test_speculative_failure_falls_back_cleanly(self):
        """(c)+(d) When the background pass fails, the consumer swallows the
        exception, recomputes synchronously, and no unretrieved background-task
        exception escapes."""
        schemes = [_make_scheme("s0"), _make_scheme("s1")]
        agent = EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=schemes)
        ctx = _evaluable_context()

        # Force ONLY the background speculative compute to blow up; the
        # synchronous path (_compute_with_fallback) is left intact.
        original_compute = agent._compute_with_fallback
        blew_up = {"flag": False}

        async def speculative_only_failure(context: ConversationContext):
            # The speculative task deep-copies the context, so its call_id is
            # preserved but its object identity differs from `ctx`.
            if not blew_up["flag"]:
                blew_up["flag"] = True
                raise RuntimeError("boom (speculative pass)")
            return await original_compute(context)

        async def fake_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            return _income_sensitive_llm(system_prompt, IncomeCategory.BELOW_1L.value)

        with _patched_render(), _no_runtime_warnings():
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            agent._compute_with_fallback = speculative_only_failure  # type: ignore[method-assign]
            # _no_runtime_warnings turns any unretrieved-task RuntimeWarning
            # into a hard error, asserting the failed speculation is silent.
            assert agent.start_speculative(ctx) is True
            await asyncio.sleep(0.01)  # let the speculative task fail
            resp = await agent.process(ctx, "")

        # The speculative pass raised; the synchronous recompute then succeeded.
        assert blew_up["flag"] is True
        # Fell back cleanly: no hit, no error, real result, entry consumed.
        assert resp.metadata["eligibility_speculative_hit"] is False
        assert resp.error is None
        assert _eligible_ids(resp.eligibility_result) == ["s0", "s1"]
        assert ctx.call_id not in agent._speculative

    @pytest.mark.asyncio
    async def test_cancel_speculative_stops_inflight_task(self):
        """cancel_speculative() cancels the background task and drops the entry,
        so an abandoned session never leaks a running task."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            started.set()
            await release.wait()  # block until the test releases it
            return _ineligible_match_for_each(system_prompt)

        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[_make_scheme("s0")]
        )
        ctx = _evaluable_context()
        with _patched_render():
            agent._call_llm_json = AsyncMock(side_effect=blocking_llm)  # type: ignore[method-assign]
            assert agent.start_speculative(ctx) is True
            task = agent._speculative[ctx.call_id].task
            await asyncio.wait_for(started.wait(), timeout=1.0)

            agent.cancel_speculative(ctx.call_id)
            assert ctx.call_id not in agent._speculative
            # The task is cancelled; awaiting it raises CancelledError (handled).
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()  # unblock in case cancellation lost the race (no-op otherwise)

    @pytest.mark.asyncio
    async def test_cancel_speculative_is_safe_when_nothing_inflight(self):
        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[_make_scheme("s0")]
        )
        # Should not raise even though no speculation was ever started.
        agent.cancel_speculative("never-started")

    @pytest.mark.asyncio
    async def test_speculative_map_is_bounded(self):
        """Abandoned sessions cannot leak entries forever: the map is pruned to
        a cap, evicting (and cancelling) the oldest entries."""
        import vaidya.agents.eligibility as elig_mod

        agent = EligibilityAgent(
            client=_mock_client(), model="sarvam-105b", schemes=[_make_scheme("s0")]
        )

        started: list[asyncio.Event] = []

        async def blocking_llm(system_prompt: str, *_a: Any, **_k: Any) -> dict[str, Any]:
            ev = asyncio.Event()
            started.append(ev)
            await ev.wait()  # never released -> entries stay "in flight"
            return _ineligible_match_for_each(system_prompt)

        with (
            _patched_render(),
            patch.object(elig_mod, "_MAX_SPECULATIVE_ENTRIES", 5),
        ):
            agent._call_llm_json = AsyncMock(side_effect=blocking_llm)  # type: ignore[method-assign]
            tasks = []
            for i in range(20):
                ctx = _evaluable_context(call_id=f"leaky-{i}")
                assert agent.start_speculative(ctx) is True
                tasks.append(agent._speculative.get(ctx.call_id))

            # Never exceeds the cap despite 20 distinct sessions starting.
            assert len(agent._speculative) <= 5
            # The oldest sessions were evicted...
            assert "leaky-0" not in agent._speculative
            # ...and the newest are retained.
            assert "leaky-19" in agent._speculative

        # Clean up: cancel everything still in flight so no task leaks the test.
        for call_id in list(agent._speculative.keys()):
            agent.cancel_speculative(call_id)
        await asyncio.gather(
            *(t.task for t in (tasks or []) if t is not None), return_exceptions=True
        )


# ---------------------------------------------------------------------------
# Flagship floor (PM-JAY deterministic recall safety net)
# ---------------------------------------------------------------------------


class TestFlagshipFloor:
    """PM-JAY must be proposed eligible whenever the structured profile
    objectively meets its criteria, so the LLM's intermittent recall misses
    never drop the flagship for a clearly-qualifying poor family. The floor
    never overrides a hard exclusion (govt employee, employer cover, WB/DL)."""

    def _agent(self) -> EligibilityAgent:
        return EligibilityAgent(client=_mock_client(), model="sarvam-105b", schemes=[])

    def _pmjay(self) -> SchemeRecord:
        return _make_scheme(
            scheme_id="PMJAY-2024-v3",
            name="Ayushman Bharat PM-JAY",
            geo_restrictions=["WB", "DL"],
        )

    def _profile(self, **kw: Any) -> UserProfile:
        base: dict[str, Any] = dict(
            state="Rajasthan",
            income_bracket=IncomeCategory.BELOW_1L,
            occupation_type=OccupationType.DAILY_WAGE,
            existing_coverage=CoverageType.NONE,
        )
        base.update(kw)
        return UserProfile(**base)

    def test_adds_pmjay_when_missing_and_criteria_met(self):
        agent = self._agent()
        out = agent._apply_flagship_floor([], [self._pmjay()], self._profile())
        pmjay = next((m for m in out if m.scheme_id == "PMJAY-2024-v3"), None)
        assert pmjay is not None
        assert pmjay.verdict == EligibilityVerdict.ELIGIBLE

    def test_upgrades_flaky_ineligible_pmjay_without_duplicating(self):
        agent = self._agent()
        miss = SchemeMatch(
            scheme_id="PMJAY-2024-v3",
            scheme_name="PM-JAY",
            verdict=EligibilityVerdict.INELIGIBLE,
            confidence=0.6,
            reasoning_trace="",
            matched_criteria=[],
            failed_criteria=["secc_category"],
            coverage_summary="",
        )
        out = agent._apply_flagship_floor([miss], [self._pmjay()], self._profile())
        pmjays = [m for m in out if m.scheme_id == "PMJAY-2024-v3"]
        assert len(pmjays) == 1
        assert pmjays[0].verdict == EligibilityVerdict.ELIGIBLE

    def test_does_not_fire_for_government_employee(self):
        agent = self._agent()
        out = agent._apply_flagship_floor(
            [], [self._pmjay()], self._profile(occupation_type=OccupationType.SALARIED_GOVT)
        )
        assert "PMJAY-2024-v3" not in [m.scheme_id for m in out]

    def test_does_not_fire_in_opt_out_state(self):
        agent = self._agent()
        out = agent._apply_flagship_floor([], [self._pmjay()], self._profile(state="West Bengal"))
        assert "PMJAY-2024-v3" not in [m.scheme_id for m in out]

    def test_respects_employer_coverage_hard_exclusion(self):
        agent = self._agent()
        out = agent._apply_flagship_floor(
            [], [self._pmjay()], self._profile(existing_coverage=CoverageType.EMPLOYER)
        )
        assert "PMJAY-2024-v3" not in [m.scheme_id for m in out]

    def test_does_not_fire_above_income_ceiling(self):
        agent = self._agent()
        out = agent._apply_flagship_floor(
            [], [self._pmjay()], self._profile(income_bracket=IncomeCategory.ABOVE_5L)
        )
        assert "PMJAY-2024-v3" not in [m.scheme_id for m in out]
