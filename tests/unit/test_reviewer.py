"""Unit tests for ReviewerAgent batching behavior."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.agents.reviewer import ReviewerAgent
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    EligibilityVerdict,
    FamilyCriteria,
    Jurisdiction,
    SchemeCoverageType,
    SchemeRecord,
)
from vaidya.models.user_profile import UserProfile


def _make_scheme(scheme_id: str = "s0") -> SchemeRecord:
    return SchemeRecord(
        scheme_id=scheme_id,
        canonical_name=f"Scheme {scheme_id}",
        aliases=[],
        local_names={},
        jurisdiction=Jurisdiction.CENTRAL,
        state_code=None,
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
        geographic_restrictions=[],
        coverage_amount_inr=500000,
        coverage_type=SchemeCoverageType.PER_FAMILY_PER_YEAR,
        covered_procedures=[],
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
        description_for_embedding="test scheme",
        keywords=[],
    )


def _context() -> ConversationContext:
    ctx = ConversationContext(
        call_id="test-call-001",
        phone_number_hash="abc123",
        language="hi-IN",
        phase=ConversationPhase.PROCESSING,
        user_profile=UserProfile(state=None),
    )
    ctx.add_turn(
        role="user",
        text="Main Rajasthan se hoon aur daily wage kaam karta hoon.",
        raw_text="Main Rajasthan se hoon aur daily wage kaam karta hoon.",
        language="hi-IN",
    )
    return ctx


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value="")
    client.chat_json = AsyncMock(return_value={})
    client.costs = MagicMock()
    return client


class TestReviewerBatches:
    @pytest.mark.asyncio
    async def test_reviews_all_candidates_across_batches(self) -> None:
        schemes = [_make_scheme(f"s{i}") for i in range(46)]
        agent = ReviewerAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=20,
            max_parallel_batches=1,
        )

        async def fake_llm(system_prompt: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            payload = json.loads(system_prompt)
            return {
                "matches": [
                    {
                        "scheme_id": item["scheme_id"],
                        "verdict": "ineligible",
                        "confidence": 0.8,
                        "reasoning_trace": "reviewed",
                        "matched_criteria": [],
                        "failed_criteria": ["test"],
                        "coverage_summary": "test coverage",
                        "transcript_evidence": [],
                    }
                    for item in payload
                ]
            }

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "vaidya.agents.reviewer.prompts.render",
                lambda _name, **kwargs: kwargs["schemes"],
            )
            agent._call_llm_json = AsyncMock(side_effect=fake_llm)  # type: ignore[method-assign]
            result = await agent._review(_context())

        assert len(result.matches) == 46
        assert agent._call_llm_json.call_count == 3

    @pytest.mark.asyncio
    async def test_persistent_missing_scheme_ids_become_uncertain(self) -> None:
        schemes = [_make_scheme("s0"), _make_scheme("s1")]
        agent = ReviewerAgent(
            client=_mock_client(),
            model="sarvam-105b",
            schemes=schemes,
            batch_size=2,
            max_parallel_batches=1,
        )
        ctx = _context()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "vaidya.agents.reviewer.prompts.render",
                lambda _name, **kwargs: kwargs["schemes"],
            )
            agent._call_llm_json = AsyncMock(return_value={"matches": []})  # type: ignore[method-assign]
            result = await agent._review(ctx)

        assert [m.scheme_id for m in result.matches] == ["s0", "s1"]
        assert all(m.verdict == EligibilityVerdict.UNCERTAIN for m in result.matches)
        assert ctx.metadata["reviewer_missing_scheme_ids"] == ["s0", "s1"]
