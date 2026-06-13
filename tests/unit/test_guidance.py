"""Unit tests for the GuidanceAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vaidya.agents.guidance import GuidanceAgent
from vaidya.i18n import get_msg
from vaidya.models.conversation import ConversationContext, ConversationPhase
from vaidya.models.scheme import (
    ConvergenceResult,
    DisagreementRecord,
    EligibilityVerdict,
    GuidanceOutput,
    SchemeMatch,
)
from vaidya.models.user_profile import UserProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match(
    scheme_id: str = "pmjay",
    name: str = "PM-JAY",
    confidence: float = 0.95,
    coverage: str = "5 lakh per family per year",
) -> SchemeMatch:
    return SchemeMatch(
        scheme_id=scheme_id,
        scheme_name=name,
        verdict=EligibilityVerdict.ELIGIBLE,
        confidence=confidence,
        reasoning_trace="all criteria met",
        matched_criteria=["income", "state"],
        failed_criteria=[],
        coverage_summary=coverage,
    )


def _convergence(
    eligible: list[SchemeMatch] | None = None,
    disagreements: list[DisagreementRecord] | None = None,
    conservative: list[SchemeMatch] | None = None,
) -> ConvergenceResult:
    return ConvergenceResult(
        agreed_eligible=eligible or [],
        agreed_ineligible=[],
        disagreements=disagreements or [],
        conservative_eligible=conservative or [],
    )


def _context(
    language: str = "hi-IN",
    convergence: ConvergenceResult | None = None,
) -> ConversationContext:
    return ConversationContext(
        call_id="test-call-001",
        phone_number_hash="abc123",
        language=language,
        phase=ConversationPhase.GUIDANCE,
        user_profile=UserProfile(state="Maharashtra"),
        convergence_result=convergence,
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.chat = AsyncMock(return_value="")
    client.chat_json = AsyncMock(return_value={})
    client.costs = MagicMock()
    return client


# ---------------------------------------------------------------------------
# _format_schemes_for_prompt
# ---------------------------------------------------------------------------


class TestFormatSchemesForPrompt:
    def test_single_high_confidence_scheme(self):
        agent = GuidanceAgent(client=_mock_client())
        m = _match(confidence=0.95)
        result = agent._format_schemes_for_prompt([m])
        assert "1. PM-JAY" in result
        assert "95%" in result
        assert "NEEDS VERIFICATION" not in result

    def test_low_confidence_gets_verification_tag(self):
        agent = GuidanceAgent(client=_mock_client())
        m = _match(confidence=0.5)
        result = agent._format_schemes_for_prompt([m])
        assert "[NEEDS VERIFICATION]" in result

    def test_multiple_schemes_numbered(self):
        agent = GuidanceAgent(client=_mock_client())
        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aarogyasri", "Aarogyasri", confidence=0.8)
        result = agent._format_schemes_for_prompt([m1, m2])
        assert "1. PM-JAY" in result
        assert "2. Aarogyasri" in result

    def test_includes_name_and_coverage_not_internal_tokens(self):
        agent = GuidanceAgent(client=_mock_client())
        m = _match()
        result = agent._format_schemes_for_prompt([m])
        assert m.scheme_name in result
        assert "Coverage:" in result
        # Internal id + raw field names are excluded so the model can't echo
        # them into speech ("underscore underscore" after translation).
        assert "ID:" not in result
        assert "Matched:" not in result


# ---------------------------------------------------------------------------
# _parse_guidance_output
# ---------------------------------------------------------------------------


class TestParseGuidanceOutput:
    def _agent(self) -> GuidanceAgent:
        return GuidanceAgent(client=_mock_client())

    def test_valid_json(self):
        raw = {
            "spoken_parts": [
                {"type": "headline", "text": "Good news!"},
                {"type": "benefit", "text": "5 lakh coverage"},
                {"type": "action", "text": "Visit Jan Seva Kendra"},
            ],
            "sms_summary": "Eligible for PM-JAY. Visit JSK.",
            "has_more_schemes": False,
            "caveat_needed": False,
        }
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.spoken_parts) == 3
        assert result.sms_summary == "Eligible for PM-JAY. Visit JSK."
        assert result.has_more_schemes is False

    def test_missing_spoken_parts_uses_fallback(self):
        raw = {"sms_summary": "test"}
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.spoken_parts) == 3
        assert result.spoken_parts[0].type == "headline"

    def test_empty_spoken_parts_uses_fallback(self):
        raw = {"spoken_parts": []}
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.spoken_parts) == 3

    def test_parse_error_uses_fallback(self):
        raw = {"_parse_error": True}
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.spoken_parts) == 3
        assert "PM-JAY" in result.spoken_parts[0].text

    def test_sms_truncated_to_160_chars(self):
        raw = {
            "spoken_parts": [{"type": "headline", "text": "hi"}],
            "sms_summary": "x" * 200,
        }
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.sms_summary) <= 160
        assert result.sms_summary.endswith("...")

    def test_invalid_part_dict_filtered(self):
        raw = {
            "spoken_parts": [
                {"type": "headline", "text": "Good"},
                {"missing_type": True},
                {"type": "action"},
            ],
        }
        eligible = [_match()]
        convergence = _convergence(eligible=eligible)
        result = self._agent()._parse_guidance_output(raw, eligible, convergence)
        assert len(result.spoken_parts) == 1
        assert result.spoken_parts[0].text == "Good"


# ---------------------------------------------------------------------------
# _build_caveats
# ---------------------------------------------------------------------------


class TestBuildCaveats:
    def _agent(self) -> GuidanceAgent:
        return GuidanceAgent(client=_mock_client())

    def test_no_caveats(self):
        convergence = _convergence()
        result = self._agent()._build_caveats(convergence)
        assert result == "None"

    def test_disagreement_with_uncertain_verdict(self):
        d = DisagreementRecord(
            scheme_id="pmjay",
            scheme_name="PM-JAY",
            eligibility_verdict=EligibilityVerdict.ELIGIBLE,
            reviewer_verdict=EligibilityVerdict.INELIGIBLE,
            eligibility_reasoning="ok",
            reviewer_reasoning="missing doc",
            disagreement_field="income",
            resolved_from_transcript=False,
            final_verdict=EligibilityVerdict.UNCERTAIN,
            caveat="verify income",
        )
        convergence = _convergence(disagreements=[d])
        result = self._agent()._build_caveats(convergence)
        assert "PM-JAY" in result
        assert "income" in result
        assert "Jan Seva Kendra" in result

    def test_conservative_matches(self):
        m = _match("aaby", "AABY")
        convergence = _convergence(conservative=[m])
        result = self._agent()._build_caveats(convergence)
        assert "AABY" in result
        assert "could not be fully verified" in result

    def test_disagreement_with_non_uncertain_not_shown(self):
        d = DisagreementRecord(
            scheme_id="pmjay",
            scheme_name="PM-JAY",
            eligibility_verdict=EligibilityVerdict.ELIGIBLE,
            reviewer_verdict=EligibilityVerdict.INELIGIBLE,
            eligibility_reasoning="ok",
            reviewer_reasoning="missing doc",
            disagreement_field="income",
            resolved_from_transcript=True,
            final_verdict=EligibilityVerdict.ELIGIBLE,
            caveat="",
        )
        convergence = _convergence(disagreements=[d])
        result = self._agent()._build_caveats(convergence)
        assert "PM-JAY" not in result


# ---------------------------------------------------------------------------
# _build_fallback_parts / _build_fallback_sms
# ---------------------------------------------------------------------------


class TestFallbacks:
    def _agent(self) -> GuidanceAgent:
        return GuidanceAgent(client=_mock_client())

    def test_fallback_parts_structure(self):
        parts = self._agent()._build_fallback_parts([_match()])
        assert len(parts) == 3
        types = [p.type for p in parts]
        assert types == ["headline", "benefit", "action"]
        assert "PM-JAY" in parts[0].text

    def test_fallback_sms_within_limit(self):
        sms = self._agent()._build_fallback_sms([_match()])
        assert len(sms) <= 160
        assert "Vaidya" in sms

    def test_fallback_sms_with_multiple_schemes(self):
        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        sms = self._agent()._build_fallback_sms([m1, m2])
        assert "PM-JAY" in sms
        assert "AABY" in sms


# ---------------------------------------------------------------------------
# _no_match_response
# ---------------------------------------------------------------------------


class TestNoMatchResponse:
    def _agent(self) -> GuidanceAgent:
        return GuidanceAgent(client=_mock_client())

    def test_hindi(self):
        resp = self._agent()._no_match_response("hi-IN")
        assert resp.text == get_msg("guidance", "no_match", "hi-IN")
        assert resp.guidance_output is not None
        assert resp.guidance_output.has_more_schemes is False

    def test_tamil(self):
        resp = self._agent()._no_match_response("ta-IN")
        assert resp.text == get_msg("guidance", "no_match", "ta-IN")

    def test_bengali(self):
        resp = self._agent()._no_match_response("bn-IN")
        assert resp.text == get_msg("guidance", "no_match", "bn-IN")

    def test_english(self):
        resp = self._agent()._no_match_response("en-IN")
        assert resp.text == get_msg("guidance", "no_match", "en-IN")

    def test_unknown_language_falls_back_to_hindi(self):
        resp = self._agent()._no_match_response("xx-XX")
        assert resp.text == get_msg("guidance", "no_match", "hi-IN")

    def test_kannada_has_own_translation(self):
        resp = self._agent()._no_match_response("kn-IN")
        assert resp.text == get_msg("guidance", "no_match", "kn-IN")


# ---------------------------------------------------------------------------
# process() — __deliver_scheme_index parsing
# ---------------------------------------------------------------------------


class TestProcessSchemeIndex:
    @pytest.mark.asyncio
    async def test_deliver_single_scheme_by_index(self):
        """Verify the index-parsing logic slices the eligible list correctly."""
        agent = GuidanceAgent(client=_mock_client())

        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        conv = _convergence(eligible=[m1, m2])
        ctx = _context(convergence=conv)

        captured_eligible: list[list[SchemeMatch]] = []

        async def fake_generate(eligible, convergence, context):
            captured_eligible.append(list(eligible))
            return GuidanceOutput(
                spoken_parts=[],
                sms_summary="test",
                has_more_schemes=False,
                caveat_needed=False,
                processing_time_ms=0.0,
            )

        agent._generate_guidance = fake_generate
        await agent.process(ctx, "__deliver_scheme_index:1")

        assert len(captured_eligible) == 1
        assert len(captured_eligible[0]) == 1
        assert captured_eligible[0][0].scheme_id == "aaby"

    @pytest.mark.asyncio
    async def test_invalid_index_delivers_all(self):
        """Bad index string should fall through and deliver all schemes."""
        agent = GuidanceAgent(client=_mock_client())

        m1 = _match("pmjay", "PM-JAY")
        conv = _convergence(eligible=[m1])
        ctx = _context(convergence=conv)

        captured: list[list[SchemeMatch]] = []

        async def fake_generate(eligible, convergence, context):
            captured.append(list(eligible))
            return GuidanceOutput(
                spoken_parts=[],
                sms_summary="test",
                has_more_schemes=False,
                caveat_needed=False,
                processing_time_ms=0.0,
            )

        agent._generate_guidance = fake_generate
        await agent.process(ctx, "__deliver_scheme_index:not_a_number")

        assert len(captured) == 1
        assert len(captured[0]) == 1

    @pytest.mark.asyncio
    async def test_no_convergence_returns_no_match(self):
        agent = GuidanceAgent(client=_mock_client())
        ctx = _context(convergence=None)
        resp = await agent.process(ctx, "")
        assert resp.text == get_msg("guidance", "no_match", "hi-IN")

    @pytest.mark.asyncio
    async def test_empty_eligible_returns_no_match(self):
        agent = GuidanceAgent(client=_mock_client())
        conv = _convergence(eligible=[], conservative=[])
        ctx = _context(convergence=conv)
        resp = await agent.process(ctx, "")
        assert resp.text == get_msg("guidance", "no_match", "hi-IN")
