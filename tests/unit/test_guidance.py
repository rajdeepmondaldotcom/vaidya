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
        # Combined fallback for one scheme: intro + scheme line + offer.
        assert len(result.spoken_parts) == 3
        assert result.spoken_parts[0].type == "intro"

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

    def test_fallback_parts_single_scheme(self):
        """One scheme: intro + one scheme line + offer, naming the scheme."""
        parts = self._agent()._build_fallback_parts([_match()])
        # intro + 1 scheme line + offer
        assert len(parts) == 3
        types = [p.type for p in parts]
        assert types == ["intro", "scheme", "offer"]
        assert "PM-JAY" in parts[1].text

    def test_fallback_parts_lists_all_schemes_in_one_message(self):
        """Multiple schemes: ONE combined message naming EVERY scheme, no gate."""
        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        m3 = _match("chiranjeevi", "Chiranjeevi Yojana")
        parts = self._agent()._build_fallback_parts([m1, m2, m3])

        # intro + 3 scheme lines + offer
        types = [p.type for p in parts]
        assert types == ["intro", "scheme", "scheme", "scheme", "offer"]

        combined = " ".join(p.text for p in parts)
        # All three scheme names appear in the single combined turn.
        assert "PM-JAY" in combined
        assert "AABY" in combined
        assert "Chiranjeevi Yojana" in combined
        # No "want to hear the next one" gate.
        assert "Ek aur yojana" not in combined
        assert "Sunna chahenge" not in combined

    def test_fallback_sms_within_limit(self):
        sms = self._agent()._build_fallback_sms([_match()])
        assert len(sms) <= 160
        assert "Vaidya" in sms

    def test_fallback_sms_covers_every_scheme(self):
        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        m3 = _match("rsby", "RSBY")
        sms = self._agent()._build_fallback_sms([m1, m2, m3])
        assert "PM-JAY" in sms
        assert "AABY" in sms
        assert "RSBY" in sms
        assert len(sms) <= 160


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
# process() — delivers ALL eligible schemes in one turn
# ---------------------------------------------------------------------------


class TestProcessDeliversAllSchemes:
    @pytest.mark.asyncio
    async def test_process_delivers_all_schemes_one_turn(self):
        """process() delivers results in ONE turn (no per-scheme drip-feed):
        it speaks the relevant schemes deterministically and counts them all."""
        agent = GuidanceAgent(client=_mock_client())

        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        m3 = _match("chiranjeevi", "Chiranjeevi Yojana")
        conv = _convergence(eligible=[m1, m2, m3])
        ctx = _context(convergence=conv)

        resp = await agent.process(ctx, "")

        # One combined turn; every eligible scheme counted.
        assert resp.metadata["schemes_delivered"] == 3
        out = ctx.guidance_output
        assert out is not None
        scheme_lines = [p for p in out.spoken_parts if p.type == "scheme"]
        assert len(scheme_lines) == 3  # all three fit under the spoken cap
        assert out.sms_summary  # SMS carries the full list

    @pytest.mark.asyncio
    async def test_process_caps_spoken_schemes_but_counts_all(self):
        """With many eligible schemes only the top few are spoken aloud, while
        the SMS + count cover them all -- never a multi-minute monologue."""
        agent = GuidanceAgent(client=_mock_client())

        matches = [_match(f"s{i}", f"Scheme {i}", confidence=0.9 - i * 0.01) for i in range(9)]
        conv = _convergence(eligible=matches)
        ctx = _context(convergence=conv)

        resp = await agent.process(ctx, "")

        # All counted, but only the top few are spoken.
        assert resp.metadata["schemes_delivered"] == 9
        out = ctx.guidance_output
        assert out is not None
        scheme_lines = [p for p in out.spoken_parts if p.type == "scheme"]
        assert len(scheme_lines) == 3  # capped to _MAX_SPOKEN_SCHEMES (voice stays ~20s)
        assert out.has_more_schemes is True

    @pytest.mark.asyncio
    async def test_deliver_index_sentinel_is_not_special_cased(self):
        """A stray ``__deliver_scheme_index`` input is treated as normal input.

        It must NOT slice the eligible list -- all schemes still delivered.
        """
        agent = GuidanceAgent(client=_mock_client())

        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("aaby", "AABY")
        conv = _convergence(eligible=[m1, m2])
        ctx = _context(convergence=conv)

        resp = await agent.process(ctx, "__deliver_scheme_index:1")

        assert resp.metadata["schemes_delivered"] == 2

    @pytest.mark.asyncio
    async def test_followup_after_results_gives_next_steps_not_redump(self):
        """A post-results follow-up (phase GUIDANCE, results already delivered)
        must NOT re-read the whole scheme list -- that produced the duplicate
        multi-minute monologue. A bare ack yields the next-step line only."""
        agent = GuidanceAgent(client=_mock_client())
        matches = [_match(f"s{i}", f"Yojana {i}") for i in range(5)]
        ctx = _context(convergence=_convergence(eligible=matches))  # phase=GUIDANCE

        first = await agent.process(ctx, "")  # RESULTS-style first delivery
        assert first.metadata["schemes_delivered"] == 5
        assert ctx.guidance_output is not None

        followup = await agent.process(ctx, "thik hai dhanyavaad")
        assert followup.metadata.get("guidance_followup") == "next_steps"
        assert "schemes_delivered" not in followup.metadata  # not a re-delivery
        assert followup.text == get_msg("guidance", "fallback_action", "hi-IN")

    @pytest.mark.asyncio
    async def test_followup_naming_a_scheme_repeats_only_that_one(self):
        """If the caller names a delivered scheme, repeat ONLY its line + action."""
        agent = GuidanceAgent(client=_mock_client())
        m1 = _match("pmjay", "PM-JAY")
        m2 = _match("chiranjeevi", "Chiranjeevi Yojana")
        ctx = _context(language="en-IN", convergence=_convergence(eligible=[m1, m2]))

        await agent.process(ctx, "")  # first delivery
        followup = await agent.process(ctx, "tell me more about chiranjeevi")
        assert followup.metadata.get("guidance_followup") == "scheme_detail"
        assert "Chiranjeevi" in followup.text
        assert "PM-JAY" not in followup.text  # only the named scheme, not a re-dump

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
