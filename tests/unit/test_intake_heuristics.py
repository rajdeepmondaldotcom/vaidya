"""Deterministic intake extraction fallback tests."""

from __future__ import annotations

import pytest

from vaidya.agents.intake import IntakeAgent


class TestIntakeHeuristics:
    def test_extracts_state_family_income_occupation_and_coverage(self):
        agent = IntakeAgent(client=object(), model="mock")

        merged = agent._merge_heuristic_fields(
            {"extracted_fields": {}, "field_confidence": {}},
            "Main Rajasthan mein rehta hoon. Family mein 5 log hain. "
            "Daily wage construction kaam, mahine ka 6-7 hazaar. "
            "Koi health insurance nahi hai, BPL card aur ration card hai.",
            0,
        )

        fields = merged["extracted_fields"]
        assert fields["state"] == "Rajasthan"
        assert fields["family_size"] == 5
        assert fields["occupation_type"] == "daily_wage"
        assert fields["income_bracket"] == "below_1l"
        assert fields["existing_coverage"] == "none"
        assert fields["bpl_card"] is True
        assert fields["ration_card"] is True

    def test_extracts_tax_payer_as_above_five_lakh(self):
        agent = IntakeAgent(client=object(), model="mock")

        merged = agent._merge_heuristic_fields(
            {"extracted_fields": {}},
            "Salary 50 hazaar hai, income tax bharta hoon.",
            3,
        )

        assert merged["extracted_fields"]["occupation_type"] == "salaried_pvt"
        assert merged["extracted_fields"]["income_bracket"] == "above_5l"

    def test_extracts_annual_lakh_amounts(self):
        """ "X lakh" is how most callers state annual income — must map
        deterministically so a boundary answer never triggers a re-ask."""
        cases = [
            ("I do farming work, our yearly income is about two lakh rupees", "1l_to_2.5l"),
            ("Hum kheti karte hain, saal ka 2 lakh kamate hain", "1l_to_2.5l"),
            ("3.5 lakh saalana income hai", "2.5l_to_5l"),
            ("6 lakhs per year", "above_5l"),
            ("50 hazaar lakh nahi, bas 80 hazaar saal ka", None),  # no lakh figure
        ]
        for text, expected in cases:
            got = IntakeAgent._extract_income(text.lower())
            if expected is None:
                continue  # mixed phrasing — just must not crash
            assert got == expected, f"{text!r} -> {got}, expected {expected}"

    def test_monthly_lakh_amounts_are_annualized(self):
        got = IntakeAgent._extract_income("1 lakh per month salary hai")
        assert got == "above_5l"


class TestApplyExtractedRobustness:
    """_apply_extracted must survive malformed LLM output without crashing
    the whole turn into an error fallback."""

    def test_scalar_field_confidence_does_not_crash(self):
        from vaidya.models.user_profile import UserProfile

        agent = IntakeAgent(client=object(), model="mock")
        profile = UserProfile()
        # LLM returned field_confidence as a float instead of an object.
        extracted = {
            "extracted_fields": {"state": "Bihar", "family_size": 5},
            "field_confidence": 0.9,
        }
        out = agent._apply_extracted(profile, extracted)
        assert out.state == "Bihar"
        assert out.family_size == 5

    def test_scalar_extracted_fields_does_not_crash(self):
        from vaidya.models.user_profile import UserProfile

        agent = IntakeAgent(client=object(), model="mock")
        profile = UserProfile()
        extracted = {"extracted_fields": "Bihar", "field_confidence": {}}
        out = agent._apply_extracted(profile, extracted)
        assert out is profile  # nothing applied, but no crash


class TestRobustConfirmation:
    """Confirmation is biased to proceed: only an explicit correction
    request blocks (a false 'not confirmed' traps the caller forever)."""

    @pytest.mark.asyncio
    async def test_yes_proceeds_to_complete(self):
        from unittest.mock import AsyncMock

        from vaidya.models.conversation import ConversationContext, ConversationPhase
        from vaidya.models.user_profile import UserProfile

        agent = IntakeAgent(client=object(), model="mock")
        agent._extract_confirmation = AsyncMock(
            return_value={"confirmed": False}
        )  # flaky LLM says no
        ctx = ConversationContext(
            call_id="c", phone_number_hash="h", language="en-IN", phase=ConversationPhase.INTAKE
        )
        ctx.metadata["confirmation_pending"] = True
        for yes in [
            "Yes, everything is correct",
            "haan sahi hai",
            "yes that's right",
            "thik ache",
        ]:
            ctx.metadata["confirmation_pending"] = True
            resp = await agent._handle_confirmation_response(ctx, UserProfile(), yes, "en-IN")
            assert resp.metadata.get("intake_complete"), yes

    @pytest.mark.asyncio
    async def test_explicit_correction_blocks(self):
        from unittest.mock import AsyncMock

        from vaidya.models.conversation import ConversationContext, ConversationPhase
        from vaidya.models.user_profile import UserProfile

        agent = IntakeAgent(client=object(), model="mock")
        agent._extract_confirmation = AsyncMock(
            return_value={"confirmed": False, "correction_field": "family_size", "spoken_text": ""}
        )
        ctx = ConversationContext(
            call_id="c", phone_number_hash="h", language="en-IN", phase=ConversationPhase.INTAKE
        )
        ctx.metadata["confirmation_pending"] = True
        resp = await agent._handle_confirmation_response(
            ctx, UserProfile(), "No, the family size is wrong", "en-IN"
        )
        assert not resp.metadata.get("intake_complete")
        assert resp.metadata.get("intake_correction")
