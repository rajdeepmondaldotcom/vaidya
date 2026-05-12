"""Deterministic intake extraction fallback tests."""

from __future__ import annotations

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
