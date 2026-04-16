"""Unit tests for vaidya.utils.states — Indian state/UT code mappings."""

from __future__ import annotations

from vaidya.utils.states import (
    INDIAN_STATES,
    normalize_state,
    state_code_to_name,
    state_name_to_code,
)

# ---------------------------------------------------------------------------
# state_name_to_code
# ---------------------------------------------------------------------------


class TestStateNameToCode:
    def test_full_name(self):
        assert state_name_to_code("Maharashtra") == "MH"

    def test_abbreviation(self):
        assert state_name_to_code("RJ") == "RJ"

    def test_lowercase(self):
        assert state_name_to_code("tamil nadu") == "TN"

    def test_none_input(self):
        assert state_name_to_code(None) is None

    def test_empty_string(self):
        assert state_name_to_code("") is None

    def test_alias(self):
        assert state_name_to_code("bengal") == "WB"
        assert state_name_to_code("orissa") == "OD"

    def test_unknown_name(self):
        assert state_name_to_code("Narnia") is None

    def test_whitespace_handling(self):
        assert state_name_to_code("  Maharashtra  ") == "MH"

    def test_mixed_case(self):
        assert state_name_to_code("MAHARASHTRA") == "MH"
        assert state_name_to_code("maharashtra") == "MH"


# ---------------------------------------------------------------------------
# state_code_to_name
# ---------------------------------------------------------------------------


class TestStateCodeToName:
    def test_valid_code(self):
        assert state_code_to_name("WB") == "West Bengal"

    def test_valid_code_lowercase(self):
        assert state_code_to_name("wb") == "West Bengal"

    def test_invalid_code(self):
        assert state_code_to_name("XX") is None

    def test_all_28_states(self):
        state_codes = [
            "AP",
            "AR",
            "AS",
            "BR",
            "CG",
            "GA",
            "GJ",
            "HR",
            "HP",
            "JH",
            "KA",
            "KL",
            "MP",
            "MH",
            "MN",
            "ML",
            "MZ",
            "NL",
            "OD",
            "PB",
            "RJ",
            "SK",
            "TN",
            "TS",
            "TR",
            "UK",
            "UP",
            "WB",
        ]
        for code in state_codes:
            assert state_code_to_name(code) is not None, f"Missing state for code {code}"

    def test_all_8_uts(self):
        ut_codes = ["AN", "CH", "DN", "DL", "JK", "LA", "LD", "PY"]
        for code in ut_codes:
            assert state_code_to_name(code) is not None, f"Missing UT for code {code}"


# ---------------------------------------------------------------------------
# normalize_state
# ---------------------------------------------------------------------------


class TestNormalizeState:
    def test_full_name_returned_as_canonical(self):
        assert normalize_state("Maharashtra") == "Maharashtra"

    def test_code_to_canonical_name(self):
        assert normalize_state("MH") == "Maharashtra"

    def test_alias_to_canonical_name(self):
        assert normalize_state("bengal") == "West Bengal"
        assert normalize_state("orissa") == "Odisha"

    def test_none_input(self):
        assert normalize_state(None) is None

    def test_empty_string(self):
        assert normalize_state("") == ""

    def test_unknown_returns_unchanged(self):
        assert normalize_state("SomePlace") == "SomePlace"


# ---------------------------------------------------------------------------
# INDIAN_STATES completeness
# ---------------------------------------------------------------------------


class TestIndianStatesCompleteness:
    def test_28_states_present(self):
        state_codes = {
            "AP",
            "AR",
            "AS",
            "BR",
            "CG",
            "GA",
            "GJ",
            "HR",
            "HP",
            "JH",
            "KA",
            "KL",
            "MP",
            "MH",
            "MN",
            "ML",
            "MZ",
            "NL",
            "OD",
            "PB",
            "RJ",
            "SK",
            "TN",
            "TS",
            "TR",
            "UK",
            "UP",
            "WB",
        }
        for code in state_codes:
            assert code in INDIAN_STATES, f"State {code} missing from INDIAN_STATES"

    def test_8_uts_present(self):
        ut_codes = {"AN", "CH", "DN", "DL", "JK", "LA", "LD", "PY"}
        for code in ut_codes:
            assert code in INDIAN_STATES, f"UT {code} missing from INDIAN_STATES"

    def test_total_count_is_36(self):
        assert len(INDIAN_STATES) == 36

    def test_specific_names(self):
        assert INDIAN_STATES["DL"] == "Delhi"
        assert INDIAN_STATES["TN"] == "Tamil Nadu"
        assert INDIAN_STATES["JK"] == "Jammu and Kashmir"
        assert INDIAN_STATES["KA"] == "Karnataka"
