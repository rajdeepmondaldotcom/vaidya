"""Comprehensive PM-JAY scheme unit tests.

Tests the PM-JAY (Pradhan Mantri Jan Arogya Yojana) scheme data for:
- Data completeness (IDs, names, coverage, jurisdiction, metadata)
- All 7 exclusion rules (rule IDs, types, fields, state overrides)
- Coverage details (procedures, exclusions)
- Document and enrollment requirements
- State-based filtering behavior
- Occupation lists (included and excluded)
"""

from __future__ import annotations

import pytest

from vaidya.agents.scheme_utils import filter_schemes_by_state, serialize_for_prompt
from vaidya.models.scheme import (
    ConfidenceLevel,
    ExclusionType,
    Jurisdiction,
    SchemeCoverageType,
    SchemeRecord,
)
from vaidya.schemes.registry import get_schemes


@pytest.fixture(scope="module")
def pmjay() -> SchemeRecord:
    """Load PM-JAY once for the entire module."""
    matches = [s for s in get_schemes() if s.scheme_id == "PMJAY-2024-v3"]
    assert matches, "PM-JAY scheme (PMJAY-2024-v3) not found in registry"
    return matches[0]


@pytest.fixture(scope="module")
def all_schemes() -> list[SchemeRecord]:
    """Load all schemes once for filtering tests."""
    return get_schemes()


# ---------------------------------------------------------------------------
# Class 1: Data completeness
# ---------------------------------------------------------------------------


class TestPMJAYDataCompleteness:
    """Verify the PM-JAY JSON has all required production data."""

    def test_scheme_loads_successfully(self, pmjay: SchemeRecord) -> None:
        assert pmjay.scheme_id == "PMJAY-2024-v3"

    def test_canonical_name(self, pmjay: SchemeRecord) -> None:
        assert "Pradhan Mantri" in pmjay.canonical_name
        assert "Jan Arogya" in pmjay.canonical_name

    def test_coverage_amount(self, pmjay: SchemeRecord) -> None:
        assert pmjay.coverage_amount_inr == 500000, "PM-JAY coverage must be Rs 5,00,000"

    def test_coverage_type(self, pmjay: SchemeRecord) -> None:
        assert pmjay.coverage_type == SchemeCoverageType.PER_FAMILY_PER_YEAR

    def test_jurisdiction_is_central(self, pmjay: SchemeRecord) -> None:
        assert pmjay.jurisdiction == Jurisdiction.CENTRAL

    def test_no_state_code(self, pmjay: SchemeRecord) -> None:
        assert pmjay.state_code is None

    def test_geographic_restrictions_opt_out_states(self, pmjay: SchemeRecord) -> None:
        assert "WB" in pmjay.geographic_restrictions
        assert "DL" in pmjay.geographic_restrictions

    def test_helpline_number(self, pmjay: SchemeRecord) -> None:
        assert pmjay.helpline_number == "14555"

    def test_has_secc_categories(self, pmjay: SchemeRecord) -> None:
        expected = {"D1", "D2", "D3", "D4", "D5", "D6", "D7"}
        actual = set(pmjay.secc_categories)
        assert expected.issubset(actual), f"Missing SECC categories: {expected - actual}"

    def test_has_aliases(self, pmjay: SchemeRecord) -> None:
        aliases_lower = [a.lower() for a in pmjay.aliases]
        assert any("pmjay" in a for a in aliases_lower), "Missing PMJAY alias"
        assert any("ayushman bharat" in a for a in aliases_lower), "Missing Ayushman Bharat alias"
        assert any("ab-pmjay" in a for a in aliases_lower), "Missing AB-PMJAY alias"

    def test_has_local_names(self, pmjay: SchemeRecord) -> None:
        required_langs = {"hi", "bn", "ta", "te"}
        actual_langs = set(pmjay.local_names.keys())
        assert required_langs.issubset(actual_langs), (
            f"Missing local names for: {required_langs - actual_langs}"
        )

    def test_has_keywords(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.keywords) >= 20, (
            f"Expected at least 20 keywords, got {len(pmjay.keywords)}"
        )

    def test_confidence_level_verified(self, pmjay: SchemeRecord) -> None:
        assert pmjay.confidence_level == ConfidenceLevel.VERIFIED

    def test_effective_date(self, pmjay: SchemeRecord) -> None:
        assert pmjay.effective_date == "2018-09-23"

    def test_no_expiry(self, pmjay: SchemeRecord) -> None:
        assert pmjay.expiry_date is None


# ---------------------------------------------------------------------------
# Class 2: Exclusion rules
# ---------------------------------------------------------------------------


class TestPMJAYExclusionRules:
    """Tests for all 7 exclusion rules."""

    def test_has_seven_exclusion_rules(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.exclusion_rules) == 7

    def test_employer_insurance_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-001"), None)
        assert rule is not None, "PMJAY-EX-001 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "existing_coverage"

    def test_govt_employee_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-002"), None)
        assert rule is not None, "PMJAY-EX-002 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "occupation_type"

    def test_income_tax_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-003"), None)
        assert rule is not None, "PMJAY-EX-003 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "income_tax"

    def test_motorized_vehicle_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-004"), None)
        assert rule is not None, "PMJAY-EX-004 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "motorized_vehicle"

    def test_mechanized_farming_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-005"), None)
        assert rule is not None, "PMJAY-EX-005 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "mechanized_farming"

    def test_kisan_credit_card_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-006"), None)
        assert rule is not None, "PMJAY-EX-006 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "kisan_credit"

    def test_land_ownership_exclusion(self, pmjay: SchemeRecord) -> None:
        rule = next((r for r in pmjay.exclusion_rules if r.rule_id == "PMJAY-EX-007"), None)
        assert rule is not None, "PMJAY-EX-007 not found"
        assert rule.exclusion_type == ExclusionType.HARD
        assert rule.field == "land_ownership"

    def test_all_exclusions_are_hard(self, pmjay: SchemeRecord) -> None:
        for rule in pmjay.exclusion_rules:
            assert rule.exclusion_type == ExclusionType.HARD, (
                f"{rule.rule_id} is not hard exclusion"
            )

    def test_no_state_override_on_exclusions(self, pmjay: SchemeRecord) -> None:
        for rule in pmjay.exclusion_rules:
            assert rule.override_by_state is False, f"{rule.rule_id} has override_by_state=True"


# ---------------------------------------------------------------------------
# Class 3: Coverage details
# ---------------------------------------------------------------------------


class TestPMJAYCoverage:
    """Tests for coverage details: procedures, exclusions."""

    def test_covered_procedures_count(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.covered_procedures) >= 15, (
            f"Expected at least 15 covered procedures, got {len(pmjay.covered_procedures)}"
        )

    def test_covers_cardiology(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("cardio" in p for p in procs_lower), "No cardiology procedure found"

    def test_covers_oncology(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("oncology" in p or "chemotherapy" in p for p in procs_lower), (
            "No oncology procedure found"
        )

    def test_covers_dialysis(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("dialysis" in p or "nephro" in p for p in procs_lower), (
            "No dialysis/nephrology procedure found"
        )

    def test_covers_maternity(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        # Check for obstetrics, gynecology, maternity, neonatal
        assert any(
            term in p
            for p in procs_lower
            for term in ("obstetric", "gynecol", "matern", "neonatal")
        ), "No maternity/obstetrics/neonatal procedure found"

    def test_covers_emergency(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("emergency" in p or "trauma" in p for p in procs_lower), (
            "No emergency/trauma procedure found"
        )

    def test_covers_mental_health(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("mental" in p or "psychiatr" in p for p in procs_lower), (
            "No mental health procedure found"
        )

    def test_covers_pre_post_hospitalization(self, pmjay: SchemeRecord) -> None:
        procs_lower = [p.lower() for p in pmjay.covered_procedures]
        assert any("pre-hospital" in p or "pre hospital" in p for p in procs_lower), (
            "No pre-hospitalization entry found"
        )
        assert any(
            "post-hospital" in p
            or "post hospital" in p
            or "post-discharge" in p
            or "after discharge" in p
            for p in procs_lower
        ), "No post-hospitalization entry found"

    def test_excludes_opd(self, pmjay: SchemeRecord) -> None:
        excl_lower = [p.lower() for p in pmjay.excluded_procedures]
        assert any("opd" in p or "out-patient" in p for p in excl_lower), (
            "OPD not in excluded procedures"
        )

    def test_excludes_cosmetic(self, pmjay: SchemeRecord) -> None:
        excl_lower = [p.lower() for p in pmjay.excluded_procedures]
        assert any("cosmetic" in p for p in excl_lower), (
            "Cosmetic surgery not in excluded procedures"
        )

    def test_excludes_fertility(self, pmjay: SchemeRecord) -> None:
        excl_lower = [p.lower() for p in pmjay.excluded_procedures]
        assert any("fertility" in p or "ivf" in p for p in excl_lower), (
            "Fertility/IVF not in excluded procedures"
        )


# ---------------------------------------------------------------------------
# Class 4: Documents and enrollment
# ---------------------------------------------------------------------------


class TestPMJAYDocumentsAndEnrollment:
    """Tests for enrollment process and required documents."""

    def test_aadhaar_required(self, pmjay: SchemeRecord) -> None:
        aadhaar_docs = [d for d in pmjay.required_documents if "aadhaar" in d.name.lower()]
        assert aadhaar_docs, "No Aadhaar document found"
        assert any(d.mandatory for d in aadhaar_docs), "Aadhaar is not marked mandatory"

    def test_ration_card_listed(self, pmjay: SchemeRecord) -> None:
        doc_names_lower = [d.name.lower() for d in pmjay.required_documents]
        assert any("ration" in n for n in doc_names_lower), "Ration card not in required documents"

    def test_enrollment_channels_count(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.enrollment_channels) >= 6, (
            f"Expected at least 6 enrollment channels, got {len(pmjay.enrollment_channels)}"
        )

    def test_csc_in_channels(self, pmjay: SchemeRecord) -> None:
        channels_lower = [c.lower() for c in pmjay.enrollment_channels]
        assert any("csc" in c or "jan seva" in c for c in channels_lower), (
            "CSC / Jan Seva Kendra not in enrollment channels"
        )

    def test_enrollment_steps_count(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.enrollment_steps) >= 7, (
            f"Expected at least 7 enrollment steps, got {len(pmjay.enrollment_steps)}"
        )

    def test_enrollment_mentions_ekyc(self, pmjay: SchemeRecord) -> None:
        steps_text = " ".join(pmjay.enrollment_steps).lower()
        assert "e-kyc" in steps_text or "ekyc" in steps_text, (
            "e-KYC not mentioned in enrollment steps"
        )

    def test_enrollment_mentions_golden_card(self, pmjay: SchemeRecord) -> None:
        steps_text = " ".join(pmjay.enrollment_steps).lower()
        assert "golden card" in steps_text or "ayushman card" in steps_text, (
            "Golden Card / Ayushman card not mentioned in enrollment steps"
        )


# ---------------------------------------------------------------------------
# Class 5: State filtering
# ---------------------------------------------------------------------------


class TestPMJAYStateFiltering:
    """Tests for state-based scheme filtering using scheme_utils."""

    def test_pmjay_included_for_rajasthan(
        self,
        pmjay: SchemeRecord,
        all_schemes: list[SchemeRecord],
    ) -> None:
        filtered = filter_schemes_by_state(all_schemes, "Rajasthan")
        ids = [s.scheme_id for s in filtered]
        assert pmjay.scheme_id in ids

    def test_pmjay_included_for_tamil_nadu(
        self,
        pmjay: SchemeRecord,
        all_schemes: list[SchemeRecord],
    ) -> None:
        filtered = filter_schemes_by_state(all_schemes, "Tamil Nadu")
        ids = [s.scheme_id for s in filtered]
        assert pmjay.scheme_id in ids

    def test_pmjay_included_for_karnataka(
        self,
        pmjay: SchemeRecord,
        all_schemes: list[SchemeRecord],
    ) -> None:
        filtered = filter_schemes_by_state(all_schemes, "Karnataka")
        ids = [s.scheme_id for s in filtered]
        assert pmjay.scheme_id in ids

    def test_pmjay_included_for_unknown_state(
        self,
        pmjay: SchemeRecord,
        all_schemes: list[SchemeRecord],
    ) -> None:
        # When state is None, all schemes (including PM-JAY) are returned
        filtered = filter_schemes_by_state(all_schemes, None)
        ids = [s.scheme_id for s in filtered]
        assert pmjay.scheme_id in ids

    def test_pmjay_included_for_west_bengal(
        self,
        pmjay: SchemeRecord,
        all_schemes: list[SchemeRecord],
    ) -> None:
        # filter_schemes_by_state includes ALL central schemes regardless of
        # geographic_restrictions.  The geographic_restrictions field is used
        # by the LLM for eligibility decisions, not by this pre-filter.
        # Central schemes always pass the jurisdiction check.
        filtered = filter_schemes_by_state(all_schemes, "West Bengal")
        ids = [s.scheme_id for s in filtered]
        assert pmjay.scheme_id in ids

    def test_pmjay_serialization_includes_procedures(
        self,
        pmjay: SchemeRecord,
    ) -> None:
        serialized = serialize_for_prompt([pmjay], include_procedures=True)
        assert len(serialized) == 1
        assert "covered_procedures" in serialized[0]
        assert len(serialized[0]["covered_procedures"]) > 0

    def test_pmjay_serialization_excludes_procedures(
        self,
        pmjay: SchemeRecord,
    ) -> None:
        serialized = serialize_for_prompt([pmjay], include_procedures=False)
        assert len(serialized) == 1
        assert "covered_procedures" not in serialized[0]


# ---------------------------------------------------------------------------
# Class 6: Occupations
# ---------------------------------------------------------------------------


class TestPMJAYOccupations:
    """Tests for included and excluded occupation lists."""

    def test_daily_wage_included(self, pmjay: SchemeRecord) -> None:
        assert "daily_wage" in pmjay.occupation_included

    def test_farmer_included(self, pmjay: SchemeRecord) -> None:
        assert "farmer" in pmjay.occupation_included

    def test_construction_worker_included(self, pmjay: SchemeRecord) -> None:
        assert "construction_worker" in pmjay.occupation_included

    def test_govt_employee_excluded(self, pmjay: SchemeRecord) -> None:
        assert "government_employee" in pmjay.occupation_excluded

    def test_income_tax_payer_excluded(self, pmjay: SchemeRecord) -> None:
        assert "income_tax_payer" in pmjay.occupation_excluded

    def test_occupation_included_count(self, pmjay: SchemeRecord) -> None:
        assert len(pmjay.occupation_included) >= 14, (
            f"Expected at least 14 included occupations, got {len(pmjay.occupation_included)}"
        )
