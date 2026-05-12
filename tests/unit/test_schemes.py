"""Tests for scheme data loading and validation.

Verifies:
- All scheme JSONs load successfully via get_schemes()
- Required fields present on every scheme
- PM-JAY has exclusion rules
- Swasthya Sathi is state=WB
- PM-JAY 70+ has age criteria with min_age=70
- PMSBY has age criteria with max_age=70
- Chiranjeevi has state_code="RJ"
- Additional validation: enrollment steps, documents, state lookups
"""

from __future__ import annotations

from vaidya.models.scheme import (
    ConfidenceLevel,
    ExclusionType,
    Jurisdiction,
)
from vaidya.schemes.registry import get_scheme_by_id, get_schemes, get_schemes_for_state

# ---------------------------------------------------------------------------
# Loading and count
# ---------------------------------------------------------------------------


class TestSchemeLoading:
    def test_loads_all_schemes(self) -> None:
        schemes = get_schemes()
        # 8 original + 25+ new central and state schemes
        assert len(schemes) >= 30, f"Expected 30+ schemes, got {len(schemes)}"

    def test_all_scheme_ids_unique(self) -> None:
        schemes = get_schemes()
        ids = [s.scheme_id for s in schemes]
        assert len(ids) == len(set(ids)), f"Duplicate scheme_ids found: {ids}"

    def test_schemes_are_cached(self) -> None:
        """Second call returns the same list object (module-level caching)."""
        first = get_schemes()
        second = get_schemes()
        assert first is second


# ---------------------------------------------------------------------------
# Required fields validation
# ---------------------------------------------------------------------------


class TestSchemeRequiredFields:
    def test_all_have_scheme_id(self) -> None:
        for s in get_schemes():
            assert s.scheme_id, "Missing scheme_id"

    def test_all_have_canonical_name(self) -> None:
        for s in get_schemes():
            assert s.canonical_name, f"Missing canonical_name for {s.scheme_id}"

    def test_all_have_nonnegative_coverage(self) -> None:
        """Coverage must be >= 0. ESIC has 0 (comprehensive social security, not fixed amount)."""
        for s in get_schemes():
            assert s.coverage_amount_inr >= 0, f"Negative coverage for {s.scheme_id}"

    def test_most_have_positive_coverage(self) -> None:
        """Most schemes have coverage_amount_inr > 0; free-service schemes have 0."""
        # These are free-service/comprehensive schemes without a fixed
        # coverage amount: ESIC, CGHS, AB-HWC, JSSK, RBSK, NPHCE, PMNDP
        zero_coverage_ok = {
            "ESIC-2024-v2",
            "CGHS-2024-v1",
            "AB-HWC-2024-v1",
            "JSSK-2024-v1",
            "RBSK-2024-v1",
            "NPHCE-2024-v1",
            "PMNDP-2024-v1",
            "CMCHCS-SK-2024-v1",  # Sikkim: universal free healthcare
        }
        for s in get_schemes():
            if s.scheme_id not in zero_coverage_ok:
                assert s.coverage_amount_inr > 0, f"Zero coverage for {s.scheme_id}"

    def test_all_have_description_for_embedding(self) -> None:
        for s in get_schemes():
            assert s.description_for_embedding, f"Missing embedding desc for {s.scheme_id}"

    def test_all_have_keywords(self) -> None:
        for s in get_schemes():
            assert len(s.keywords) > 0, f"No keywords for {s.scheme_id}"

    def test_all_have_jurisdiction(self) -> None:
        for s in get_schemes():
            assert s.jurisdiction in (Jurisdiction.CENTRAL, Jurisdiction.STATE)

    def test_all_have_version(self) -> None:
        for s in get_schemes():
            assert s.version, f"Missing version for {s.scheme_id}"

    def test_all_have_effective_date(self) -> None:
        for s in get_schemes():
            assert s.effective_date, f"Missing effective_date for {s.scheme_id}"

    def test_all_have_enrollment_steps(self) -> None:
        for s in get_schemes():
            assert len(s.enrollment_steps) > 0, f"No enrollment steps for {s.scheme_id}"

    def test_all_have_required_documents(self) -> None:
        for s in get_schemes():
            assert len(s.required_documents) > 0, f"No documents for {s.scheme_id}"

    def test_all_have_confidence_level(self) -> None:
        for s in get_schemes():
            assert s.confidence_level in (
                ConfidenceLevel.VERIFIED,
                ConfidenceLevel.PROVISIONAL,
                ConfidenceLevel.STALE,
            )


# ---------------------------------------------------------------------------
# PM-JAY
# ---------------------------------------------------------------------------


class TestPMJAY:
    def test_pmjay_exists(self) -> None:
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None

    def test_pmjay_has_exclusion_rules(self) -> None:
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        assert len(pmjay.exclusion_rules) > 0

    def test_pmjay_employer_insurance_exclusion(self) -> None:
        """PM-JAY should exclude families with employer-provided insurance."""
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        exclusion_fields = [r.field for r in pmjay.exclusion_rules]
        assert "existing_coverage" in exclusion_fields

    def test_pmjay_exclusion_rules_are_hard(self) -> None:
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        for rule in pmjay.exclusion_rules:
            assert rule.exclusion_type == ExclusionType.HARD

    def test_pmjay_is_central(self) -> None:
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        assert pmjay.jurisdiction == Jurisdiction.CENTRAL

    def test_pmjay_coverage_5_lakh(self) -> None:
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        assert pmjay.coverage_amount_inr == 500000

    def test_pmjay_excludes_wb_and_dl(self) -> None:
        """West Bengal and Delhi are excluded from PM-JAY."""
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        assert "WB" in pmjay.geographic_restrictions
        assert "DL" in pmjay.geographic_restrictions

    def test_pmjay_no_age_criteria(self) -> None:
        """Base PM-JAY has no age restriction."""
        pmjay = get_scheme_by_id("PMJAY-2024-v3")
        assert pmjay is not None
        assert pmjay.age_criteria is None


# ---------------------------------------------------------------------------
# Swasthya Sathi (West Bengal)
# ---------------------------------------------------------------------------


class TestSwasthyaSathi:
    def test_exists(self) -> None:
        ss = get_scheme_by_id("SS-WB-2024-v2")
        assert ss is not None

    def test_state_is_wb(self) -> None:
        ss = get_scheme_by_id("SS-WB-2024-v2")
        assert ss is not None
        assert ss.state_code == "WB"

    def test_jurisdiction_is_state(self) -> None:
        ss = get_scheme_by_id("SS-WB-2024-v2")
        assert ss is not None
        assert ss.jurisdiction == Jurisdiction.STATE


# ---------------------------------------------------------------------------
# PM-JAY 70+
# ---------------------------------------------------------------------------


class TestPMJAY70Plus:
    def test_exists(self) -> None:
        pmjay70 = get_scheme_by_id("PMJAY-70PLUS-2024-v1")
        assert pmjay70 is not None

    def test_has_age_criteria(self) -> None:
        pmjay70 = get_scheme_by_id("PMJAY-70PLUS-2024-v1")
        assert pmjay70 is not None
        assert pmjay70.age_criteria is not None

    def test_min_age_is_70(self) -> None:
        pmjay70 = get_scheme_by_id("PMJAY-70PLUS-2024-v1")
        assert pmjay70 is not None
        assert pmjay70.age_criteria is not None
        assert pmjay70.age_criteria.min_age == 70

    def test_no_max_age(self) -> None:
        pmjay70 = get_scheme_by_id("PMJAY-70PLUS-2024-v1")
        assert pmjay70 is not None
        assert pmjay70.age_criteria is not None
        assert pmjay70.age_criteria.max_age is None

    def test_is_variant_of_base_pmjay(self) -> None:
        pmjay70 = get_scheme_by_id("PMJAY-70PLUS-2024-v1")
        assert pmjay70 is not None
        assert pmjay70.state_variant_of == "PMJAY-2024-v3"


# ---------------------------------------------------------------------------
# PMSBY
# ---------------------------------------------------------------------------


class TestPMSBY:
    def test_exists(self) -> None:
        pmsby = get_scheme_by_id("PMSBY-2024-v2")
        assert pmsby is not None

    def test_has_age_criteria(self) -> None:
        pmsby = get_scheme_by_id("PMSBY-2024-v2")
        assert pmsby is not None
        assert pmsby.age_criteria is not None

    def test_min_age_18(self) -> None:
        pmsby = get_scheme_by_id("PMSBY-2024-v2")
        assert pmsby is not None
        assert pmsby.age_criteria is not None
        assert pmsby.age_criteria.min_age == 18

    def test_max_age_70(self) -> None:
        pmsby = get_scheme_by_id("PMSBY-2024-v2")
        assert pmsby is not None
        assert pmsby.age_criteria is not None
        assert pmsby.age_criteria.max_age == 70


# ---------------------------------------------------------------------------
# Chiranjeevi (Rajasthan)
# ---------------------------------------------------------------------------


class TestChiranjeevi:
    def test_exists(self) -> None:
        chir = get_scheme_by_id("CHIR-RJ-2024-v2")
        assert chir is not None

    def test_state_code_rj(self) -> None:
        chir = get_scheme_by_id("CHIR-RJ-2024-v2")
        assert chir is not None
        assert chir.state_code == "RJ"

    def test_jurisdiction_is_state(self) -> None:
        chir = get_scheme_by_id("CHIR-RJ-2024-v2")
        assert chir is not None
        assert chir.jurisdiction == Jurisdiction.STATE


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


class TestSchemeLookup:
    def test_get_scheme_by_id_returns_none_for_invalid(self) -> None:
        assert get_scheme_by_id("NONEXISTENT") is None

    def test_get_schemes_for_state_rajasthan(self) -> None:
        """Rajasthan should get central schemes (minus geo-restricted) plus RJ state schemes."""
        rj_schemes = get_schemes_for_state("RJ")
        ids = [s.scheme_id for s in rj_schemes]
        # Chiranjeevi is RJ-specific
        assert "CHIR-RJ-2024-v2" in ids
        # PM-JAY is central and not restricted from RJ
        assert "PMJAY-2024-v3" in ids

    def test_get_schemes_for_state_west_bengal(self) -> None:
        """West Bengal should get Swasthya Sathi but NOT PM-JAY (WB is excluded)."""
        wb_schemes = get_schemes_for_state("WB")
        ids = [s.scheme_id for s in wb_schemes]
        assert "SS-WB-2024-v2" in ids
        assert "PMJAY-2024-v3" not in ids  # WB is in geographic_restrictions

    def test_get_schemes_for_state_returns_central_schemes(self) -> None:
        """A generic state should get all central schemes without geo restrictions."""
        schemes = get_schemes_for_state("UP")
        # At minimum, should include PM-JAY and other central schemes
        central = [s for s in schemes if s.jurisdiction == Jurisdiction.CENTRAL]
        assert len(central) >= 1
