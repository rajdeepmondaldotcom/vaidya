"""Scheme, eligibility, and convergence models for Vaidya."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Jurisdiction(StrEnum):
    """Whether a scheme is run centrally or by a state government."""

    CENTRAL = "central"
    STATE = "state"


class SchemeCoverageType(StrEnum):
    """How the scheme's coverage amount applies."""

    PER_FAMILY_PER_YEAR = "per_family_per_year"
    PER_PERSON_PER_YEAR = "per_person_per_year"
    PER_EVENT = "per_event"


class ConfidenceLevel(StrEnum):
    """Freshness / reliability of the scheme record."""

    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    STALE = "stale"


class ExclusionType(StrEnum):
    """Hard exclusions are absolute; soft exclusions may be overridden."""

    HARD = "hard"
    SOFT = "soft"


class EligibilityVerdict(StrEnum):
    """Outcome of eligibility evaluation for a single scheme."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Scheme record sub-models
# ---------------------------------------------------------------------------


class IncomeThreshold(BaseModel):
    """Income ceiling for scheme eligibility."""

    max_income_inr: int
    unit: str  # "per_annum" | "per_month"
    applies_to: str  # "family" | "individual"
    description: str
    source: str


class ExclusionRule(BaseModel):
    """A single exclusion criterion attached to a scheme."""

    rule_id: str
    exclusion_type: ExclusionType
    field: str
    condition: str
    description: str
    override_by_state: bool


class AgeCriteria(BaseModel):
    """Age bounds for scheme eligibility."""

    min_age: int | None = None
    max_age: int | None = None
    applies_to: str  # e.g. "beneficiary", "head_of_family"


class FamilyCriteria(BaseModel):
    """Family-related eligibility constraints."""

    max_family_size: int | None = None
    family_definition: str
    head_of_family_required: bool


class Document(BaseModel):
    """A document required for scheme enrollment."""

    name: str
    category: str
    mandatory: bool
    alternatives: list[str]
    local_name_hi: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Scheme record
# ---------------------------------------------------------------------------


class SchemeRecord(BaseModel):
    """Complete representation of a government healthcare scheme.

    This is the canonical data object stored in the scheme registry and
    used by the eligibility and reviewer agents.
    """

    model_config = ConfigDict(populate_by_name=True)

    scheme_id: str
    canonical_name: str
    aliases: list[str]
    local_names: dict[str, str]
    state_variant_of: str | None = None

    # Jurisdiction
    jurisdiction: Jurisdiction
    state_code: str | None = None

    # Eligibility criteria
    income_thresholds: list[IncomeThreshold]
    secc_categories: list[str]
    occupation_included: list[str]
    occupation_excluded: list[str]
    exclusion_rules: list[ExclusionRule]
    age_criteria: AgeCriteria | None = None
    family_criteria: FamilyCriteria
    geographic_restrictions: list[str]

    # Coverage
    coverage_amount_inr: int
    coverage_type: SchemeCoverageType
    covered_procedures: list[str]
    excluded_procedures: list[str]

    # Enrollment
    required_documents: list[Document]
    enrollment_channels: list[str]
    enrollment_steps: list[str]
    processing_time_days: int
    helpline_number: str = ""
    portal_url: str = ""

    # Metadata
    version: str
    effective_date: str
    expiry_date: str | None = None
    last_verified: str
    source_url: str
    confidence_level: ConfidenceLevel

    # Search / embedding
    description_for_embedding: str
    keywords: list[str]


# ---------------------------------------------------------------------------
# Eligibility evaluation results
# ---------------------------------------------------------------------------


class SchemeMatch(BaseModel):
    """A single scheme's eligibility evaluation outcome."""

    scheme_id: str
    scheme_name: str
    verdict: EligibilityVerdict
    confidence: float
    reasoning_trace: str
    matched_criteria: list[str]
    failed_criteria: list[str]
    coverage_summary: str


class EligibilityResult(BaseModel):
    """Output of the eligibility agent across all candidate schemes."""

    matches: list[SchemeMatch]
    processing_time_ms: float
    model_used: str
    schemes_evaluated: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def eligible_schemes(self) -> list[SchemeMatch]:
        """Convenience filter: only schemes with ELIGIBLE verdict."""
        return [m for m in self.matches if m.verdict == EligibilityVerdict.ELIGIBLE]


class ReviewerResult(BaseModel):
    """Output of the reviewer agent that cross-checks eligibility."""

    matches: list[SchemeMatch]
    processing_time_ms: float
    model_used: str
    transcript_evidence: list[str] = []


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class DisagreementRecord(BaseModel):
    """A scheme where the eligibility and reviewer agents disagree."""

    scheme_id: str
    scheme_name: str
    eligibility_verdict: EligibilityVerdict
    reviewer_verdict: EligibilityVerdict
    eligibility_reasoning: str
    reviewer_reasoning: str
    disagreement_field: str
    resolved_from_transcript: bool
    final_verdict: EligibilityVerdict
    caveat: str


class ConvergenceResult(BaseModel):
    """Merged output after the two agents converge."""

    agreed_eligible: list[SchemeMatch] = Field(default_factory=list)
    agreed_ineligible: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementRecord] = Field(default_factory=list)
    conservative_eligible: list[SchemeMatch] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_eligible(self) -> list[SchemeMatch]:
        """All schemes the caller should be told about."""
        return self.agreed_eligible + self.conservative_eligible


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------


class SpokenPart(BaseModel):
    """A single spoken segment (headline, benefit, action, or no_match)."""

    type: str
    text: str


class GuidanceOutput(BaseModel):
    """TTS-ready spoken guidance plus SMS summary."""

    spoken_parts: list[SpokenPart]
    sms_summary: str
    has_more_schemes: bool
    caveat_needed: bool
    processing_time_ms: float
