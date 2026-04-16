"""User profile models for Vaidya intake data."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field


class IncomeCategory(StrEnum):
    """Annual household income brackets aligned with BPL / scheme thresholds."""

    BELOW_1L = "below_1l"
    L1_TO_2_5L = "1l_to_2.5l"
    L2_5_TO_5L = "2.5l_to_5l"
    ABOVE_5L = "above_5l"
    UNKNOWN = "unknown"


class OccupationType(StrEnum):
    """Occupation categories relevant to scheme eligibility."""

    DAILY_WAGE = "daily_wage"
    SALARIED_GOVT = "salaried_govt"
    SALARIED_PVT = "salaried_pvt"
    SELF_EMPLOYED = "self_employed"
    FARMER = "farmer"
    UNKNOWN = "unknown"


class CoverageType(StrEnum):
    """Existing health coverage status."""

    NONE = "none"
    EMPLOYER = "employer"
    GOVT_SCHEME = "govt_scheme"
    PRIVATE = "private"
    UNKNOWN = "unknown"


class UserProfile(BaseModel):
    """Demographic and socio-economic data collected during intake.

    Fields are nullable because they are progressively filled as the
    conversation agent elicits information from the caller.
    """

    model_config = ConfigDict(populate_by_name=True)

    state: str | None = None
    district: str | None = None
    family_size: int | None = None
    income_bracket: IncomeCategory = IncomeCategory.UNKNOWN
    occupation_type: OccupationType = OccupationType.UNKNOWN
    existing_coverage: CoverageType = CoverageType.UNKNOWN
    health_need: str | None = None
    health_need_en: str | None = None  # English translation for RAG queries
    age: int | None = None
    bpl_card: bool | None = None
    ration_card: bool | None = None
    secc_category: str | None = None
    confidence_flags: dict[str, float] = {}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def required_fields_complete(self) -> bool:
        """True when every field needed for eligibility evaluation is set."""
        return (
            self.state is not None
            and self.family_size is not None
            and self.income_bracket != IncomeCategory.UNKNOWN
            and self.occupation_type != OccupationType.UNKNOWN
            and self.existing_coverage != CoverageType.UNKNOWN
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_fields(self) -> list[str]:
        """Return names of required fields that are still unknown/unset."""
        missing: list[str] = []
        if self.state is None:
            missing.append("state")
        if self.family_size is None:
            missing.append("family_size")
        if self.income_bracket == IncomeCategory.UNKNOWN:
            missing.append("income_bracket")
        if self.occupation_type == OccupationType.UNKNOWN:
            missing.append("occupation_type")
        if self.existing_coverage == CoverageType.UNKNOWN:
            missing.append("existing_coverage")
        return missing
