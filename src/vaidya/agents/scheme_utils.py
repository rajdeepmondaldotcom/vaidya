"""Shared scheme filtering and serialization for eligibility-type agents."""

from __future__ import annotations

import json
from typing import Any

from vaidya.agents.constants import MAX_SCHEMES_PER_LLM_CALL
from vaidya.models.scheme import (
    EligibilityVerdict,
    Jurisdiction,
    SchemeRecord,
)


def filter_schemes_by_state(
    schemes: list[SchemeRecord],
    user_state: str | None,
) -> list[SchemeRecord]:
    """Pre-filter schemes to those relevant to the user's state.

    Central schemes apply to all states. State schemes apply only
    if the state matches. When the user's state is unknown, all
    schemes are included.
    """
    if not user_state:
        return schemes

    user_state_lower = user_state.lower().strip()
    filtered: list[SchemeRecord] = []

    for scheme in schemes:
        if scheme.jurisdiction == Jurisdiction.CENTRAL:
            filtered.append(scheme)
            continue

        if not scheme.geographic_restrictions:
            filtered.append(scheme)
            continue

        if any(user_state_lower in r.lower() for r in scheme.geographic_restrictions):
            filtered.append(scheme)

    return filtered


def serialize_for_prompt(
    schemes: list[SchemeRecord],
    *,
    include_procedures: bool = True,
    max_schemes: int = MAX_SCHEMES_PER_LLM_CALL,
) -> list[dict[str, Any]]:
    """Convert scheme records to a compact dict list for LLM prompts.

    When *include_procedures* is True (eligibility agent), the output
    includes ``covered_procedures`` and ``secc_categories``.  When False
    (reviewer agent), these are omitted to keep the prompt focused on
    exclusion-rule checking.
    """
    out: list[dict[str, Any]] = []
    for s in schemes[:max_schemes]:
        entry: dict[str, Any] = {
            "scheme_id": s.scheme_id,
            "canonical_name": s.canonical_name,
            "jurisdiction": s.jurisdiction.value,
            "state_code": s.state_code,
            "income_thresholds": [t.model_dump(mode="json") for t in s.income_thresholds],
            "occupation_included": s.occupation_included,
            "occupation_excluded": s.occupation_excluded,
            "exclusion_rules": [r.model_dump(mode="json") for r in s.exclusion_rules],
            "age_criteria": (s.age_criteria.model_dump(mode="json") if s.age_criteria else None),
            "family_criteria": s.family_criteria.model_dump(mode="json"),
            "geographic_restrictions": s.geographic_restrictions,
            "coverage_amount_inr": s.coverage_amount_inr,
            "coverage_type": s.coverage_type.value,
        }
        if include_procedures:
            entry["secc_categories"] = s.secc_categories
            entry["covered_procedures"] = s.covered_procedures[:10]
        out.append(entry)
    return out


def parse_verdict(raw: str) -> EligibilityVerdict:
    """Map raw string to EligibilityVerdict, defaulting to UNCERTAIN."""
    mapping = {
        "eligible": EligibilityVerdict.ELIGIBLE,
        "ineligible": EligibilityVerdict.INELIGIBLE,
        "uncertain": EligibilityVerdict.UNCERTAIN,
    }
    return mapping.get(raw.strip().lower(), EligibilityVerdict.UNCERTAIN)


def json_compact(obj: Any) -> str:
    """Compact JSON string for prompt embedding."""
    return json.dumps(obj, ensure_ascii=False, default=str)
