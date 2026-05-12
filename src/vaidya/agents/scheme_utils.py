"""Shared scheme selection and serialization for eligibility-type agents."""

from __future__ import annotations

import json
from typing import Any

from vaidya.models.scheme import (
    EligibilityVerdict,
    SchemeMatch,
    SchemeRecord,
)
from vaidya.schemes.selection import filter_schemes_by_state

__all__ = [
    "batch_schemes",
    "filter_schemes_by_state",
    "json_compact",
    "missing_candidate_ids",
    "normalize_matches_for_candidates",
    "parse_verdict",
    "serialize_for_prompt",
    "uncertain_matches_for_missing",
]


def batch_schemes(
    schemes: list[SchemeRecord],
    batch_size: int,
) -> list[list[SchemeRecord]]:
    """Split schemes into non-empty batches, clamping invalid sizes to one."""
    size = max(1, batch_size)
    return [schemes[idx : idx + size] for idx in range(0, len(schemes), size)]


def serialize_for_prompt(
    schemes: list[SchemeRecord],
    *,
    include_procedures: bool = True,
    max_schemes: int | None = None,
) -> list[dict[str, Any]]:
    """Convert scheme records to a compact dict list for LLM prompts.

    When *include_procedures* is True (eligibility agent), the output
    includes ``covered_procedures`` and ``secc_categories``.  When False
    (reviewer agent), these are omitted to keep the prompt focused on
    exclusion-rule checking. ``max_schemes`` is a per-call safety bound only;
    callers that need complete evaluation should batch before serializing.
    """
    out: list[dict[str, Any]] = []
    selected = schemes if max_schemes is None else schemes[:max_schemes]
    for s in selected:
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


def normalize_matches_for_candidates(
    matches: list[SchemeMatch],
    candidates: list[SchemeRecord],
) -> list[SchemeMatch]:
    """Drop hallucinated IDs, fill names, dedupe, and preserve candidate order."""
    lookup = {scheme.scheme_id: scheme for scheme in candidates}
    by_id: dict[str, SchemeMatch] = {}

    for match in matches:
        if match.scheme_id not in lookup or match.scheme_id in by_id:
            continue
        scheme = lookup[match.scheme_id]
        scheme_name = match.scheme_name
        if not scheme_name or scheme_name == match.scheme_id:
            scheme_name = scheme.canonical_name
        by_id[match.scheme_id] = match.model_copy(update={"scheme_name": scheme_name})

    return [by_id[scheme.scheme_id] for scheme in candidates if scheme.scheme_id in by_id]


def missing_candidate_ids(
    matches: list[SchemeMatch],
    candidates: list[SchemeRecord],
) -> list[str]:
    """Return candidate scheme IDs not represented in the parsed matches."""
    seen = {match.scheme_id for match in matches}
    return [scheme.scheme_id for scheme in candidates if scheme.scheme_id not in seen]


def uncertain_matches_for_missing(
    missing_ids: list[str],
    candidates: list[SchemeRecord],
    *,
    source: str,
) -> list[SchemeMatch]:
    """Build auditable conservative matches for schemes omitted by the LLM."""
    lookup = {scheme.scheme_id: scheme for scheme in candidates}
    matches: list[SchemeMatch] = []

    for scheme_id in missing_ids:
        scheme = lookup.get(scheme_id)
        if scheme is None:
            continue
        matches.append(
            SchemeMatch(
                scheme_id=scheme.scheme_id,
                scheme_name=scheme.canonical_name,
                verdict=EligibilityVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning_trace=(
                    f"{source} did not return a verdict for this scheme after retry; "
                    "using a conservative uncertain fallback."
                ),
                matched_criteria=[],
                failed_criteria=[f"{source}_missing_verdict"],
                coverage_summary=(
                    f"Potential coverage up to Rs {scheme.coverage_amount_inr:,}"
                    if scheme.coverage_amount_inr
                    else "Potential comprehensive coverage or service benefit"
                ),
            )
        )

    return matches


def json_compact(obj: Any) -> str:
    """Compact JSON string for prompt embedding."""
    return json.dumps(obj, ensure_ascii=False, default=str)
