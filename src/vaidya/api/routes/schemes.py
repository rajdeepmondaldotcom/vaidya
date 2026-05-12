"""Scheme reference data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from vaidya.dependencies import get_schemes as get_schemes_dep
from vaidya.models.api import SchemeResponse
from vaidya.models.scheme import SchemeRecord
from vaidya.schemes.selection import filter_schemes_by_state

router = APIRouter()


def _scheme_to_response(scheme: SchemeRecord) -> SchemeResponse:
    """Convert a SchemeRecord to its API response representation."""
    return SchemeResponse(
        scheme_id=scheme.scheme_id,
        canonical_name=scheme.canonical_name,
        coverage_amount_inr=scheme.coverage_amount_inr,
        jurisdiction=scheme.jurisdiction.value,
        state_code=scheme.state_code,
        description=scheme.description_for_embedding,
    )


@router.get("", response_model=list[SchemeResponse])
async def list_schemes(
    state: str | None = None,
    schemes: list[SchemeRecord] = Depends(get_schemes_dep),
) -> list[SchemeResponse]:
    """List all available healthcare schemes."""
    filtered = filter_schemes_by_state(schemes, state) if state else schemes

    return [_scheme_to_response(s) for s in filtered]


@router.get("/{scheme_id}", response_model=SchemeResponse)
async def get_scheme(
    scheme_id: str,
    schemes: list[SchemeRecord] = Depends(get_schemes_dep),
) -> SchemeResponse:
    """Get details for a specific scheme."""
    scheme = next((s for s in schemes if s.scheme_id == scheme_id), None)
    if scheme is None:
        raise HTTPException(status_code=404, detail=f"Scheme {scheme_id} not found")

    return _scheme_to_response(scheme)
