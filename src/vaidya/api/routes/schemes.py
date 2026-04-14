"""Scheme reference data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from vaidya.dependencies import get_schemes as get_schemes_dep
from vaidya.models.api import SchemeResponse
from vaidya.models.scheme import Jurisdiction, SchemeRecord

router = APIRouter()


@router.get("", response_model=list[SchemeResponse])
async def list_schemes(
    state: str | None = None,
    schemes: list[SchemeRecord] = Depends(get_schemes_dep),
) -> list[SchemeResponse]:
    """List all available healthcare schemes."""
    filtered = schemes
    if state:
        state_upper = state.upper()
        filtered = [
            s
            for s in schemes
            if s.jurisdiction == Jurisdiction.CENTRAL or s.state_code == state_upper
        ]

    return [
        SchemeResponse(
            scheme_id=s.scheme_id,
            canonical_name=s.canonical_name,
            coverage_amount_inr=s.coverage_amount_inr,
            jurisdiction=s.jurisdiction.value,
            state_code=s.state_code,
            description=s.description_for_embedding,
        )
        for s in filtered
    ]


@router.get("/{scheme_id}", response_model=SchemeResponse)
async def get_scheme(
    scheme_id: str,
    schemes: list[SchemeRecord] = Depends(get_schemes_dep),
) -> SchemeResponse:
    """Get details for a specific scheme."""
    scheme = next((s for s in schemes if s.scheme_id == scheme_id), None)
    if scheme is None:
        raise HTTPException(status_code=404, detail=f"Scheme {scheme_id} not found")

    return SchemeResponse(
        scheme_id=scheme.scheme_id,
        canonical_name=scheme.canonical_name,
        coverage_amount_inr=scheme.coverage_amount_inr,
        jurisdiction=scheme.jurisdiction.value,
        state_code=scheme.state_code,
        description=scheme.description_for_embedding,
    )
