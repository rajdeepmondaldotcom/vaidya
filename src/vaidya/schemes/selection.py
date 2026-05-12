"""Scheme applicability helpers shared by agents and API routes."""

from __future__ import annotations

from vaidya.models.scheme import Jurisdiction, SchemeRecord
from vaidya.utils.states import state_name_to_code


def filter_schemes_by_state(
    schemes: list[SchemeRecord],
    user_state: str | None,
) -> list[SchemeRecord]:
    """Return every scheme applicable to the user's state.

    Central schemes apply unless the normalized state code is listed in
    ``geographic_restrictions``. State schemes apply only when their
    ``state_code`` exactly matches the normalized user state. When the state
    is unknown, return the full registry so potentially relevant schemes are
    not silently excluded.
    """
    state_code = state_name_to_code(user_state)
    if not state_code:
        return schemes

    filtered: list[SchemeRecord] = []
    for scheme in schemes:
        if scheme.jurisdiction == Jurisdiction.CENTRAL:
            restrictions = {
                restriction.upper().strip() for restriction in scheme.geographic_restrictions
            }
            if state_code not in restrictions:
                filtered.append(scheme)
            continue

        if scheme.state_code == state_code:
            filtered.append(scheme)

    return filtered
