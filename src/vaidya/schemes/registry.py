"""Scheme registry: loads all scheme JSONs and provides lookup functions."""

from __future__ import annotations

import json
from pathlib import Path

from vaidya.models.scheme import Jurisdiction, SchemeRecord

_SCHEME_DIR = Path(__file__).parent / "data"
_SCHEMES: list[SchemeRecord] = []


def _load_schemes() -> list[SchemeRecord]:
    """Read every ``*.json`` file in the data directory and parse into *SchemeRecord*."""
    schemes: list[SchemeRecord] = []
    for json_file in sorted(_SCHEME_DIR.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        schemes.append(SchemeRecord.model_validate(data))
    return schemes


def get_schemes() -> list[SchemeRecord]:
    """Return all loaded schemes, loading from disk on first call."""
    global _SCHEMES  # noqa: PLW0603
    if not _SCHEMES:
        _SCHEMES = _load_schemes()
    return _SCHEMES


def get_scheme_by_id(scheme_id: str) -> SchemeRecord | None:
    """Look up a single scheme by its unique *scheme_id*."""
    return next((s for s in get_schemes() if s.scheme_id == scheme_id), None)


def get_schemes_for_state(state_code: str) -> list[SchemeRecord]:
    """Return central schemes applicable to *state_code* plus state-specific schemes.

    Central schemes are included unless *state_code* appears in the scheme's
    ``geographic_restrictions`` list.  State schemes are included when their
    ``state_code`` matches.
    """
    return [
        s
        for s in get_schemes()
        if (s.jurisdiction == Jurisdiction.CENTRAL and state_code not in s.geographic_restrictions)
        or s.state_code == state_code
    ]
