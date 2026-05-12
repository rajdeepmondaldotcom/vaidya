"""Scheme registry: loads all scheme JSONs and provides lookup functions."""

from __future__ import annotations

import json
from pathlib import Path

from vaidya.models.scheme import SchemeRecord
from vaidya.schemes.selection import filter_schemes_by_state

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
    """Return central schemes applicable to *state_code* plus state-specific schemes."""
    return filter_schemes_by_state(get_schemes(), state_code)
