"""Per-state metadata for all 36 Indian states and union territories.

Maps each 2-letter state code to its primary languages (BCP-47),
whether it operates its own health insurance scheme beyond PM-JAY,
the scheme IDs present in ``src/vaidya/schemes/data/``, and whether
the state has opted out of PM-JAY (currently only Delhi and West Bengal).

Usage::

    from vaidya.utils.state_metadata import STATE_METADATA, get_state_info

    info = get_state_info("TN")
    assert info is not None
    assert info.primary_languages == ["ta-IN"]
    assert "CMCHIS-TN-2024-v1" in info.scheme_ids
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateInfo:
    """Immutable metadata for a single Indian state or union territory."""

    primary_languages: list[str]
    """BCP-47 language codes spoken in this state, most common first."""

    has_own_scheme: bool
    """Whether the state runs its own health insurance scheme beyond PM-JAY."""

    scheme_ids: list[str]
    """Scheme record IDs from ``src/vaidya/schemes/data/`` (empty if none)."""

    pmjay_excluded: bool = False
    """True only for states that have opted out of PM-JAY (WB, DL)."""


# ---------------------------------------------------------------------------
# States WITH their own health scheme
# ---------------------------------------------------------------------------

_STATES_WITH_SCHEME: dict[str, StateInfo] = {
    "AP": StateInfo(
        primary_languages=["te-IN"],
        has_own_scheme=True,
        scheme_ids=["AAROGYASRI-AP-2024-v1"],
    ),
    "AS": StateInfo(
        primary_languages=["as-IN"],
        has_own_scheme=True,
        scheme_ids=["ATAL-AMRIT-AS-2024-v1"],
    ),
    "BR": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["MSBY-BR-2024-v1"],
    ),
    "CG": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["DKBSSY-CG-2024-v1"],
    ),
    "DL": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["DAK-DL-2024-v1", "DAN-DL-2024-v1"],
        pmjay_excluded=True,
    ),
    "GA": StateInfo(
        primary_languages=["kok-IN", "en-IN"],
        has_own_scheme=True,
        scheme_ids=["DDSSY-GA-2024-v1"],
    ),
    "GJ": StateInfo(
        primary_languages=["gu-IN"],
        has_own_scheme=True,
        scheme_ids=["MA-VATSALYA-GJ-2024-v1"],
    ),
    "HR": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["CHIRAYU-HR-2024-v1"],
    ),
    "HP": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["HIMCARE-HP-2024-v1"],
    ),
    "JH": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["ABUA-JH-2024-v1"],
    ),
    "JK": StateInfo(
        primary_languages=["ur-IN", "hi-IN"],
        has_own_scheme=True,
        scheme_ids=["SEHAT-JK-2024-v1"],
    ),
    "KA": StateInfo(
        primary_languages=["kn-IN"],
        has_own_scheme=True,
        scheme_ids=["AK-KA-2024-v2", "YESHASVINI-KA-2024-v1"],
    ),
    "KL": StateInfo(
        primary_languages=["ml-IN"],
        has_own_scheme=True,
        scheme_ids=["KASP-KL-2024-v1"],
    ),
    "MH": StateInfo(
        primary_languages=["mr-IN"],
        has_own_scheme=True,
        scheme_ids=["MJPJAY-MH-2024-v2"],
    ),
    "ML": StateInfo(
        primary_languages=["en-IN"],
        has_own_scheme=True,
        scheme_ids=["MHIS-ML-2024-v1"],
    ),
    "MP": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["NIRAMAYAM-MP-2024-v1"],
    ),
    "OD": StateInfo(
        primary_languages=["od-IN"],
        has_own_scheme=True,
        scheme_ids=["BSKY-OD-2024-v1"],
    ),
    "PB": StateInfo(
        primary_languages=["pa-IN"],
        has_own_scheme=True,
        scheme_ids=["MMSY-PB-2024-v1"],
    ),
    "PY": StateInfo(
        primary_languages=["ta-IN"],
        has_own_scheme=True,
        scheme_ids=["CMCHIS-PY-2024-v1"],
    ),
    "RJ": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["CHIR-RJ-2024-v2"],
    ),
    "SK": StateInfo(
        primary_languages=["ne-IN"],
        has_own_scheme=True,
        scheme_ids=["CMCHCS-SK-2024-v1"],
    ),
    "TN": StateInfo(
        primary_languages=["ta-IN"],
        has_own_scheme=True,
        scheme_ids=["CMCHIS-TN-2024-v1"],
    ),
    "TS": StateInfo(
        primary_languages=["te-IN"],
        has_own_scheme=True,
        scheme_ids=["AAROGYASRI-TS-2024-v1"],
    ),
    "UK": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["ATAL-UK-2024-v1"],
    ),
    "UP": StateInfo(
        primary_languages=["hi-IN"],
        has_own_scheme=True,
        scheme_ids=["MJAY-UP-2024-v1"],
    ),
    "AR": StateInfo(
        primary_languages=["en-IN"],
        has_own_scheme=True,
        scheme_ids=["CMAAY-AR-2024-v1"],
    ),
    "WB": StateInfo(
        primary_languages=["bn-IN"],
        has_own_scheme=True,
        scheme_ids=["SS-WB-2024-v2"],
        pmjay_excluded=True,
    ),
}

# ---------------------------------------------------------------------------
# States WITHOUT their own health scheme (PM-JAY only)
# ---------------------------------------------------------------------------

_STATES_WITHOUT_SCHEME: dict[str, StateInfo] = {
    "MN": StateInfo(
        primary_languages=["mni-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "MZ": StateInfo(
        primary_languages=["en-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "NL": StateInfo(
        primary_languages=["en-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "TR": StateInfo(
        primary_languages=["bn-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "AN": StateInfo(
        primary_languages=["hi-IN", "en-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "CH": StateInfo(
        primary_languages=["hi-IN", "pa-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "DN": StateInfo(
        primary_languages=["gu-IN", "hi-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "LA": StateInfo(
        primary_languages=["ur-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
    "LD": StateInfo(
        primary_languages=["ml-IN"],
        has_own_scheme=False,
        scheme_ids=[],
    ),
}

# ---------------------------------------------------------------------------
# Combined lookup table
# ---------------------------------------------------------------------------

STATE_METADATA: dict[str, StateInfo] = {**_STATES_WITH_SCHEME, **_STATES_WITHOUT_SCHEME}
"""All 36 states/UTs keyed by 2-letter code."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_state_info(state_code: str) -> StateInfo | None:
    """Return metadata for a state code, or ``None`` if not found.

    The lookup is case-insensitive and strips whitespace::

        >>> get_state_info("tn")
        StateInfo(primary_languages=['ta-IN'], has_own_scheme=True, ...)
    """
    return STATE_METADATA.get(state_code.upper().strip())


def get_primary_language(state_code: str) -> str:
    """Return the first primary language for a state code.

    Falls back to ``"hi-IN"`` (Hindi) when the code is unknown::

        >>> get_primary_language("TN")
        'ta-IN'
        >>> get_primary_language("XX")
        'hi-IN'
    """
    info = get_state_info(state_code)
    if info and info.primary_languages:
        return info.primary_languages[0]
    return "hi-IN"
