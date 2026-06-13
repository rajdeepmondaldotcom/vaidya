"""Unit tests for the pure formatting helpers in scripts/simulate_call.py.

These exercise the screen-recording-friendly transcript/summary formatters
without any live server -- they are plain string functions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# scripts/ is not an importable package (no __init__.py, and pytest's testpaths
# is "tests"), so load the module directly from its file path.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "simulate_call.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("simulate_call", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


simulate_call = _load_module()


# ---------------------------------------------------------------------------
# format_rupees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (None, "n/a"),
        (0, "Rs 0"),
        (0.0, "Rs 0"),
        (1.5, "Rs 1.5"),
        (2.0, "Rs 2"),
        (0.0123, "Rs 0.0123"),
        (0.00001, "Rs 0"),  # rounds to 4 dp -> 0.0000 -> stripped to 0
    ],
)
def test_format_rupees(amount, expected):
    assert simulate_call.format_rupees(amount) == expected


def test_format_rupees_trims_trailing_zeros():
    # 0.0500 should render compactly, not with padding zeros.
    assert simulate_call.format_rupees(0.05) == "Rs 0.05"


# ---------------------------------------------------------------------------
# format_meta
# ---------------------------------------------------------------------------


def test_format_meta_minimal():
    line = simulate_call.format_meta("intake", None, 123.4)
    assert "phase=intake" in line
    assert "latency=123ms" in line
    assert "schemes_found" not in line
    assert "cost=" not in line


def test_format_meta_full():
    line = simulate_call.format_meta("results", 2, 87.0, 0.0123)
    assert "phase=results" in line
    assert "schemes_found=2" in line
    assert "latency=87ms" in line
    assert "cost=Rs 0.0123" in line


def test_format_meta_schemes_found_zero_is_shown():
    # 0 is a meaningful value (eligibility ran, found nothing) and must appear.
    line = simulate_call.format_meta("results", 0, 10.0)
    assert "schemes_found=0" in line


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------


def test_format_summary_with_schemes():
    summary = simulate_call.format_summary(
        ["Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat)", "Mukhyamantri Chiranjeevi"],
        0.0123,
        turn_count=6,
    )
    assert "Call summary" in summary
    assert "Caller turns:    6 turns" in summary
    assert "Total cost:      Rs 0.0123" in summary
    assert "Eligible for 2 schemes:" in summary
    # Numbered list preserves order.
    assert "1. Pradhan Mantri Jan Arogya Yojana (Ayushman Bharat)" in summary
    assert "2. Mukhyamantri Chiranjeevi" in summary
    # No ANSI escape codes leak into the pure formatter output.
    assert "\033" not in summary


def test_format_summary_single_scheme_and_turn_singular():
    summary = simulate_call.format_summary(["Swasthya Sathi (West Bengal Health Scheme)"], 0.0, 1)
    assert "Caller turns:    1 turn" in summary  # singular
    assert "Eligible for 1 scheme:" in summary  # singular
    assert "1. Swasthya Sathi (West Bengal Health Scheme)" in summary
    assert "Total cost:      Rs 0" in summary


def test_format_summary_no_schemes():
    summary = simulate_call.format_summary(None, None, turn_count=3)
    assert "Eligible schemes: none determined" in summary
    assert "Total cost:      n/a" in summary
    assert "Caller turns:    3 turns" in summary


def test_format_summary_empty_list_treated_as_none():
    summary = simulate_call.format_summary([], 0.01, turn_count=2)
    assert "Eligible schemes: none determined" in summary


def test_format_summary_custom_width():
    summary = simulate_call.format_summary(["X"], 0.0, 1, width=20)
    lines = summary.splitlines()
    assert lines[0] == "=" * 20
    assert lines[-1] == "=" * 20
