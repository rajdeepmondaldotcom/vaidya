"""Unit tests for scripts/profile_turn.py.

These tests exercise turn-script construction, argument parsing, and the
profiling control flow with ``httpx`` fully mocked -- no live network calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# scripts/ is not an importable package, so load the module by path. The module
# must be registered in sys.modules before exec_module so that @dataclass can
# resolve cls.__module__ during class processing.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "profile_turn.py"
_spec = importlib.util.spec_from_file_location("profile_turn", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
profile_turn = importlib.util.module_from_spec(_spec)
sys.modules["profile_turn"] = profile_turn
_spec.loader.exec_module(profile_turn)


# ---------------------------------------------------------------------------
# Turn-script construction
# ---------------------------------------------------------------------------


def test_default_turns_are_a_five_step_hindi_script() -> None:
    turns = profile_turn.load_turns(None)
    assert turns == profile_turn.DEFAULT_TURNS
    # greeting -> location -> family -> income/occupation -> insurance
    assert len(turns) == 5
    assert all(isinstance(t, str) and t for t in turns)
    # Returns a fresh copy, not the module-level list.
    assert turns is not profile_turn.DEFAULT_TURNS


def test_load_turns_from_file(tmp_path: Path) -> None:
    custom = ["pehla turn", "doosra turn"]
    f = tmp_path / "turns.json"
    f.write_text(json.dumps(custom), encoding="utf-8")
    assert profile_turn.load_turns(str(f)) == custom


def test_load_turns_rejects_non_list(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"turns": ["x"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        profile_turn.load_turns(str(f))


def test_load_turns_rejects_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "empty.json"
    f.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        profile_turn.load_turns(str(f))


def test_load_turns_rejects_blank_strings(tmp_path: Path) -> None:
    f = tmp_path / "blank.json"
    f.write_text(json.dumps(["ok", "   "]), encoding="utf-8")
    with pytest.raises(ValueError):
        profile_turn.load_turns(str(f))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = profile_turn.parse_args([])
    assert args.base_url == profile_turn.DEFAULT_BASE_URL
    assert args.language == profile_turn.DEFAULT_LANGUAGE
    assert args.turns_file is None
    assert args.single_request is False
    assert args.timeout == profile_turn.DEFAULT_TIMEOUT


def test_parse_args_overrides() -> None:
    args = profile_turn.parse_args(
        [
            "--base-url",
            "https://vaidya.example.com/",
            "--language",
            "ta-IN",
            "--turns-file",
            "turns.json",
            "--single-request",
            "--timeout",
            "30",
        ]
    )
    assert args.base_url == "https://vaidya.example.com/"
    assert args.language == "ta-IN"
    assert args.turns_file == "turns.json"
    assert args.single_request is True
    assert args.timeout == 30.0


# ---------------------------------------------------------------------------
# Mocked profiling flow (no real network)
# ---------------------------------------------------------------------------


def _mock_client(payload: dict[str, Any]) -> MagicMock:
    """Build a fake httpx.AsyncClient whose POST returns *payload*."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value=payload)

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_profile_per_turn_sends_growing_prefixes() -> None:
    turns = ["a", "b", "c"]
    payload = {
        "final_phase": "guidance",
        "eligible_scheme_names": ["PM-JAY"],
        "total_cost_inr": 0.1234,
    }
    client = _mock_client(payload)

    result = await profile_turn.profile(
        client,
        "http://localhost:8000",
        turns,
        "hi-IN",
        single_request=False,
    )

    # One request per turn, each with a growing prefix.
    assert client.post.await_count == 3
    sent_turns = [call.kwargs["json"]["turns"] for call in client.post.await_args_list]
    assert sent_turns == [["a"], ["a", "b"], ["a", "b", "c"]]
    # Every request carries the language and hits the simulate endpoint.
    assert all(call.kwargs["json"]["language"] == "hi-IN" for call in client.post.await_args_list)
    assert all(call.args[0].endswith("/simulate/text") for call in client.post.await_args_list)

    # One timing per turn, indexed 1..N, carrying the right text.
    assert [t.index for t in result.timings] == [1, 2, 3]
    assert [t.text for t in result.timings] == turns
    # Per-turn marginals must reconcile with the reported total latency, so the
    # table column sums to the headline number rather than over-counting re-runs.
    assert sum(t.latency_ms for t in result.timings) == pytest.approx(result.total_ms)
    assert result.final_phase == "guidance"
    assert result.eligible_scheme_names == ["PM-JAY"]
    assert result.total_cost_inr == pytest.approx(0.1234)
    assert result.single_request is False


@pytest.mark.asyncio
async def test_profile_single_request_sends_all_turns_once() -> None:
    turns = ["a", "b", "c"]
    client = _mock_client({"final_phase": "guidance", "eligible_scheme_names": []})

    result = await profile_turn.profile(
        client,
        "http://localhost:8000",
        turns,
        "hi-IN",
        single_request=True,
    )

    assert client.post.await_count == 1
    assert client.post.await_args.kwargs["json"]["turns"] == turns
    assert result.timings == []  # no per-turn breakdown in single-request mode
    assert result.total_cost_inr is None  # absent key -> None
    assert result.single_request is True


@pytest.mark.asyncio
async def test_profile_strips_trailing_slash_is_callers_job() -> None:
    # profile() uses base_url as given; ensure no double slash sneaks in for a
    # clean base url.
    client = _mock_client({})
    await profile_turn.profile(client, "http://x", ["a"], "hi-IN", single_request=True)
    assert client.post.await_args.args[0] == "http://x/simulate/text"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_report_per_turn_contains_key_fields() -> None:
    result = profile_turn.ProfileResult(
        timings=[
            profile_turn.TurnTiming(index=1, text="greeting", latency_ms=120.0),
            profile_turn.TurnTiming(index=2, text="location", latency_ms=80.0),
        ],
        total_ms=200.0,
        final_phase="guidance",
        eligible_scheme_names=["PM-JAY", "Chiranjeevi"],
        total_cost_inr=0.5,
        single_request=False,
    )
    out = profile_turn.render_report(result, "http://localhost:8000", "hi-IN")
    assert "Vaidya turn-latency profile" in out
    assert "greeting" in out and "location" in out
    assert "Total latency" in out
    assert "INR 0.5000" in out
    assert "PM-JAY, Chiranjeevi" in out
    assert "guidance" in out


def test_render_report_handles_no_cost_and_no_schemes() -> None:
    result = profile_turn.ProfileResult(
        timings=[],
        total_ms=50.0,
        final_phase="intake",
        eligible_scheme_names=[],
        total_cost_inr=None,
        single_request=True,
    )
    out = profile_turn.render_report(result, "http://localhost:8000", "hi-IN")
    assert "n/a" in out  # cost
    assert "(none)" in out  # schemes
    assert "single-request" in out
