"""Latency profiler for the Vaidya conversation pipeline.

Drives the ``POST /simulate/text`` endpoint of a running Vaidya server with a
multi-turn conversation and reports how snappy each turn feels. Use it to
quantify and track conversational latency as the pipeline is optimised toward
the "real real" / instant bar.

How per-turn latency is measured
--------------------------------
``/simulate/text`` replays an *entire* conversation server-side in a single
request, so one call only yields a total wall-clock number. To attribute time
to individual turns, this profiler (by default) issues a sequence of requests
with progressively longer turn prefixes::

    request 1 -> [turn_1]
    request 2 -> [turn_1, turn_2]
    request 3 -> [turn_1, turn_2, turn_3]
    ...

The marginal latency of turn *n* is ``elapsed(n) - elapsed(n-1)``. This is an
approximation -- it re-runs earlier turns each time and includes fixed
per-request overhead (conversation start + language handshake) in turn 1 -- but
it surfaces which turns dominate end-to-end latency. Pass ``--single-request``
to instead fire one request with all turns and report total latency only (no
per-turn breakdown), which is cheaper and avoids the re-run overhead.

Usage
-----
    # Default Hindi turn script against a local server
    python scripts/profile_turn.py

    # Point at a deployed server, change language
    python scripts/profile_turn.py --base-url https://vaidya.up.railway.app --language ta-IN

    # Supply your own turns (JSON list of strings)
    python scripts/profile_turn.py --turns-file my_turns.json

    # One request, total latency only
    python scripts/profile_turn.py --single-request

Real numbers require a running server (``make run`` /
``uvicorn vaidya.app:create_app --factory --port 8000``) **and** a valid
``SARVAM_API_KEY`` in the server's environment -- the pipeline makes live STT /
LLM / translation calls per turn. Without them the script will simply report a
connection error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Default multi-turn Hindi conversation: greeting -> location -> family ->
# income/occupation -> insurance. Mirrors the eval scenario style (romanised
# Hindi) so it exercises the full intake -> eligibility -> guidance flow.
DEFAULT_LANGUAGE = "hi-IN"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0  # seconds -- multi-turn calls can be slow

DEFAULT_TURNS: list[str] = [
    "Namaste, mujhe apne parivaar ke liye sarkari swasthya yojana ke baare mein jaanna hai",
    "Main Rajasthan mein rehta hoon, Jaipur ke paas ek gaon mein",
    "Meri family mein 5 log hain - main, meri patni, do bacche aur meri maa",
    "Main construction mein daily wage pe kaam karta hoon, mahine ka 7 hazaar kamata hoon",
    "Nahi, humare paas koi bhi health insurance nahi hai",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TurnTiming:
    """Latency attributed to a single conversational turn."""

    index: int  # 1-based turn number
    text: str
    latency_ms: float


@dataclass
class ProfileResult:
    """Outcome of a full profiling run."""

    timings: list[TurnTiming]
    total_ms: float
    final_phase: str
    eligible_scheme_names: list[str]
    total_cost_inr: float | None
    single_request: bool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Accepts an explicit *argv* (excluding the program name) to make the parser
    unit-testable without touching ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="profile_turn",
        description="Profile per-turn latency of the Vaidya /simulate/text pipeline.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running Vaidya API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code for the conversation (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--turns-file",
        default=None,
        help="Path to a JSON file containing a list of user turn strings. "
        "Defaults to a built-in Hindi turn script.",
    )
    parser.add_argument(
        "--single-request",
        action="store_true",
        help="Send one request with all turns and report total latency only "
        "(no per-turn breakdown).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT:.0f})",
    )
    return parser.parse_args(argv)


def load_turns(turns_file: str | None) -> list[str]:
    """Load the turn script from *turns_file*, or return the default script.

    Raises ``ValueError`` if the file is not a JSON list of non-empty strings.
    """
    if turns_file is None:
        return list(DEFAULT_TURNS)

    path = Path(turns_file)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, list) or not data:
        raise ValueError(f"{turns_file}: expected a non-empty JSON list of turn strings")
    if not all(isinstance(turn, str) and turn.strip() for turn in data):
        raise ValueError(f"{turns_file}: every turn must be a non-empty string")

    return [str(turn) for turn in data]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


async def _simulate(
    client: httpx.AsyncClient,
    base_url: str,
    turns: list[str],
    language: str,
) -> dict:
    """POST a set of turns to /simulate/text and return the parsed response."""
    resp = await client.post(
        f"{base_url}/simulate/text",
        json={"turns": turns, "language": language},
    )
    resp.raise_for_status()
    return resp.json()


async def profile(
    client: httpx.AsyncClient,
    base_url: str,
    turns: list[str],
    language: str,
    single_request: bool,
) -> ProfileResult:
    """Run the profiling requests and assemble a :class:`ProfileResult`.

    With *single_request* false (default), sends one request per growing turn
    prefix and derives each turn's marginal latency. The final response (the
    full conversation) supplies phase / eligibility / cost.
    """
    timings: list[TurnTiming] = []

    if single_request:
        start = time.perf_counter()
        data = await _simulate(client, base_url, turns, language)
        total_ms = (time.perf_counter() - start) * 1000.0
    else:
        prev_cumulative_ms = 0.0
        data: dict = {}
        for i in range(1, len(turns) + 1):
            prefix = turns[:i]
            req_start = time.perf_counter()
            data = await _simulate(client, base_url, prefix, language)
            cumulative_ms = (time.perf_counter() - req_start) * 1000.0
            # Marginal latency relative to the previous (shorter) prefix run.
            marginal_ms = cumulative_ms - prev_cumulative_ms
            prev_cumulative_ms = cumulative_ms
            timings.append(TurnTiming(index=i, text=turns[i - 1], latency_ms=marginal_ms))
        # The full-conversation latency is the last (longest) prefix's wall
        # time -- which equals the sum of the per-turn marginals above. (Summing
        # every prefix request would double-count the re-run turns.)
        total_ms = prev_cumulative_ms

    return ProfileResult(
        timings=timings,
        total_ms=total_ms,
        final_phase=data.get("final_phase", "unknown"),
        eligible_scheme_names=data.get("eligible_scheme_names", []),
        total_cost_inr=data.get("total_cost_inr"),
        single_request=single_request,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_report(result: ProfileResult, base_url: str, language: str) -> str:
    """Render a human-readable latency report as a string."""
    lines: list[str] = []
    bar = "=" * 72
    lines.append(bar)
    lines.append("  Vaidya turn-latency profile")
    lines.append(f"  API: {base_url}  |  Language: {language}")
    mode = "single-request (total only)" if result.single_request else "per-turn prefixes"
    lines.append(f"  Mode: {mode}")
    lines.append(bar)

    if result.timings:
        text_w = 46
        lines.append(f"  {'#':>2}  {'turn':<{text_w}}  {'latency':>11}")
        lines.append(f"  {'-' * 2}  {'-' * text_w}  {'-' * 11}")
        for t in result.timings:
            lines.append(
                f"  {t.index:>2}  {_truncate(t.text, text_w):<{text_w}}  {t.latency_ms:>8.0f} ms"
            )
        lines.append(f"  {'-' * 2}  {'-' * text_w}  {'-' * 11}")
        per_turn = result.total_ms / max(len(result.timings), 1)
        lines.append(f"  {'avg/turn':>{text_w + 4}}  {per_turn:>8.0f} ms")
    else:
        lines.append("  (per-turn breakdown unavailable in single-request mode)")

    lines.append("")
    lines.append(f"  Total latency:        {result.total_ms:>10.0f} ms")
    cost = result.total_cost_inr
    cost_str = "n/a" if cost is None else f"INR {cost:.4f}"
    lines.append(f"  Total cost:           {cost_str:>14}")
    lines.append(f"  Final phase:          {result.final_phase}")
    names = ", ".join(result.eligible_scheme_names) if result.eligible_scheme_names else "(none)"
    lines.append(f"  Eligible schemes:     {names}")
    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")

    try:
        turns = load_turns(args.turns_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not load turns: {exc}", file=sys.stderr)
        return 2

    print(
        f"Profiling {len(turns)} turn(s) against {base_url} ({args.language}).\n"
        "Note: real latency numbers require a running Vaidya server "
        "(make run) and a valid SARVAM_API_KEY in its environment.\n"
    )

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        try:
            result = await profile(
                client,
                base_url,
                turns,
                args.language,
                single_request=args.single_request,
            )
        except httpx.HTTPStatusError as exc:
            print(
                f"Server returned {exc.response.status_code} for {exc.request.url}. "
                "Is the server healthy and configured with SARVAM_API_KEY?",
                file=sys.stderr,
            )
            return 1
        except httpx.HTTPError as exc:
            print(
                f"Could not reach the Vaidya API at {base_url}: {exc}\n"
                "Start it with: make run  (or uvicorn vaidya.app:create_app --factory)",
                file=sys.stderr,
            )
            return 1

    print(render_report(result, base_url, args.language))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
