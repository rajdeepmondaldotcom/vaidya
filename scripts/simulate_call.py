"""Interactive CLI call simulator for Vaidya.

Simulates a phone call by sending user messages to the Vaidya conversation API
one turn at a time. Displays the assistant's response after each turn.

Usage
-----
    python scripts/simulate_call.py
    python scripts/simulate_call.py --base-url http://localhost:8000
    python scripts/simulate_call.py --language ta-IN
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time

import httpx

# ANSI color codes for terminal output
_BOLD = "\033[1m"
_RESET = "\033[0m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI call simulator for Vaidya",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running Vaidya API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--language",
        default="hi-IN",
        help="Language code for the conversation (default: hi-IN)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout per request in seconds (default: 60)",
    )
    return parser.parse_args()


async def _start_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    language: str,
) -> tuple[str, str]:
    """Start a new conversation and return (call_id, welcome_message)."""
    phone_hash = hashlib.sha256(f"cli-sim-{time.time()}".encode()).hexdigest()[:16]
    resp = await client.post(
        f"{base_url}/conversation/start",
        json={"phone_number_hash": phone_hash, "language": language},
    )
    resp.raise_for_status()
    data = resp.json()
    return data["call_id"], data["message"]


async def _send_turn(
    client: httpx.AsyncClient,
    base_url: str,
    call_id: str,
    text: str,
    language: str,
) -> dict:
    """Send one user turn and return the full response dict."""
    resp = await client.post(
        f"{base_url}/conversation/{call_id}/turn",
        json={"text": text, "language": language},
    )
    resp.raise_for_status()
    return resp.json()


async def _get_conversation_state(
    client: httpx.AsyncClient,
    base_url: str,
    call_id: str,
) -> dict:
    """Fetch the current conversation state."""
    resp = await client.get(f"{base_url}/conversation/{call_id}")
    resp.raise_for_status()
    return resp.json()


def _print_assistant(text: str) -> None:
    print(f"\n{_GREEN}{_BOLD}Vaidya:{_RESET} {_GREEN}{text}{_RESET}")


def _print_meta(phase: str, schemes_found: int | None, latency_ms: float) -> None:
    parts = [f"phase={phase}"]
    if schemes_found is not None:
        parts.append(f"schemes_found={schemes_found}")
    parts.append(f"latency={latency_ms:.0f}ms")
    print(f"{_DIM}  [{', '.join(parts)}]{_RESET}")


async def _run_interactive(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    language = args.language

    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}  Vaidya Call Simulator{_RESET}")
    print(f"{_DIM}  API: {base_url}  |  Language: {language}{_RESET}")
    print(f"{_DIM}  Type your messages. Commands: /quit, /status, /lang <code>{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # Health check
        try:
            health = await client.get(f"{base_url}/health")
            health.raise_for_status()
            print(f"{_DIM}  Server: OK ({health.json().get('version', '?')}){_RESET}")
        except Exception as exc:
            print(f"\n{_YELLOW}Cannot reach Vaidya API at {base_url}: {exc}{_RESET}")
            print("Start the server with: python -m vaidya")
            return

        # Start conversation
        try:
            start_time = time.perf_counter()
            call_id, welcome = await _start_conversation(client, base_url, language)
            start_latency = (time.perf_counter() - start_time) * 1000
        except Exception as exc:
            print(f"\n{_YELLOW}Failed to start conversation: {exc}{_RESET}")
            return

        print(f"\n{_DIM}  call_id: {call_id}{_RESET}")
        _print_assistant(welcome)
        _print_meta("welcome", None, start_latency)

        # Turn loop
        while True:
            try:
                user_input = input(f"\n{_CYAN}{_BOLD}You:{_RESET} {_CYAN}").strip()
                print(_RESET, end="")  # reset color after input
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{_DIM}Call ended.{_RESET}")
                break

            if not user_input:
                continue

            # Handle CLI commands
            if user_input.lower() == "/quit":
                print(f"\n{_DIM}Call ended.{_RESET}")
                break

            if user_input.lower() == "/status":
                try:
                    state = await _get_conversation_state(client, base_url, call_id)
                    print(f"\n{_DIM}  Phase: {state.get('phase')}")
                    print(f"  Intake progress: {state.get('intake_progress')}")
                    profile = state.get("user_profile", {})
                    print(f"  Profile: state={profile.get('state')}, "
                          f"income={profile.get('income_bracket')}, "
                          f"occupation={profile.get('occupation_type')}, "
                          f"coverage={profile.get('existing_coverage')}")
                    eligible = state.get("eligible_schemes")
                    if eligible:
                        print(f"  Eligible schemes: {', '.join(eligible)}")
                    print(_RESET, end="")
                except Exception as exc:
                    print(f"{_YELLOW}  Status error: {exc}{_RESET}")
                continue

            if user_input.lower().startswith("/lang "):
                language = user_input.split(maxsplit=1)[1].strip()
                print(f"{_DIM}  Language set to: {language}{_RESET}")
                continue

            # Send turn
            try:
                turn_start = time.perf_counter()
                response = await _send_turn(client, base_url, call_id, user_input, language)
                turn_latency = (time.perf_counter() - turn_start) * 1000
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    print(f"\n{_YELLOW}Session expired or not found. Start a new call.{_RESET}")
                    break
                print(f"\n{_YELLOW}API error: {exc}{_RESET}")
                continue
            except Exception as exc:
                print(f"\n{_YELLOW}Request failed: {exc}{_RESET}")
                continue

            _print_assistant(response["text"])
            _print_meta(
                response.get("phase", "?"),
                response.get("schemes_found"),
                turn_latency,
            )

        # Show final state
        print(f"\n{_DIM}--- Final conversation state ---{_RESET}")
        try:
            final = await _get_conversation_state(client, base_url, call_id)
            print(f"{_DIM}  Phase: {final.get('phase')}")
            eligible = final.get("eligible_schemes")
            if eligible:
                print(f"  Eligible schemes: {', '.join(eligible)}")
            else:
                print("  No eligible schemes determined")
            print(_RESET)
        except Exception:
            print(f"{_DIM}  (could not fetch final state){_RESET}")


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run_interactive(args))
    except KeyboardInterrupt:
        print(f"\n{_DIM}Interrupted.{_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
