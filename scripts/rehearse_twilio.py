#!/usr/bin/env python3
"""Rehearse Twilio HTTP webhook validation against a running Vaidya server."""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run signed Twilio webhook checks.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Public HTTPS base URL for the running app, e.g. https://x.trycloudflare.com",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout per request in seconds.",
    )
    return parser.parse_args()


def _require_https_base_url(raw: str) -> str:
    base_url = raw.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("--base-url must be a public https:// URL")
    return base_url


def _signature(url: str, params: dict[str, str], token: str) -> str:
    try:
        from twilio.request_validator import RequestValidator
    except ImportError as exc:
        raise RuntimeError("Install Twilio support with: pip install -e '.[telephony]'") from exc
    return str(RequestValidator(token).compute_signature(url, params))


def _post_signed(
    client: httpx.Client,
    url: str,
    params: dict[str, str],
    token: str,
) -> httpx.Response:
    return client.post(
        url,
        data=params,
        headers={"X-Twilio-Signature": _signature(url, params, token)},
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = _parse_args()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not token:
        print("TWILIO_AUTH_TOKEN is required in the environment", file=sys.stderr)
        sys.exit(2)

    try:
        base_url = _require_https_base_url(args.base_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    incoming_url = f"{base_url}/voice/incoming"
    status_url = f"{base_url}/voice/status"
    expected_stream_url = f"wss://{urlparse(base_url).netloc}/voice/stream"

    incoming_params = {"From": "+15551234567", "CallSid": "CA_REHEARSAL"}
    status_params = {
        "CallStatus": "completed",
        "CallSid": "CA_REHEARSAL",
        "CallDuration": "1",
    }

    with httpx.Client(timeout=args.timeout) as client:
        incoming = _post_signed(client, incoming_url, incoming_params, token)
        _assert(incoming.status_code == 200, f"/voice/incoming returned {incoming.status_code}")
        _assert(
            expected_stream_url in incoming.text,
            f"TwiML did not include expected stream URL {expected_stream_url}",
        )
        _assert("+15551234567" not in incoming.text, "TwiML leaked the raw caller number")
        _assert("phone_hash" in incoming.text, "TwiML did not include hashed phone parameter")

        invalid = client.post(
            incoming_url,
            data=incoming_params,
            headers={"X-Twilio-Signature": "invalid"},
        )
        _assert(
            invalid.status_code == 403,
            f"invalid incoming signature returned {invalid.status_code}",
        )

        status = _post_signed(client, status_url, status_params, token)
        _assert(status.status_code == 200, f"/voice/status returned {status.status_code}")
        _assert(status.json().get("status") == "ok", "/voice/status did not return ok")

        invalid_status = client.post(
            status_url,
            data=status_params,
            headers={"X-Twilio-Signature": "invalid"},
        )
        _assert(
            invalid_status.status_code == 403,
            f"invalid status signature returned {invalid_status.status_code}",
        )

    print("Twilio webhook rehearsal passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"Twilio rehearsal failed: {exc}", file=sys.stderr)
        sys.exit(1)
