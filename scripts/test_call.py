#!/usr/bin/env python3
"""Make a test outbound call via Twilio to verify the voice pipeline.

Usage:
    python scripts/test_call.py --to +919876543210

Requirements:
    pip install twilio
    Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER env vars.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> None:
    try:
        from twilio.rest import Client
    except ImportError:
        print("Install twilio: pip install 'twilio>=9.0'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Make a test call via Twilio")
    parser.add_argument(
        "--to", required=True, help="Destination phone number (e.g. +919876543210)"
    )
    parser.add_argument("--base-url", default="", help="Your server's public URL")
    args = parser.parse_args()

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER", "")

    if not all([account_sid, auth_token, from_number]):
        print("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
        sys.exit(1)

    base_url = args.base_url or os.environ.get("VOICE_STATUS_CALLBACK_URL", "").rsplit("/", 1)[0]
    if not base_url:
        print("Set --base-url or VOICE_STATUS_CALLBACK_URL env var")
        sys.exit(1)

    client = Client(account_sid, auth_token)

    print(f"Calling {args.to} from {from_number}...")
    call = client.calls.create(
        to=args.to,
        from_=from_number,
        url=f"{base_url}/voice/incoming",
        method="POST",
        status_callback=f"{base_url}/voice/status",
        status_callback_method="POST",
    )
    print(f"Call SID: {call.sid}")

    for i in range(30):
        time.sleep(2)
        call = client.calls(call.sid).fetch()
        print(f"  [{i * 2:3d}s] Status: {call.status}")
        if call.status in ("completed", "failed", "busy", "no-answer", "canceled"):
            break

    print(f"\nFinal: {call.status}, duration: {call.duration}s")


if __name__ == "__main__":
    main()
