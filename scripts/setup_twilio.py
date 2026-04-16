#!/usr/bin/env python3
"""Configure a Twilio phone number for Vaidya voice calls.

Usage:
    # List available numbers
    python scripts/setup_twilio.py --list

    # Configure webhooks on your existing Twilio number
    python scripts/setup_twilio.py --configure --base-url https://your-app.up.railway.app

Requirements:
    pip install twilio
    Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN env vars.
"""

import argparse
import os
import sys


def main() -> None:
    try:
        from twilio.rest import Client
    except ImportError:
        print("Install twilio: pip install 'twilio>=9.0'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Configure Twilio for Vaidya")
    parser.add_argument("--list", action="store_true", help="List your Twilio phone numbers")
    parser.add_argument(
        "--configure", action="store_true", help="Configure webhooks on your number"
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Your server's public URL (e.g. https://xxx.up.railway.app)",
    )
    parser.add_argument(
        "--phone", default="", help="Twilio phone number to configure (e.g. +1234567890)"
    )
    args = parser.parse_args()

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")

    if not account_sid or not auth_token:
        print("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables")
        sys.exit(1)

    client = Client(account_sid, auth_token)

    if args.list:
        print("\nYour Twilio phone numbers:")
        numbers = client.incoming_phone_numbers.list()
        if not numbers:
            print("  (none -- buy one in the Twilio console)")
        for n in numbers:
            print(f"  {n.phone_number}  ({n.friendly_name})")
            print(f"    Voice URL: {n.voice_url or '(not set)'}")
        return

    if args.configure:
        if not args.base_url:
            print("--base-url required (e.g. https://your-app.up.railway.app)")
            sys.exit(1)

        base = args.base_url.rstrip("/")
        voice_url = f"{base}/voice/incoming"
        status_url = f"{base}/voice/status"

        phone = args.phone or os.environ.get("TWILIO_PHONE_NUMBER", "")
        if not phone:
            print("Specify --phone or set TWILIO_PHONE_NUMBER")
            sys.exit(1)

        # Find the number
        numbers = client.incoming_phone_numbers.list(phone_number=phone)
        if not numbers:
            print(f"Number {phone} not found in your account")
            sys.exit(1)

        number = numbers[0]
        number.update(
            voice_url=voice_url,
            voice_method="POST",
            status_callback=status_url,
            status_callback_method="POST",
        )
        print(f"\nConfigured {phone}:")
        print(f"  Answer URL:  {voice_url}")
        print(f"  Status URL:  {status_url}")
        print(f"\nCall {phone} to test!")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
