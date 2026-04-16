#!/usr/bin/env python3
"""Setup Plivo: list / buy Indian phone numbers and configure webhooks.

Usage:
    python scripts/setup_plivo.py list          # Show available numbers
    python scripts/setup_plivo.py buy           # Buy first available number
    python scripts/setup_plivo.py configure     # Set webhook URLs on existing number

Requires PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, and (for configure) PLIVO_PHONE_NUMBER
in the environment or .env file.
"""

from __future__ import annotations

import os
import sys

try:
    import plivo
except ImportError:
    print("Error: plivo SDK not installed. Run: pip install plivo")
    sys.exit(1)


def _get_client() -> plivo.RestClient:
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        print("Error: Set PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN environment variables.")
        sys.exit(1)
    return plivo.RestClient(auth_id, auth_token)


def list_numbers() -> None:
    """List available Indian phone numbers for purchase."""
    client = _get_client()
    try:
        response = client.numbers.search(
            country_iso="IN",
            type="local",
            limit=10,
        )
        print(f"\nAvailable Indian phone numbers ({len(response)} found):\n")
        for num in response:
            print(f"  {num.number}  (region: {num.region}, monthly: ${num.monthly_rental_rate})")
    except plivo.exceptions.PlivoRestError as e:
        print(f"Error searching numbers: {e}")
        sys.exit(1)


def buy_number() -> None:
    """Buy the first available Indian phone number."""
    client = _get_client()
    try:
        numbers = client.numbers.search(country_iso="IN", type="local", limit=1)
        if not numbers:
            print("No numbers available. Try a different region or type.")
            sys.exit(1)

        number = numbers[0].number
        print(f"Buying number: {number}")

        response = client.numbers.buy(number)
        print(f"Purchased: {number}")
        print(f"  Status: {response.status}")
        print(f"\nSet PLIVO_PHONE_NUMBER={number} in your .env file.")
    except plivo.exceptions.PlivoRestError as e:
        print(f"Error buying number: {e}")
        sys.exit(1)


def configure_webhooks() -> None:
    """Configure webhook URLs on the Plivo phone number."""
    client = _get_client()
    phone_number = os.getenv("PLIVO_PHONE_NUMBER", "")
    base_url = os.getenv("VOICE_WEBSOCKET_URL", "").replace("wss://", "https://").rstrip("/")

    if not phone_number:
        print("Error: Set PLIVO_PHONE_NUMBER environment variable.")
        sys.exit(1)

    if not base_url:
        print("Error: Set VOICE_WEBSOCKET_URL environment variable.")
        sys.exit(1)

    # Derive the incoming call webhook from the WebSocket URL's base
    # e.g., wss://example.com/voice/stream -> https://example.com/voice/incoming
    incoming_url = base_url.rsplit("/", 1)[0] + "/incoming"
    status_url = base_url.rsplit("/", 1)[0] + "/status"

    print(f"Configuring number: {phone_number}")
    print(f"  Answer URL: {incoming_url}")
    print(f"  Status URL: {status_url}")

    try:
        client.numbers.update(
            phone_number,
            answer_url=incoming_url,
            answer_method="POST",
            hangup_url=status_url,
            hangup_method="POST",
        )
        print("\nWebhooks configured successfully.")
    except plivo.exceptions.PlivoRestError as e:
        print(f"Error configuring webhooks: {e}")
        sys.exit(1)


def main() -> None:
    # Load .env if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    match command:
        case "list":
            list_numbers()
        case "buy":
            buy_number()
        case "configure":
            configure_webhooks()
        case _:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)


if __name__ == "__main__":
    main()
