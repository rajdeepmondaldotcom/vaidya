"""i18n message loader and accessor.

Messages are flat JSON files in ``strings/``, keyed by message name, with
language-code sub-keys::

    {
      "welcome": {
        "hi-IN": "Namaste! ...",
        "en-IN": "Hello! ..."
      }
    }

The :func:`get_msg` function resolves a message with a fallback chain:
requested language → ``hi-IN`` → ``en-IN`` → the key name itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_STRINGS_DIR = Path(__file__).parent / "strings"

# Module-level cache: domain -> {key -> {lang -> text}}
_CACHE: dict[str, dict[str, dict[str, str]]] = {}

_FALLBACK_CHAIN = ("hi-IN", "en-IN")


def _load_domain(domain: str) -> dict[str, dict[str, str]]:
    """Load a domain JSON file into the cache."""
    if domain in _CACHE:
        return _CACHE[domain]

    path = _STRINGS_DIR / f"{domain}.json"
    if not path.exists():
        logger.warning("i18n domain file not found: %s", path)
        _CACHE[domain] = {}
        return _CACHE[domain]

    with path.open(encoding="utf-8") as f:
        raw: Any = json.load(f)

    data = cast(dict[str, dict[str, str]], raw)
    _CACHE[domain] = data
    return data


def get_msg(domain: str, key: str, lang: str) -> str:
    """Look up a message string.

    Parameters
    ----------
    domain:
        JSON filename without extension (e.g. ``"orchestrator"``).
    key:
        Message key within the domain (e.g. ``"welcome"``).
    lang:
        BCP-47 language code (e.g. ``"hi-IN"``).

    Returns
    -------
    str
        The resolved message, or *key* if nothing is found.
    """
    messages = _load_domain(domain)
    entry = messages.get(key)
    if entry is None:
        logger.warning("i18n key not found: %s.%s", domain, key)
        return key

    # Try requested language, then fallback chain
    text = entry.get(lang)
    if text is not None:
        return text

    for fallback in _FALLBACK_CHAIN:
        text = entry.get(fallback)
        if text is not None:
            return text

    logger.warning("i18n no translation for %s.%s in %s", domain, key, lang)
    return key


def get_msg_template(domain: str, key: str, lang: str, **kwargs: object) -> str:
    """Look up a message and format it with ``str.format(**kwargs)``.

    Useful for messages with placeholders like ``{state}`` or ``{count}``.
    """
    template = get_msg(domain, key, lang)
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template


def list_keys(domain: str) -> list[str]:
    """Return all message keys in a domain (for testing)."""
    messages = _load_domain(domain)
    return list(messages.keys())


def reload() -> None:
    """Clear the cache so domains are re-loaded on next access."""
    _CACHE.clear()
