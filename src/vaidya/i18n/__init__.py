"""Internationalisation module for Vaidya UI strings.

All user-facing messages are stored as JSON files in ``strings/`` and
accessed via :func:`get_msg`.  Fallback chain: requested language →
``hi-IN`` → ``en-IN`` → message key.
"""

from vaidya.i18n.messages import get_msg, get_msg_template, list_keys

__all__ = ["get_msg", "get_msg_template", "list_keys"]
