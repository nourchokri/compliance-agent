"""Environment and API key validation for the compliance assistant."""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("swedish_compliance")

# Valid Gemini API key prefixes (Google is migrating AIza -> AQ.)
_VALID_KEY_PREFIXES = ("AIza", "AQ.")

# Detect accidental key duplication (paste twice in .env).
_DUPLICATE_KEY_PATTERN = re.compile(
    r"^(?P<first>(?:AIza|AQ\.)[A-Za-z0-9._-]+)(?:AIza|AQ\.).+$"
)


def get_api_key() -> str | None:
    """Return the configured Gemini API key, if any."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def validate_api_key(api_key: str | None = None) -> None:
    """Log warnings for common API key misconfiguration issues."""
    key = (api_key or get_api_key() or "").strip()
    if not key:
        logger.warning(
            "GOOGLE_API_KEY is not set. Add a key from "
            "https://aistudio.google.com/apikey to your .env file."
        )
        return

    if match := _DUPLICATE_KEY_PATTERN.match(key):
        logger.error(
            "GOOGLE_API_KEY appears duplicated in .env (pasted twice). "
            "Keep only one key value. Example length after fix: %d chars.",
            len(match.group("first")),
        )
        return

    if not key.startswith(_VALID_KEY_PREFIXES):
        logger.warning(
            "GOOGLE_API_KEY has an unexpected format. Gemini keys usually "
            "start with 'AIza' or 'AQ.'. Get a key at "
            "https://aistudio.google.com/apikey"
        )

    if len(key) < 20:
        logger.warning("GOOGLE_API_KEY looks too short to be valid.")
