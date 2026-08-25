"""Safe preservation of operator-facing error details."""

from __future__ import annotations

import json
import re
from typing import Any

_URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"Bearer\s+\S+", flags=re.IGNORECASE)
_LICENSE_PATTERN = re.compile(r"ONPRM(?:-[0-9A-Z]{5}){5}")
_DETAIL_FIELDS = ("detail", "message", "error")


def sanitize_error_detail(value: object, maximum: int = 1000) -> str:
    """Retain diagnostic text while removing credentials and signed URLs."""
    message = str(value or "unspecified failure")
    message = _URL_PATTERN.sub("[URL REDACTED]", message)
    message = _BEARER_PATTERN.sub("Bearer [REDACTED]", message)
    message = _LICENSE_PATTERN.sub("[LICENSE REDACTED]", message)
    return message[:maximum]


def extract_http_error_detail(body: bytes, maximum: int = 1000) -> str | None:
    """Extract bounded operator detail from a JSON or plain-text HTTP response."""
    if not body:
        return None
    decoded = body[:maximum].decode("utf-8", errors="replace").strip()
    if not decoded:
        return None
    try:
        payload: Any = json.loads(decoded)
    except json.JSONDecodeError:
        return sanitize_error_detail(decoded, maximum)
    if isinstance(payload, dict):
        for field_name in _DETAIL_FIELDS:
            field_value = payload.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                return sanitize_error_detail(field_value.strip(), maximum)
    return sanitize_error_detail(decoded, maximum)
