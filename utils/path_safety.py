from __future__ import annotations

import re

_SAFE_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_REQUEST_ID_LENGTH = 100


def sanitize_request_id(request_id: str) -> str:
    value = str(request_id or "").strip()
    if not value:
        raise ValueError("request_id cannot be empty")
    sanitized = _SAFE_REQUEST_ID_RE.sub("_", value)[:_MAX_REQUEST_ID_LENGTH]
    if not sanitized:
        raise ValueError("request_id cannot be empty after sanitization")
    return sanitized
