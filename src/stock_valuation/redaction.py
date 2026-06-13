"""Secret redaction for public errors and diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_SECRET_NAME_PATTERN = re.compile(
    r"(?:^|[_-])(api[_-]?key|key|token|secret|password|passwd|credential)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|credential)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)


def redact_secrets(value: Any, *, field_name: str | None = None) -> Any:
    """Return a recursively redacted, JSON-compatible value.

    Parameters
    ----------
    value:
        Value that may contain provider credentials.
    field_name:
        Optional field name used to identify secret-bearing values.

    Returns
    -------
    Any
        Value with credentials and URL user information replaced.
    """

    if field_name is not None and _is_secret_name(field_name):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(key): redact_secrets(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        redacted = [redact_secrets(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _is_secret_name(name: str) -> bool:
    normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    return bool(_SECRET_NAME_PATTERN.search(normalized))


def _redact_string(value: str) -> str:
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
        value,
    )
    try:
        parsed = urlsplit(redacted)
    except ValueError:
        return redacted
    if parsed.scheme and parsed.hostname and (parsed.username or parsed.password):
        hostname = parsed.hostname
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit(
            (
                parsed.scheme,
                f"{REDACTED}@{hostname}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return redacted
