"""Metadata for local CLI logs without retaining request or response content."""

from __future__ import annotations

import hashlib
import hmac
import secrets

# An ephemeral process key prevents offline guessing of short support requests.
# It is deliberately never written beside the log or loaded from configuration.
_LOG_DIGEST_KEY = secrets.token_bytes(32)


def content_metadata(value: str) -> str:
    digest = hmac.new(_LOG_DIGEST_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"chars={len(value)} digest={digest}"


def content_fields(name: str, value: str) -> dict[str, int | str]:
    digest = hmac.new(_LOG_DIGEST_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return {f"{name}_chars": len(value), f"{name}_digest": digest}
