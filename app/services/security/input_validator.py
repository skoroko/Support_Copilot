from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass
from typing import Final

MAX_INPUT_CHARS: Final[int] = 4000
MAX_MESSAGE_CHARS: Final[int] = 16_000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10
ATYPICAL_UNICODE_RATIO_LIMIT: Final[float] = 0.35
DECODED_PRINTABLE_RATIO_LIMIT: Final[float] = 0.9


INJECTION_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|system)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+|any\s+)?previous\s+and\s+following\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(the\s+)?(system|previous|above)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a|an|the|dan|do anything now)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|previous|your rules)\b", re.IGNORECASE),
    re.compile(r"\b(jailbroken|developer mode|godmode|do anything now|DAN)\b", re.IGNORECASE),
    re.compile(r"\b(reveal|print|repeat|show)\s+(the\s+)?(system prompt|hidden instructions|developer message)\b", re.IGNORECASE),
    re.compile(r"\bstop\s+everything\b.*\b(just\s+)?(print|say)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bno\s+takesies\s+backsies\b", re.IGNORECASE),
]

# Opaque alphabet runs long enough to hide an instruction phrase. They are
# decoded and inspected instead of blocked on sight, so license keys, hashes
# and error signatures pasted into tickets stay allowed.
_ENCODED_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9+/_-]{20,}={0,2}")
_BASE64_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_BASE64URL_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+={0,2}")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    rule: str | None = None


def validate_input(text: str, *, max_chars: int = MAX_INPUT_CHARS) -> ValidationResult:
    normalized = unicodedata.normalize("NFKC", text)

    if len(normalized) > max_chars:
        return ValidationResult(False, "input too long", rule="length")

    non_printable = sum(1 for char in normalized if not char.isprintable() and char not in "\n\r\t")
    if non_printable / max(len(normalized), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return ValidationResult(False, "high non-printable ratio", rule="encoding")

    if _atypical_unicode_ratio(normalized) > ATYPICAL_UNICODE_RATIO_LIMIT:
        return ValidationResult(False, "atypical unicode ratio", rule="encoding")

    for pattern in INJECTION_PATTERNS:
        if pattern.search(normalized):
            return ValidationResult(False, "prompt injection marker", rule="injection")

    if _encoded_injection(normalized):
        return ValidationResult(False, "encoded prompt injection marker", rule="encoded_injection")

    return ValidationResult(True)


def decoded_fragments(text: str) -> list[str]:
    """Readable text recovered from opaque base64/base64url tokens.

    Callers feed the fragments to moderation so an encoded harmful request is
    judged by its decoded content rather than slipping through as opaque.
    """

    normalized = unicodedata.normalize("NFKC", text)
    fragments: list[str] = []
    for match in _ENCODED_TOKEN_RE.finditer(normalized):
        decoded = _decode_token(match.group(0))
        if decoded is not None:
            fragments.append(decoded)
    return fragments


def _encoded_injection(text: str) -> bool:
    if any(pattern.search(codecs.decode(text, "rot13")) for pattern in INJECTION_PATTERNS):
        return True
    return any(
        pattern.search(fragment)
        for fragment in decoded_fragments(text)
        for pattern in INJECTION_PATTERNS
    )


def _decode_token(token: str) -> str | None:
    padded = token + "=" * (-len(token) % 4)
    raw = _b64decode(padded) or _b64urldecode(padded)
    if not raw:
        return None
    decoded = raw.decode("utf-8", errors="ignore")
    if not decoded:
        return None
    printable = sum(1 for char in decoded if char.isprintable() or char in "\t\n")
    if printable / len(decoded) < DECODED_PRINTABLE_RATIO_LIMIT:
        return None
    return decoded


def _b64decode(padded: str) -> bytes | None:
    if not _BASE64_TOKEN_RE.fullmatch(padded):
        return None
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _b64urldecode(padded: str) -> bytes | None:
    if not _BASE64URL_TOKEN_RE.fullmatch(padded):
        return None
    try:
        return base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError):
        return None


def _atypical_unicode_ratio(text: str) -> float:
    relevant = [char for char in text if not char.isspace() and not char.isdigit() and not unicodedata.category(char).startswith("P")]
    if not relevant:
        return 0.0

    atypical = 0
    for char in relevant:
        name = unicodedata.name(char, "")
        if any(block in name for block in ("LATIN", "CYRILLIC")):
            continue
        if unicodedata.category(char).startswith(("S", "M")):
            atypical += 1
            continue
        if ord(char) > 0x2FFF:
            atypical += 1
    return atypical / len(relevant)
