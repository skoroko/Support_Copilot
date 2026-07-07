from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final


MAX_INPUT_CHARS: Final[int] = 4000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10
ATYPICAL_UNICODE_RATIO_LIMIT: Final[float] = 0.35


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

ENCODING_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(base64|rot13|morse|decode this|atob\(|btoa\()", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{6,}={1,2}(?![A-Za-z0-9+/_=-])"),
    re.compile(r"\b[A-Za-z0-9+/]{48,}={0,2}\b"),
]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    rule: str | None = None


def validate_input(text: str) -> ValidationResult:
    normalized = unicodedata.normalize("NFKC", text)

    if len(normalized) > MAX_INPUT_CHARS:
        return ValidationResult(False, "input too long", rule="length")

    non_printable = sum(1 for char in normalized if not char.isprintable() and char not in "\n\r\t")
    if non_printable / max(len(normalized), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return ValidationResult(False, "high non-printable ratio", rule="encoding")

    if _atypical_unicode_ratio(normalized) > ATYPICAL_UNICODE_RATIO_LIMIT:
        return ValidationResult(False, "atypical unicode ratio", rule="encoding")

    for pattern in ENCODING_PATTERNS:
        if pattern.search(normalized):
            return ValidationResult(False, "encoded prompt injection marker", rule="encoding")

    for pattern in INJECTION_PATTERNS:
        if pattern.search(normalized):
            return ValidationResult(False, "prompt injection marker", rule="injection")

    return ValidationResult(True)


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
