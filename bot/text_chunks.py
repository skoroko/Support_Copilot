"""Lossless Telegram-sized chunks, preferring prose and formatting boundaries."""

from __future__ import annotations

import re

_FORMATTED = re.compile(r"```.*?```|`[^`\n]+`|\*\*[^*\n]+\*\*", re.DOTALL)
_HEADING = re.compile(
    r"^(?:#{1,6}\s+.+|(?:\d+[.)]\s+)?(?:\*\*[^*]+\*\*|`[^`]+`|[A-Za-z][\w{}^]*):?)$"
)
_BREAKS = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"\n"),
    re.compile(r"[.!?…](?:\s*\[\d+(?:\s*,\s*\d+)*\])*\s+"),
    re.compile(r"\s+"),
)


def split_text(text: str, max_chars: int = 1800) -> list[str]:
    """Preserve every character, keeping labels with body whenever they fit.

    At 1800 Python characters even astral characters fit Telegram's UTF-16
    limit. Only an indivisible overlong span needs a hard character boundary.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    spans = [match.span() for match in _FORMATTED.finditer(text)]
    chunks = []
    start = 0
    while len(text) - start > max_chars:
        limit = start + max_chars
        boundary = None
        for pattern in _BREAKS:
            candidates = []
            for match in pattern.finditer(text, start, limit):
                end = match.end()
                if any(left < end < right for left, right in spans):
                    continue
                previous_lines = text[start:end].rstrip().splitlines()
                if previous_lines and _HEADING.fullmatch(previous_lines[-1].strip()):
                    continue
                candidates.append(end)
            if candidates:
                boundary = candidates[-1]
                break
        end = boundary or limit
        chunks.append(text[start:end])
        start = end
    chunks.append(text[start:])
    return chunks
