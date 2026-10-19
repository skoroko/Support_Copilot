"""Detect incomplete prose without rewriting technical identifiers."""

from __future__ import annotations

import re

ANSWER_QUALITY_UNAVAILABLE = (
    "Не удалось сформировать полный ответ без ошибок текста. "
    "Попробуйте уточнить вопрос или передайте его инженеру поддержки."
)
INCOMPLETE_NOTICE = (
    "Ответ сокращён по лимиту длины. "
    "Расширенный ответ можно запросить кнопкой «Подробнее»."
)


def incomplete_notice(finish_reason: str | None, *, expandable: bool) -> str:
    """Describe generation limits separately from missing user information."""
    if finish_reason == "length" and expandable:
        return INCOMPLETE_NOTICE
    reason = (
        "Ответ сокращён по лимиту длины."
        if finish_reason == "length"
        else "Из ответа удалён фрагмент с ошибкой текста."
    )
    action = (
        "Расширенный ответ можно запросить кнопкой «Подробнее»."
        if expandable
        else "Для продолжения задайте отдельный вопрос о нужном пункте."
    )
    return reason + " " + action

# Identifiers and literal source/code excerpts are not natural-language prose.
_LITERALS = re.compile(
    r"```.*?(?:```|$)|`[^`\n]*`|https?://\S+|\b[\w.+-]+@[\w.-]+\b"
    r"|\b[\w.-]+\.(?:pdf|chm|txt|json|py|md|csv|xlsx|exe|dll)\b",
    re.DOTALL | re.IGNORECASE,
)
_WORDS = re.compile(r"\b[^\W\d_]+\b")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD_RUNS = re.compile(r"[A-Za-z]+|[А-Яа-яЁё]+")
# Technical notation legitimately fuses scripts: a Latin SI prefix joins a
# Cyrillic unit base (``kПа``, ``kВ``, ``MПa``) or a Cyrillic prefix precedes a
# Latin unit (``кPa``, ``мHz``).  Corrupted prose instead mixes scripts inside
# longer words (``программa``), which must still be flagged.
_SI_LATIN_PREFIXES = frozenset({"k", "m", "g", "t", "p", "e", "z", "y", "n", "u", "d", "c", "h", "da", "mu", "µ"})
_CYRILLIC_PREFIXES = frozenset({"к", "м", "г", "т", "п", "е", "з", "и", "н", "ц", "д", "мк"})
_LATIN_UNITS = frozenset({
    "pa", "n", "w", "v", "a", "hz", "j", "l", "g", "m", "s", "k", "bar",
    "rad", "ohm", "min", "h", "wh", "var", "kn", "kw", "kv", "kg", "tf",
    "mw", "kj", "nm", "db", "rpm", "mph", "kpa", "mpa",
})
_CYRILLIC_UNITS = frozenset({
    "па", "н", "вт", "в", "а", "ом", "гц", "дж", "л", "г", "м", "с", "к",
    "бар", "рад", "мин", "ч", "т", "кгс", "тс", "мм", "см", "км", "кн",
    "мн", "мпа", "кпа", "гпа", "кв", "квт", "мвт", "кдж", "ккал", "мг",
    "мл", "мкм", "нм", "кг", "сут", "об",
})
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+")
_SENTENCES = re.compile(
    r".*?(?:[!?…]|\.(?!\d))(?:[ \t]*\[\d+(?:\s*,\s*\d+)*\])*(?:\*\*)?(?=\s|$)"
)


def _is_heading(line: str) -> bool:
    """Recognize structural labels, never an isolated numbering marker."""
    label = _LIST_PREFIX.sub("", line).strip()
    if re.fullmatch(r"\d+[.)]", label):
        return False
    if re.match(r"^#{1,6}\s+\S", label):
        return True
    return bool(re.fullmatch(
        r"(?:\*\*[^*\n]+\*\*|`[^`\n]+`|[A-Za-z][A-Za-z0-9_{}^]*)"
        r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])*:?",
        label,
    ))


def _is_technical_mixed_word(word: str) -> bool:
    """Distinguish fused units/notation (``kПа``, ``кPa``) from mojibake."""
    if len(word) <= 3:
        return True
    runs = _WORD_RUNS.findall(word)
    if len(runs) != 2:
        return False
    first, second = runs
    if _LATIN.fullmatch(first):
        return first.lower() in _SI_LATIN_PREFIXES and second.lower() in _CYRILLIC_UNITS
    return first.lower() in _CYRILLIC_PREFIXES and second.lower() in _LATIN_UNITS


def answer_quality_issue(text: str, finish_reason: str | None = None) -> str | None:
    if finish_reason == "length":
        return "length"
    if not text.strip():
        return "empty"
    prose = _LITERALS.sub(" ", text)
    if any(
        _CYRILLIC.search(word) and _LATIN.search(word) and not _is_technical_mixed_word(word)
        for word in _WORDS.findall(prose)
    ):
        return "mixed_script"
    return None


def compact_answer(text: str, finish_reason: str | None = None) -> tuple[str, bool]:
    """Keep complete, clean prose deterministically, without another model call.

    Technical literals are inspected as opaque spans. On a length stop the
    trailing unfinished sentence/list item is removed, including its Markdown.
    """
    issue = answer_quality_issue(text, finish_reason)
    if issue is None:
        return text, False
    literals = []

    def protect(match):
        literal = match.group()
        suffix = ""
        if literal.startswith(("https://", "http://")):
            trimmed = literal.rstrip(".,!?;")
            suffix, literal = literal[len(trimmed):], trimmed
        literals.append(literal)
        return f"\x00{len(literals) - 1}\x00" + suffix

    # Protect globally so a multiline fenced block stays one opaque span.
    protected = _LITERALS.sub(protect, text)
    kept = []
    pending_heading = None
    for line in protected.splitlines():
        restored = line
        for index, literal in enumerate(literals):
            restored = restored.replace(f"\x00{index}\x00", literal)
        if _is_heading(restored):
            # A label is only meaningful together with retained following body.
            pending_heading = restored.strip() if answer_quality_issue(restored) is None else None
            continue
        prefix = _LIST_PREFIX.match(line)
        body = line[prefix.end():] if prefix else line
        sentences = _SENTENCES.findall(body)
        table_row = body.strip().startswith("|") and body.strip().endswith("|")
        if table_row and not re.fullmatch(r"[|:\-\s]+", body):
            sentences = [body]
        if re.fullmatch(r"\s*\x00\d+\x00\s*", line):
            literal = literals[int(line.strip(" \t\x00"))]
            if literal.startswith("```") and literal.endswith("```"):
                sentences = [line]
        clean = []
        for sentence in sentences:
            if answer_quality_issue(sentence) is not None:
                continue
            # Literal spans are opaque: exponentiation/code is not Markdown.
            if sentence.count("**") % 2 or sentence.count("`") % 2:
                continue
            for index, literal in enumerate(literals):
                sentence = sentence.replace(f"\x00{index}\x00", literal)
            terminal = re.sub(r"(?:\s*\[\d+(?:\s*,\s*\d+)*\]|\*\*)+$", "", sentence.strip())
            if not table_row and not terminal.endswith((".", "!", "?", "…", "```")):
                continue
            if re.fullmatch(r"\d+[.)]", terminal):
                continue
            clean.append(sentence.strip())
        if clean:
            if pending_heading is not None:
                kept.append(pending_heading)
                pending_heading = None
            kept.append((prefix.group() if prefix else "") + " ".join(clean))
        elif line.strip():
            # Do not attach an orphan label to an unrelated later paragraph.
            pending_heading = None
    return "\n".join(kept).strip() or ANSWER_QUALITY_UNAVAILABLE, True
