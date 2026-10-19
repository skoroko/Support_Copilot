"""Detect incomplete prose without rewriting technical identifiers.

Also owns the answer post-processing contract: citation normalization,
grounding against the retrieved source list and abstention recognition.
"""

from __future__ import annotations

import re
from typing import Any

from app.observability.logging import get_logger

logger = get_logger("answer-quality")

NO_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."

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


CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Escalation/recommendation verbs allowed inside an abstention answer.
ABSTENTION_ADJACENT_RE = re.compile(
    r"обратит|уточнит|проверь|проверя|рекоменд|инженер|специалист|эскалац",
    re.IGNORECASE,
)
# Alternative citation shapes providers emit; all are normalized to canonical
# ``[N]`` before grounding so the citation contract stays single-format.
CJK_CITATION_RE = re.compile(r"【(\d+(?:\s*[,;、]\s*\d+)*)\s*(?:†[^】\d]*)?】")
FOOTNOTE_CITATION_RE = re.compile(r"\[\^(\d+)\]")
SOURCE_WORD_CITATION_RE = re.compile(
    r"\[?(\s*\b(?:источник\w*|sources?)\s*[:№]?\s*)(\d+(?:\s*(?:,|;|или|и|and|or)\s*\d+)*)\s*\]?",
    re.IGNORECASE,
)
# ``(1)`` doubles as list enumeration, so only markers glued to same-line
# text (non-space character plus optional spaces/tabs) qualify, and only as
# a fallback when the answer has no canonical citations at all.
PAREN_CITATION_RE = re.compile(r"([^\s\n(][ \t]*)\((\d+(?:\s*,\s*\d+)*)\)")


def valid_cited_numbers(text: str, source_count: int) -> list[int]:
    return sorted(
        {
            number
            for match in CITATION_RE.finditer(text)
            for raw_number in match.group(1).split(",")
            if 1 <= (number := int(raw_number.strip())) <= source_count
        }
    )


def alt_citation_replacement(
    match: re.Match[str],
    source_count: int,
    *,
    numbers_group: int,
    prefix_group: int | None = None,
) -> str:
    valid = [
        number
        for number in dict.fromkeys(
            int(value) for value in re.findall(r"\d+", match.group(numbers_group))
        )
        if 1 <= number <= source_count
    ]
    if not valid:
        return match.group(0)
    prefix = match.group(prefix_group) if prefix_group is not None else ""
    return f"{prefix}[{', '.join(str(number) for number in valid)}]"


def normalize_citation_formats(answer: str, source_count: int) -> str:
    """Rewrite unambiguous non-canonical citation shapes as ``[N]`` markers.

    ``【1】``, ``[^1]`` and word forms (``источник 1``, ``sources 1 и 2``)
    only ever reference context fragments, so they are converted whenever
    at least one of their numbers points at an existing source.  A match
    whose numbers are all out of range is left untouched: there it is
    more likely prose than a citation.
    """

    normalized = CJK_CITATION_RE.sub(
        lambda match: alt_citation_replacement(match, source_count, numbers_group=1),
        answer,
    )
    normalized = FOOTNOTE_CITATION_RE.sub(
        lambda match: alt_citation_replacement(match, source_count, numbers_group=1),
        normalized,
    )
    return SOURCE_WORD_CITATION_RE.sub(
        lambda match: alt_citation_replacement(
            match, source_count, numbers_group=2, prefix_group=1
        ),
        normalized,
    )


def normalize_paren_citations(answer: str, source_count: int) -> str:
    """Convert mid-line ``(1)`` markers to ``[N]``.

    A parenthesized number at a line start is a list item rather than a
    citation, so only markers preceded by a non-space character qualify.
    Callers run this pass only when the answer has no canonical
    citations: inside a ``[N]``-compliant answer ``(1)`` is enumeration
    more often than not.
    """

    return PAREN_CITATION_RE.sub(
        lambda match: alt_citation_replacement(
            match, source_count, numbers_group=2, prefix_group=1
        ),
        answer,
    )


def is_abstention_answer(answer: str) -> bool:
    """Recognize grounded-generation abstentions and remove false citations.

    The exact canonical sentence is preferred, but providers sometimes
    paraphrase it as "the context does not contain pricing information"
    and append citations.  A scope marker is required so factual negative
    statements such as "the model does not include rotations" remain valid.
    Abstention is decided per sentence: one scoped "нет сведений о …"
    clause inside an otherwise substantive answer must not discard the
    grounded parts — the answer counts as an abstention only when every
    sentence either reports missing data or is a short escalation remark.
    """

    without_citations = CITATION_RE.sub("", answer).strip()
    if without_citations == NO_ANSWER:
        return True
    normalized = " ".join(without_citations.casefold().replace("ё", "е").split())
    if len(normalized) > 500:
        return False
    subject = r"(?:сведен\w*|информац\w*|данн\w*|цен\w*|стоимост\w*)"
    absence = r"(?:нет|не\s+(?:найден\w*|содерж\w*|указан\w*)|отсутств\w*)"
    scope_markers = (
        "базе знаний",
        "в базе",
        "в контексте",
        "предоставленном контексте",
        "в документе",
        "в документах",
        "в материалах",
    )

    def absence_claim(text: str, *, require_scope: bool) -> bool:
        if require_scope and not any(marker in text for marker in scope_markers):
            return False
        return bool(
            re.search(rf"{subject}[^.!?]{{0,120}}{absence}", text)
            or re.search(rf"{absence}[^.!?]{{0,120}}{subject}", text)
        )

    saw_absence = False
    for raw_sentence in re.split(r"[.!?…]+", without_citations):
        sentence = " ".join(raw_sentence.casefold().replace("ё", "е").split())
        if not sentence:
            continue
        if absence_claim(sentence, require_scope=not saw_absence):
            saw_absence = True
            continue
        # Short escalation hints keep an abstention an abstention; any
        # other substantive sentence means useful content worth keeping.
        if ABSTENTION_ADJACENT_RE.search(sentence) and len(sentence) < 160:
            continue
        return False
    return saw_absence


def ensure_citations(answer: str, sources: list[dict[str, Any]]) -> str:
    """Keep only citations the model actually used in its answer.

    Retrieval may provide ten candidates, but appending references for all
    of them falsely implies that each one supports the answer.  Telegram
    later resolves the citations that are already present in ``answer``.
    """

    source_count = len(sources)

    def keep_valid(match: re.Match[str]) -> str:
        valid: list[int] = []
        for raw_number in match.group(1).split(","):
            number = int(raw_number.strip())
            if 1 <= number <= source_count and number not in valid:
                valid.append(number)
        return f"[{', '.join(str(number) for number in valid)}]" if valid else ""

    return CITATION_RE.sub(keep_valid, answer)


def ground_answer(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Validate citations, return only cited sources and keep numbering aligned.

    Non-canonical citation shapes are normalized to ``[N]`` first, and an
    answer that still has no usable citation is delivered without a source
    list instead of being masked as a knowledge-base miss: the generation
    prompt already restricts it to the retrieved context, while refusing
    made correct answers indistinguishable from empty retrieval.
    """

    source_count = len(sources)
    normalized = normalize_citation_formats(answer, source_count)
    if is_abstention_answer(normalized):
        return NO_ANSWER, []
    cited_numbers = valid_cited_numbers(normalized, source_count)
    if not cited_numbers:
        normalized = normalize_paren_citations(normalized, source_count)
        cited_numbers = valid_cited_numbers(normalized, source_count)
    if not cited_numbers:
        cleaned = CITATION_RE.sub("", normalized).strip()
        if not cleaned:
            return NO_ANSWER, []
        logger.warning(
            "rag_answer_missing_citations",
            answer_chars=len(cleaned),
            source_count=source_count,
        )
        return cleaned, []

    number_map = {old: new for new, old in enumerate(cited_numbers, start=1)}

    def remap(match: re.Match[str]) -> str:
        mapped: list[int] = []
        for raw_number in match.group(1).split(","):
            new_number = number_map.get(int(raw_number.strip()))
            if new_number is not None and new_number not in mapped:
                mapped.append(new_number)
        return f"[{', '.join(str(number) for number in mapped)}]" if mapped else ""

    grounded_answer = CITATION_RE.sub(remap, normalized)
    cited_sources = [sources[number - 1] for number in cited_numbers]
    return grounded_answer, cited_sources
