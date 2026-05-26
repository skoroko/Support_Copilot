"""Lightweight deterministic classification used by CLI demos and tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Classification:
    category: str
    priority: str
    summary: str
    escalate: bool
    reason: str


PROMPT_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all",
    "system prompt",
    "developer message",
    "раскрой систем",
    "покажи систем",
    "игнорируй инструк",
    "забудь правила",
)

COMPLAINT_MARKERS = (
    "недоволен",
    "жалоба",
    "плохо работает",
    "никто не отвечает",
    "возмущен",
    "претенз",
)

ESCALATION_MARKERS = (
    "эскалац",
    "позови специалист",
    "передай специалист",
    "срочно",
    "критично",
    "блокирует",
    "не работает у всех",
    "массов",
)

TECH_MARKERS = (
    "ошибка",
    "error",
    "503",
    "500",
    "timeout",
    "лог",
    "crash",
    "не запускается",
    "не открывается",
    "завис",
)

FAQ_MARKERS = (
    "как",
    "где",
    "инструкция",
    "документац",
    "установ",
    "лиценз",
    "настро",
)

OUT_OF_SCOPE_MARKERS = (
    "погода",
    "курс доллара",
    "рецепт",
    "анекдот",
    "фильм",
)


def classify_request(text: str) -> Classification:
    normalized = text.strip().lower()
    summary = _compact_summary(text)

    if _contains_any(normalized, PROMPT_INJECTION_MARKERS):
        return Classification(
            category="prompt_injection",
            priority="high",
            summary=summary,
            escalate=True,
            reason="попытка изменить правила ассистента или раскрыть внутренние инструкции",
        )

    if _contains_any(normalized, OUT_OF_SCOPE_MARKERS):
        return Classification(
            category="out_of_scope",
            priority="low",
            summary=summary,
            escalate=False,
            reason="вопрос не относится к техподдержке MIDAS",
        )

    if _contains_any(normalized, COMPLAINT_MARKERS):
        return Classification(
            category="complaint",
            priority="high",
            summary=summary,
            escalate=True,
            reason="негативное обращение требует внимания ответственного специалиста",
        )

    if _contains_any(normalized, ESCALATION_MARKERS):
        return Classification(
            category="escalation",
            priority="high",
            summary=summary,
            escalate=True,
            reason="пользователь явно просит специалиста или описывает блокирующий случай",
        )

    if _contains_any(normalized, TECH_MARKERS):
        return Classification(
            category="technical_issue",
            priority="medium",
            summary=summary,
            escalate="503" in normalized or "500" in normalized or "блокирует" in normalized,
            reason="нужны технические детали, логи и шаги воспроизведения",
        )

    if _contains_any(normalized, FAQ_MARKERS):
        return Classification(
            category="faq",
            priority="low",
            summary=summary,
            escalate=False,
            reason="типовой вопрос, вероятно решается инструкцией или базой знаний",
        )

    return Classification(
        category="unknown",
        priority="medium",
        summary=summary,
        escalate=False,
        reason="категория не определена, нужно уточнить контекст",
    )


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _compact_summary(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "Пустой запрос"
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
