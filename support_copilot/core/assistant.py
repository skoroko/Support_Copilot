"""Assistant orchestration helpers."""

from __future__ import annotations

from .classification import Classification
from ..prompts import PROMPT_VERSION, build_system_prompt


def build_chat_messages(
    user_text: str,
    history: list[dict[str, str]],
    *,
    include_few_shot: bool = True,
    history_limit: int = 10,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(include_few_shot)}]
    messages.extend(history[-history_limit:])
    messages.append({"role": "user", "content": user_text})
    return messages


def format_assistant_reply(
    *,
    classification: Classification,
    draft: str,
    checks: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    escalation = (
        f"Да: {classification.reason}"
        if classification.escalate
        else f"Нет: {classification.reason}"
    )

    return (
        f"1. Категория обращения: {classification.category} "
        f"(priority={classification.priority}).\n"
        f"2. Краткое резюме: {classification.summary}\n"
        f"3. Черновик ответа пользователю: {draft}\n"
        f"4. Что проверить инженеру: {checks}\n"
        f"5. Эскалация: {escalation}\n"
        f"\nPrompt version: {prompt_version}"
    )
