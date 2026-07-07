from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from functools import lru_cache
from typing import Any
from uuid import UUID

import tiktoken

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.core.config import Settings
from app.observability.logging import get_logger


DEFAULT_CONTEXT_TOKENS = 128_000
DEFAULT_RESPONSE_TOKENS = 500
DEFAULT_SAFETY_MARGIN_TOKENS = 500

logger = get_logger("chat-service")

ChatMLMessage = dict[str, str]


class ChatNotFoundError(ValueError):
    pass


def count_tokens(messages: Sequence[ChatMLMessage]) -> int:
    total = 2
    for message in messages:
        total += 4
        total += _count_text_tokens(message.get("role", ""))
        total += _count_text_tokens(message.get("content", ""))
    return total


@lru_cache(maxsize=1)
def _o200k_base_encoding() -> Any | None:
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # noqa: BLE001 - tiktoken may need a blocked first-time download
        logger.warning("tokenizer_load_failed", encoding="o200k_base", error_type=type(exc).__name__)
        return None


def _count_text_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _o200k_base_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, (len(text) + 3) // 4)


def fit_to_budget(messages: Sequence[ChatMLMessage], budget: int) -> list[ChatMLMessage]:
    if count_tokens(messages) <= budget:
        return list(messages)

    system_messages = [message for message in messages if message.get("role") == "system"]
    historic_messages = [message for message in messages if message.get("role") != "system"]
    while historic_messages and count_tokens([*system_messages, *historic_messages]) > budget:
        historic_messages.pop(0)
    return [*system_messages, *historic_messages]


class SlidingWindowContextStrategy:
    def __init__(self, history_limit: int) -> None:
        self.history_limit = history_limit

    def build(self, chat: Chat, history: Sequence[ChatMessage]) -> list[ChatMLMessage]:
        messages: list[ChatMLMessage] = []
        if chat.system_prompt:
            messages.append({"role": "system", "content": chat.system_prompt})
        for message in history[-self.history_limit :]:
            messages.append({"role": message.role, "content": message.content})
        return messages


class ChatService:
    def __init__(self, repository: ChatRepository, llm_client: Any, settings: Settings) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.settings = settings
        if settings.chat_context_strategy != "sliding":
            raise ValueError("Only CHAT_CONTEXT_STRATEGY=sliding is implemented in homework 4.1")
        self.context_strategy = SlidingWindowContextStrategy(settings.chat_context_window)

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        return await self.repository.create_chat(owner_external_id, interface, system_prompt)

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return await self.repository.list_messages(chat_id, limit)

    async def send_message(self, chat_id: UUID, user_content: str) -> AsyncIterator[str]:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            tokens=count_tokens([{"role": "user", "content": user_content}]),
        )
        await self.repository.append_message(chat_id, user_message)

        history = await self.repository.list_messages(chat_id, self.settings.chat_context_window)
        messages = self.context_strategy.build(chat, history)
        budget = (
            self.settings.chat_model_context_tokens
            - self.settings.chat_response_tokens
            - self.settings.chat_safety_margin_tokens
        )
        messages = fit_to_budget(messages, max(1, budget))

        assistant_parts: list[str] = []
        completed = False
        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.settings.llm.default_model,
                messages=messages,
                stream=True,
                temperature=0.2,
                **self._token_limit_kwargs(self.settings.llm.default_model, self.settings.chat_response_tokens),
            )
            async for chunk in stream:
                content = self._chunk_content(chunk)
                if not content:
                    continue
                assistant_parts.append(content)
                yield content
            completed = True
        finally:
            if assistant_parts:
                assistant_content = "".join(assistant_parts)
                if not completed:
                    logger.warning("chat_stream_interrupted", chat_id=str(chat_id))
                await self.repository.append_message(
                    chat_id,
                    ChatMessage(
                        chat_id=chat_id,
                        role="assistant",
                        content=assistant_content,
                        tokens=count_tokens([{"role": "assistant", "content": assistant_content}]),
                    ),
                )

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)

    def _token_limit_kwargs(self, model: str, max_tokens: int) -> dict[str, int]:
        if model.startswith("gpt-5"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _chunk_content(self, chunk: Any) -> str:
        choices = chunk.get("choices", []) if isinstance(chunk, dict) else getattr(chunk, "choices", [])
        if not choices:
            return ""
        first_choice = choices[0]
        delta = first_choice.get("delta") if isinstance(first_choice, dict) else getattr(first_choice, "delta", None)
        if delta is None:
            return ""
        content = delta.get("content", "") if isinstance(delta, dict) else getattr(delta, "content", "")
        return str(content or "")
