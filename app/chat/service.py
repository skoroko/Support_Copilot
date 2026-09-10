from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from functools import lru_cache
import inspect
import time
from typing import Any
from uuid import UUID

import tiktoken

from app.chat.domain import Chat, ChatMessage
from app.chat.feedback import FeedbackService, FeedbackTargetNotFoundError, FeedbackValue, MessageFeedback
from app.chat.media import media_to_part
from app.chat.repository import ChatRepository
from app.core.config import Settings
from app.moderation.models import ModerationResult
from app.moderation.service import ModerationService
from app.observability.logging import get_logger


DEFAULT_CONTEXT_TOKENS = 128_000
DEFAULT_RESPONSE_TOKENS = 500
DEFAULT_SAFETY_MARGIN_TOKENS = 500
MODERATION_OUTPUT_REPLACEMENT = "Не могу показать ответ — он мог нарушить правила"
RAG_EMPTY_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."

logger = get_logger("chat-service")

ChatMLMessage = dict[str, Any]


class ChatNotFoundError(ValueError):
    pass


class ModerationBlockedError(ValueError):
    def __init__(self, result: ModerationResult) -> None:
        super().__init__("Message blocked by moderation")
        self.result = result


def count_tokens(messages: Sequence[ChatMLMessage]) -> int:
    total = 2
    for message in messages:
        total += 4
        total += _count_text_tokens(message.get("role", ""))
        total += _count_content_tokens(message.get("content", ""))
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


def _count_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return _count_text_tokens(content)
    if not isinstance(content, list):
        return _count_text_tokens(str(content))

    total = 0
    for part in content:
        if not isinstance(part, dict):
            total += _count_text_tokens(str(part))
        elif part.get("type") == "text":
            total += _count_text_tokens(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            # Conservative base estimate; exact image tokens depend on model and dimensions.
            total += 255
    return total


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
            content: Any = message.content
            media_part = (message.media_refs or {}).get("part")
            if message.role == "user" and isinstance(media_part, dict):
                content = [
                    {"type": "text", "text": message.content},
                    media_part,
                ]
            messages.append({"role": message.role, "content": content})
        return messages


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_client: Any,
        settings: Settings,
        moderation_service: ModerationService | None = None,
    ) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.settings = settings
        self.moderation = moderation_service or ModerationService(
            keywords_path=settings.moderation_keywords_path,
            client=llm_client,
            enabled=settings.moderation_enabled,
            openai_enabled=settings.moderation_openai_enabled,
            model=settings.moderation_model,
            thresholds=settings.moderation_thresholds,
        )
        self.feedback = FeedbackService(repository)
        self._last_response_sources: dict[UUID, list[dict[str, Any]]] = {}
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

    async def process_media(self, media: Any) -> dict[str, Any]:
        return await media_to_part(
            media,
            client=self.llm_client,
            transcription_model=self.settings.transcription_model,
        )

    async def ensure_input_allowed(self, chat_id: UUID, content: str) -> ModerationResult:
        result = await self.moderation.check_input(content)
        await self._record_moderation(chat_id, "input", result)
        if not result.allowed:
            raise ModerationBlockedError(result)
        return result

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        *,
        media_refs: dict[str, Any] | None = None,
        input_moderated: bool = False,
        rag_service: Any | None = None,
    ) -> AsyncIterator[str]:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")

        if not input_moderated:
            moderation_content = self._moderation_content(user_content, media_refs)
            await self.ensure_input_allowed(chat_id, moderation_content)

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs=media_refs,
            tokens=count_tokens(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_content},
                            media_refs["part"],
                        ]
                        if media_refs and isinstance(media_refs.get("part"), dict)
                        else user_content,
                    }
                ]
            ),
        )
        await self.repository.append_message(chat_id, user_message)

        history = await self.repository.list_messages(chat_id, self.settings.chat_context_window)
        retrieval_history = history[:-1]
        messages = self.context_strategy.build(chat, history)
        budget = (
            self.settings.chat_model_context_tokens
            - self.settings.chat_response_tokens
            - self.settings.chat_safety_margin_tokens
        )
        messages = fit_to_budget(messages, max(1, budget))

        assistant_parts: list[str] = []
        rag_sources: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        if rag_service is not None:
            rag_result = await self._answer_with_rag(rag_service, user_content, retrieval_history)
            answer = str(rag_result.get("answer") or "").strip() or RAG_EMPTY_ANSWER
            assistant_parts = [answer]
            rag_sources = self._citation_array(rag_result.get("sources"))
        else:
            try:
                stream = await self.llm_client.chat.completions.create(
                    model=self.settings.llm.default_model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=0.2,
                    **self._token_limit_kwargs(self.settings.llm.default_model, self.settings.chat_response_tokens),
                )
                async for chunk in stream:
                    content = self._chunk_content(chunk)
                    if not content:
                        continue
                    assistant_parts.append(content)
            except Exception:
                if assistant_parts:
                    logger.warning("chat_stream_interrupted", chat_id=str(chat_id))
                raise

        generated_content = "".join(assistant_parts)
        output_result = await self.moderation.check_output(generated_content)
        await self._record_moderation(chat_id, "output", output_result)
        if output_result.allowed:
            visible_parts = assistant_parts
            visible_content = generated_content
            self._last_response_sources[chat_id] = rag_sources
        else:
            visible_parts = [MODERATION_OUTPUT_REPLACEMENT]
            visible_content = MODERATION_OUTPUT_REPLACEMENT
            self._last_response_sources[chat_id] = []

        if visible_content:
            await self.repository.append_message(
                chat_id,
                ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=visible_content,
                    rag_sources=rag_sources if output_result.allowed else [],
                    tokens=count_tokens([{"role": "assistant", "content": visible_content}]),
                    latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                ),
            )
        for part in visible_parts:
            yield part

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)

    async def append_system_message(self, chat_id: UUID, content: str) -> Chat:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")
        await self.repository.append_message(
            chat_id,
            ChatMessage(
                chat_id=chat_id,
                role="assistant",
                content=content,
                tokens=count_tokens([{"role": "assistant", "content": content}]),
            ),
        )
        return chat

    async def submit_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        value: FeedbackValue,
        shown_sources: list[dict[str, Any]] | None = None,
    ) -> MessageFeedback:
        return await self.feedback.save(chat_id, message_id, value, shown_sources)

    def last_response_sources(self, chat_id: UUID) -> list[dict[str, Any]]:
        return [dict(source) for source in self._last_response_sources.get(chat_id, [])]

    async def response_sources(self, chat_id: UUID, message_id: UUID) -> list[dict[str, Any]]:
        """Return the sources persisted with one assistant response.

        The Telegram callback carries compact chat/message identifiers, rather
        than source JSON, so it still works after either process restarts.
        """

        chat = await self.repository.get_chat(chat_id)
        message = await self.repository.get_message(chat_id, message_id)
        if chat is None or message is None or message.role != "assistant":
            raise FeedbackTargetNotFoundError(f"Assistant message {message_id} not found")
        return self._citation_array(message.rag_sources)

    async def _record_moderation(self, chat_id: UUID, direction: str, result: ModerationResult) -> None:
        recorder = getattr(self.repository, "record_moderation", None)
        if recorder is not None:
            await recorder(chat_id, direction, result)

    @staticmethod
    def _moderation_content(user_content: str, media_refs: dict[str, Any] | None) -> str:
        media_part = (media_refs or {}).get("part")
        if isinstance(media_part, dict) and media_part.get("type") == "text":
            return f"{user_content}\n{media_part.get('text', '')}".strip()
        return user_content

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

    async def _answer_with_rag(
        self,
        rag_service: Any,
        question: str,
        history: Sequence[ChatMessage],
    ) -> dict[str, Any]:
        answer = getattr(rag_service, "answer", None)
        if answer is None:
            raise TypeError("RAG service must expose an answer method")
        serialized_history = [message.model_dump(mode="json") for message in history]
        if inspect.iscoroutinefunction(answer):
            result = answer(question, history=serialized_history)
        else:
            result = await asyncio.to_thread(answer, question, history=serialized_history)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("RAG answer must be a mapping")
        return dict(result)

    @staticmethod
    def _citation_array(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(source) for source in value if isinstance(source, Mapping)]
