from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
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
RAG_MEDIA_TEXT_MAX_CHARS = 16_000
MEDIA_IMAGE_NO_TEXT_REPLY = (
    "Я пока не умею искать по базе знаний по изображению без описания. "
    "Добавьте подпись к фото или опишите проблему текстом."
)
# Used only when the chat has no custom system prompt and RAG cannot ground
# the answer; without it the fallback path streamed a bare LLM with no role.
FALLBACK_SYSTEM_PROMPT = (
    "Ты — Support Copilot, ассистент инженера технической поддержки MIDAS. "
    "Отвечай на русском языке. База знаний сейчас не подключена к ответу: "
    "полагайся только на проверенные общеизвестные сведения, прямо говори о "
    "нехватке данных и не выдумывай функции, цены, SLA, ссылки и регламенты "
    "MIDAS. Если уверенности нет, предложи передать обращение инженеру поддержки."
)

logger = get_logger("chat-service")

ChatMLMessage = dict[str, Any]


def media_parts(media_refs: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """All LLM content parts attached to a message.

    Multi-file uploads (Telegram albums) arrive as ``media_refs["parts"]``;
    single-file requests and older persisted messages carry only ``"part"``.
    """

    if not isinstance(media_refs, Mapping):
        return []
    parts = media_refs.get("parts")
    if isinstance(parts, list):
        return [part for part in parts if isinstance(part, dict)]
    part = media_refs.get("part")
    return [part] if isinstance(part, dict) else []


class ChatNotFoundError(ValueError):
    pass


class ModerationBlockedError(ValueError):
    def __init__(self, result: ModerationResult) -> None:
        super().__init__("Message blocked by moderation")
        self.result = result


@dataclass(frozen=True)
class StreamReplace:
    """Canonical answer that replaces every previously streamed delta.

    RAG deltas show the raw draft while it is generated; grounding and final
    output moderation can change the text, so consumers substitute ``text``
    for whatever they accumulated.
    """

    text: str


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
            parts = media_parts(message.media_refs)
            if message.role == "user" and parts:
                content = [
                    {"type": "text", "text": message.content},
                    *parts,
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
            max_bytes=self.settings.chat_media_max_bytes,
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
    ) -> AsyncIterator[str | StreamReplace]:
        chat = await self.repository.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(f"Chat {chat_id} not found")

        if not input_moderated:
            moderation_content = self._moderation_content(user_content, media_refs)
            await self.ensure_input_allowed(chat_id, moderation_content)

        parts = media_parts(media_refs)
        persisted_media_refs = media_refs
        if any(part.get("type") == "image_url" for part in parts):
            kept = [part for part in parts if part.get("type") != "image_url"]
            persisted_media_refs = {
                key: value for key, value in media_refs.items() if key not in ("part", "parts")
            }
            if kept:
                persisted_media_refs["part"] = kept[0]
                persisted_media_refs["parts"] = kept

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
            media_refs=persisted_media_refs,
            tokens=count_tokens(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_content},
                            *parts,
                        ]
                        if parts
                        else user_content,
                    }
                ]
            ),
        )
        history = await self.repository.list_messages(chat_id, self.settings.chat_context_window)
        retrieval_history = history
        provider_user_message = user_message.model_copy(update={"media_refs": media_refs})
        messages = self.context_strategy.build(chat, [*history, provider_user_message])
        if rag_service is None and not chat.system_prompt:
            messages.insert(0, {"role": "system", "content": FALLBACK_SYSTEM_PROMPT})
        budget = (
            self.settings.chat_model_context_tokens
            - self.settings.chat_response_tokens
            - self.settings.chat_safety_margin_tokens
        )
        messages = fit_to_budget(messages, max(1, budget))

        assistant_parts: list[str] = []
        rag_sources: list[dict[str, Any]] = []
        streamed_draft = ""
        blocked_screen: ModerationResult | None = None
        started_at = time.perf_counter()
        if rag_service is not None:
            rag_question = self._rag_question(user_content, media_refs)
            if rag_question is None:
                assistant_parts = [MEDIA_IMAGE_NO_TEXT_REPLY]
            else:
                rag_out: dict[str, Any] = {}
                async for delta in self._stream_with_rag(
                    rag_service,
                    rag_question,
                    retrieval_history,
                    chat_id,
                    rag_out,
                    system_prompt=chat.system_prompt,
                ):
                    streamed_draft += delta
                    yield delta
                blocked_screen = rag_out.get("blocked_result")
                rag_sources = list(rag_out.get("sources") or [])
                generated = str(rag_out.get("final") or streamed_draft).strip()
                assistant_parts = [generated or RAG_EMPTY_ANSWER]
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
        if blocked_screen is not None:
            # The mid-stream keyword screen already logged and recorded the hit.
            output_result = blocked_screen
        else:
            output_result = await self.moderation.check_output(generated_content)
            await self._record_moderation(chat_id, "output", output_result)
        if output_result.allowed:
            visible_content = generated_content
            self._last_response_sources[chat_id] = rag_sources
        else:
            visible_content = MODERATION_OUTPUT_REPLACEMENT
            self._last_response_sources[chat_id] = []

        assistant_message = ChatMessage(
            chat_id=chat_id,
            role="assistant",
            content=visible_content,
            rag_sources=rag_sources if output_result.allowed else [],
            tokens=count_tokens([{"role": "assistant", "content": visible_content}]),
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        append_turn = getattr(self.repository, "append_turn", None)
        if append_turn is not None:
            await append_turn(chat_id, user_message, assistant_message)
        else:  # compatibility for small injected test repositories
            await self.repository.append_message(chat_id, user_message)
            await self.repository.append_message(
                chat_id,
                assistant_message,
            )
        if rag_service is not None:
            if streamed_draft:
                # Deltas already went out; publish the canonical text when it
                # differs from the accumulated draft (grounding or moderation
                # may have rewritten it).
                if visible_content != streamed_draft:
                    yield StreamReplace(visible_content)
            else:
                # Nothing streamed (service without stream_answer): keep the
                # buffered contract and yield the answer as one token.
                yield visible_content
        else:
            for part in assistant_parts if output_result.allowed else [MODERATION_OUTPUT_REPLACEMENT]:
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
        texts = [
            str(part.get("text", ""))
            for part in media_parts(media_refs)
            if part.get("type") == "text" and part.get("text")
        ]
        if not texts:
            return user_content
        return "\n".join([user_content, *texts]).strip()

    @staticmethod
    def _rag_question(user_content: str, media_refs: dict[str, Any] | None) -> str | None:
        """Return the text RAG retrieval should actually search for.

        Voice transcripts and extracted document text arrive in
        ``media_refs`` parts; merging them keeps retrieval from running on
        the ``[документ x.pdf]``-style placeholder the bot sends as
        ``content``.  Images without a caption have no searchable text,
        so ``None`` asks the caller to refuse explicitly instead of
        embedding the placeholder.
        """

        parts = media_parts(media_refs)
        if not parts:
            return user_content
        texts = [
            str(part.get("text", ""))
            for part in parts
            if part.get("type") == "text" and part.get("text")
        ]
        if texts:
            merged = "\n".join([user_content, *texts]).strip()
            # Retrieval embeds the question, so document-sized payloads stay
            # below the embedding model input limit.
            return merged[:RAG_MEDIA_TEXT_MAX_CHARS]
        if all(part.get("type") == "image_url" for part in parts):
            stripped = user_content.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                return None
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
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        answer = getattr(rag_service, "answer", None)
        if answer is None:
            raise TypeError("RAG service must expose an answer method")
        kwargs = self._rag_call_kwargs(history, system_prompt)
        if inspect.iscoroutinefunction(answer):
            result = answer(question, **kwargs)
        else:
            result = await asyncio.to_thread(answer, question, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("RAG answer must be a mapping")
        return dict(result)

    async def _stream_with_rag(
        self,
        rag_service: Any,
        question: str,
        history: Sequence[ChatMessage],
        chat_id: UUID,
        out: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield RAG answer deltas while generation is still running.

        ``rag_service.stream_answer`` is a synchronous generator, so a worker
        thread feeds an async queue and the caller sees deltas live instead of
        waiting for the grounded answer.  Each accumulated prefix is screened
        by ``check_output_fragment``; a hit stops the stream, records the
        incident once, and marks ``out["blocked_result"]`` so the caller
        substitutes the moderation replacement.  ``out`` also receives the
        grounded ``final`` text and ``sources`` from the terminal event.

        Services without ``stream_answer`` fall back to the buffered
        ``answer`` contract with no delta events.
        """

        stream_answer = getattr(rag_service, "stream_answer", None)
        if not callable(stream_answer):
            result = await self._answer_with_rag(
                rag_service, question, history, system_prompt=system_prompt
            )
            out["final"] = str(result.get("answer") or "")
            out["sources"] = self._citation_array(result.get("sources"))
            return

        accumulated = ""
        call_kwargs = self._rag_call_kwargs(history, system_prompt)
        async for item in self._rag_events(stream_answer, question, call_kwargs):
            if not isinstance(item, Mapping):
                continue
            event_type = item.get("type")
            if event_type == "delta":
                delta = str(item.get("delta") or "")
                if not delta:
                    continue
                accumulated += delta
                screen = self.moderation.check_output_fragment(accumulated)
                if not screen.allowed:
                    await self._record_moderation(chat_id, "output", screen)
                    out["blocked_result"] = screen
                    return
                yield delta
            elif event_type == "final":
                out["final"] = str(item.get("answer") or "")
                out["sources"] = self._citation_array(item.get("sources"))

    @staticmethod
    def _rag_call_kwargs(
        history: Sequence[ChatMessage],
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """Build the keyword args shared by ``answer`` and ``stream_answer``.

        ``system_prompt`` is only forwarded when the chat actually defines
        one: duck-typed RAG integrations that predate the parameter keep the
        previous call signature for prompt-less chats.
        """

        kwargs: dict[str, Any] = {
            "history": [message.model_dump(mode="json") for message in history]
        }
        if system_prompt and system_prompt.strip():
            kwargs["system_prompt"] = system_prompt
        return kwargs

    async def _rag_events(
        self,
        stream_answer: Any,
        question: str,
        call_kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Normalize a sync or async RAG event source into an async iterator."""

        produced = stream_answer(question, **call_kwargs)
        if inspect.isawaitable(produced):
            produced = await produced
        if hasattr(produced, "__aiter__"):
            async for item in produced:
                yield item
            return

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for item in produced:
                    loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except BaseException as exc:  # noqa: BLE001 - relayed to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        runner = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "item":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    return
        finally:
            # to_thread cannot cancel the worker; it keeps filling the
            # unbounded queue and exits, while cancelling the awaiter task
            # keeps asyncio from reporting an abandoned task.
            if not runner.done():
                runner.cancel()

    @staticmethod
    def _citation_array(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(source) for source in value if isinstance(source, Mapping)]
