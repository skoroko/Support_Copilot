"""One-use, checkpoint-backed expansion of an exact persisted response."""

from __future__ import annotations

from contextlib import aclosing
from uuid import UUID, uuid4


class ExpansionUnavailableError(ValueError):
    pass


async def _target(service, chat_id: UUID, message_id: UUID):
    chat = await service.repository.get_chat(chat_id)
    message = await service.repository.get_message(chat_id, message_id)
    if chat is None or message is None or message.role != "assistant":
        raise ExpansionUnavailableError("Ответ недоступен или удалён.")
    info = (message.media_refs or {}).get("answer") or {}
    if service.agent is None or chat.interface != "telegram" or not info.get("expandable") or info.get("mode") != "brief":
        raise ExpansionUnavailableError("Для этого ответа подробности недоступны.")
    if await service.agent.pending(chat):
        raise ExpansionUnavailableError("Сначала обработайте подтверждение отправки.")
    return chat, message


async def expansion_availability(service, chat_id: UUID, message_id: UUID) -> dict:
    async with service._turn_lock(chat_id):
        chat, _ = await _target(service, chat_id, message_id)
        latest = await service.repository.list_messages(chat_id, 1)
        claimed = await service.agent.expansion_claim(chat, str(message_id))
        if claimed or not latest or latest[-1].id != message_id:
            raise ExpansionUnavailableError("Ответ устарел или подробности уже запрошены.")
        return {"available": True, "max_output_tokens": service.settings.chat_expanded_response_tokens}


def _result(message) -> dict:
    return {"answer": message.content, "sources": message.rag_sources,
            "message_id": str(message.id), "answer_info": (message.media_refs or {}).get("answer", {})}


async def expand_response(service, chat_id: UUID, message_id: UUID) -> dict:
    async with service._turn_lock(chat_id):
        chat, _ = await _target(service, chat_id, message_id)
        latest = await service.repository.list_messages(chat_id, 1)
        claimed = await service.agent.expansion_claim(chat, str(message_id))
        if claimed:
            saved = await service.repository.get_message(chat_id, UUID(claimed))
            if saved is not None and latest and latest[-1].id == saved.id:
                return _result(saved)
            raise ExpansionUnavailableError("Подробности уже запрошены. Повторная генерация отключена.")
        if not latest or latest[-1].id != message_id:
            raise ExpansionUnavailableError("Ответ устарел. Задайте новый вопрос.")
        result_id = uuid4()
        content = "Раскрой предыдущий ответ подробнее."
        # Screen before consuming the one-use claim; output follows the same
        # moderation and atomic append_turn path as an ordinary response.
        await service.ensure_input_allowed(chat_id, content)
        await service.agent.claim_expansion(chat, str(message_id), str(result_id))
        async with aclosing(service._send_message(chat_id, content, input_moderated=True,
                            expanded_from=message_id, result_message_id=result_id)) as events:
            async for _ in events:
                pass
        saved = await service.repository.get_message(chat_id, result_id)
        if saved is None:
            raise ExpansionUnavailableError("Не удалось сохранить подробный ответ.")
        return _result(saved)
