"""Explicit cost notice and owner-bound confirmation for a detailed answer."""

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery

from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    callback_owner_external_id,
    source_document_count,
    stream_to_chat,
)
from bot.keyboards.inline import answer_kb, decode_message_context, expansion_kb
from bot.services.backend_client import BackendClient, StreamReplace

router = Router()
COST_NOTICE = "Подробный ответ — дополнительный запрос к ИИ. Он потребует токены и может увеличить стоимость обработки. Подготовить?"
UNAVAILABLE = "Подробности недоступны: ответ устарел, уже обработан или принадлежит другому пользователю."


@router.callback_query(F.data.startswith("exp:"))
async def expansion(callback: CallbackQuery, backend: BackendClient) -> None:
    if callback.message is None or callback.from_user is None:
        await callback.answer("Подробности недоступны.", show_alert=True)
        return
    # Acknowledge promptly, before HTTP requests or generation.
    await callback.answer()
    try:
        _, action, token = (callback.data or "").split(":", 2)
        if action not in {"a", "y", "n"}:
            raise ValueError("invalid expansion action")
        chat_id, message_id = decode_message_context(token)
        owner = callback_owner_external_id(callback)
        if action == "a":
            await backend.preview_expansion(chat_id, message_id, owner_external_id=owner)
            await callback.message.answer(COST_NOTICE, reply_markup=expansion_kb(chat_id, message_id))
            return
        if action == "n":
            await backend.preview_expansion(chat_id, message_id, owner_external_id=owner)
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramAPIError:
                pass
            return

        result = {}

        async def detailed_answer():
            try:
                result.update(await backend.expand_response(chat_id, message_id, owner_external_id=owner))
            except (ValueError, *BACKEND_EXCEPTIONS):
                # Error text uses the same drain/new-message/status cleanup.
                yield StreamReplace(UNAVAILABLE)
                return
            # Remove only consent panel controls after successful generation.
            try:
                await callback.message.edit_reply_markup(reply_markup=None)
            except TelegramAPIError:
                pass
            yield StreamReplace(str(result.get("answer") or "Не удалось подготовить подробный ответ."))

        await stream_to_chat(
            callback.message,
            detailed_answer(),
            sources_factory=lambda: result.get("sources") or [],
            reply_markup_factory=lambda: answer_kb(
                chat_id, result["message_id"], source_document_count(result.get("sources") or []))
                if result.get("message_id") else None,
            cleanup_waiting_on_error=True,
        )
    except (ValueError, *BACKEND_EXCEPTIONS):
        await callback.message.answer(UNAVAILABLE)
        return
