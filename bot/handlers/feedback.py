from __future__ import annotations

from uuid import UUID

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    _render_telegram_html,
    callback_owner_external_id,
    format_all_sources,
    source_document_count,
)
from bot.keyboards.inline import decode_message_context, sources_kb
from bot.services.backend_client import BackendClient


router = Router()


@router.callback_query(F.data.startswith("fb:"))
async def save_feedback(callback: CallbackQuery, backend: BackendClient) -> None:
    if callback.message is None or callback.from_user is None:
        await callback.answer("Не удалось сохранить оценку.", show_alert=True)
        return
    try:
        _, raw_vote, raw_context = (callback.data or "").split(":", maxsplit=2)
        vote = {"u": "up", "d": "down"}.get(raw_vote, raw_vote)
        if vote not in {"up", "down"}:
            raise ValueError("invalid vote")
        owner_external_id = callback_owner_external_id(callback)
        try:
            chat_id, message_id = decode_message_context(raw_context)
        except ValueError:
            # Keep feedback available for reply keyboards sent before this
            # compact callback format was introduced.
            message_id = UUID(raw_context)
            chat_id = await backend.get_or_create_chat(owner_external_id, "telegram")
        sources = await backend.get_message_sources(chat_id, message_id, owner_external_id=owner_external_id)
        await backend.save_feedback(
            chat_id,
            message_id,
            vote,
            shown_sources=sources,
            owner_external_id=owner_external_id,
        )
    except (ValueError, *BACKEND_EXCEPTIONS):
        await callback.answer("Не удалось сохранить оценку. Попробуйте позже.", show_alert=True)
        return

    await callback.message.edit_reply_markup(
        reply_markup=sources_kb(chat_id, message_id, source_document_count(sources))
    )
    await callback.answer("Спасибо за оценку!")


@router.callback_query(F.data.startswith("src:"))
async def show_all_sources(callback: CallbackQuery, backend: BackendClient) -> None:
    if callback.message is None:
        await callback.answer("Не удалось показать источники.", show_alert=True)
        return
    try:
        _, raw_context = (callback.data or "").split(":", maxsplit=1)
        chat_id, message_id = decode_message_context(raw_context)
        sources = await backend.get_message_sources(
            chat_id,
            message_id,
            owner_external_id=callback_owner_external_id(callback),
        )
    except (ValueError, *BACKEND_EXCEPTIONS):
        await callback.answer("Не удалось загрузить источники. Попробуйте позже.", show_alert=True)
        return

    if not sources:
        await callback.answer("Для этого ответа источники не найдены.", show_alert=True)
        return
    await callback.message.answer(
        _render_telegram_html(format_all_sources(sources)),
        parse_mode="HTML",
    )
    await callback.answer()
