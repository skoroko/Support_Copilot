"""Buttons resume only the exact durable draft owned by the clicking user."""

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.handlers._common import (
    BACKEND_EXCEPTIONS,
    _format_cited_sources,
    callback_owner_external_id,
    response_feedback_markup,
    send_rendered_answer,
)
from bot.keyboards.inline import decode_message_context
from bot.services.backend_client import BackendClient

router = Router()


@router.callback_query(F.data.startswith("ap:"))
async def approve(callback: CallbackQuery, backend: BackendClient) -> None:
    if callback.message is None or callback.from_user is None:
        await callback.answer("Подтверждение недоступно.", show_alert=True)
        return
    try:
        _, action, context = (callback.data or "").split(":", 2)
        if action not in {"y", "n"}:
            raise ValueError("invalid action")
        chat_id, approval_id = decode_message_context(context)
        result = await backend.resume_approval(
            chat_id,
            approval_id,
            action == "y",
            owner_external_id=callback_owner_external_id(callback),
        )
    except (ValueError, *BACKEND_EXCEPTIONS):
        await callback.answer(
            "Подтверждение устарело, уже обработано или принадлежит другому пользователю.",
            show_alert=True,
        )
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    answer = str(result.get("answer") or "Действие завершено.")
    answer += _format_cited_sources(answer, result.get("sources") or [])
    if result.get("approval"):
        answer += "\n\n" + backend.last_response_notice(chat_id)
    await send_rendered_answer(
        callback.message,
        answer,
        reply_markup=response_feedback_markup(backend, chat_id),
    )
    await callback.answer()
