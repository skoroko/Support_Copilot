"""Register Telegram's always accessible native commands menu."""

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand, MenuButtonCommands

from app.observability.logging import get_logger

logger = get_logger("telegram-bot-menu")


async def register_menu(bot: Bot) -> None:
    """Best effort: temporary Telegram failures must not block polling."""
    try:
        await bot.set_my_commands([
            BotCommand(command="menu", description="Открыть меню"),
            BotCommand(command="start", description="Начать работу"),
            BotCommand(command="help", description="Как пользоваться"),
            BotCommand(command="history", description="Последние сообщения"),
            BotCommand(command="clear", description="Сразу очистить диалог и ожидающую отправку"),
        ])
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except TelegramAPIError as exc:
        logger.warning("telegram_menu_unavailable", error_type=type(exc).__name__)
