from __future__ import annotations

from aiogram import Router

from bot.handlers import commands, fsm, text


router = Router()
router.include_router(commands.router)
router.include_router(fsm.router)
router.include_router(text.router)
