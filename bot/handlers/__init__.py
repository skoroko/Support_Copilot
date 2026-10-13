from __future__ import annotations

from aiogram import Router

from bot.handlers import (
    admin,
    approval,
    commands,
    expansion,
    feedback,
    media,
    menu,
    text,
)

router = Router()
router.include_router(admin.router)
router.include_router(commands.router)
router.include_router(menu.router)
router.include_router(approval.router)
router.include_router(expansion.router)
router.include_router(feedback.router)
router.include_router(media.router)
router.include_router(text.router)
