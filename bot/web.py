from __future__ import annotations

import hmac

from aiogram import Bot
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class NotifyRequest(BaseModel):
    chat_id: int
    text: str = Field(min_length=1)


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    api = FastAPI(title="Support Copilot Bot Internal API")

    @api.post("/notify")
    async def notify(req: NotifyRequest, x_internal_token: str = Header(...)) -> dict[str, bool]:
        if not hmac.compare_digest(x_internal_token, internal_token):
            raise HTTPException(status_code=401, detail="invalid internal token")
        await bot.send_message(chat_id=req.chat_id, text=req.text)
        return {"ok": True}

    return api
