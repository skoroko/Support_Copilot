from __future__ import annotations

import hmac
from typing import Any

from aiogram import Bot
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class NotifyRequest(BaseModel):
    chat_id: int
    text: str = Field(min_length=1)
    idempotency_key: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )


def build_api(bot: Bot, internal_token: str, *, idempotency_store: Any | None = None) -> FastAPI:
    normalized_token = internal_token.strip()
    if len(normalized_token) < 32 or normalized_token.casefold() in {
        "replace-with-internal-token",
        "change-me",
        "changeme",
        "secret",
    }:
        raise ValueError("Bot internal authentication requires a strong configured token")
    api = FastAPI(title="Support Copilot Bot Internal API")

    @api.post("/notify")
    async def notify(req: NotifyRequest, x_internal_token: str = Header(...)) -> dict[str, bool]:
        if not hmac.compare_digest(x_internal_token, normalized_token):
            raise HTTPException(status_code=401, detail="invalid internal token")
        storage_key = None
        claimed = False
        if req.idempotency_key is not None:
            if idempotency_store is None:
                raise HTTPException(status_code=503, detail="idempotency store unavailable")
            storage_key = f"telegram-notify:{req.idempotency_key}"
            claimed = bool(await idempotency_store.set(storage_key, "pending", nx=True, ex=604_800))
            if not claimed:
                delivery_state = await idempotency_store.get(storage_key)
                if delivery_state == "sent":
                    return {"ok": True, "deduplicated": True, "delivery_unknown": False}
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "delivery_unknown",
                        "message": "A previous Telegram delivery outcome requires reconciliation",
                    },
                )
        try:
            await bot.send_message(chat_id=req.chat_id, text=req.text)
        except Exception as exc:
            if claimed and storage_key is not None:
                await idempotency_store.set(storage_key, "unknown", xx=True, ex=604_800)
            raise HTTPException(
                status_code=502,
                detail={"code": "delivery_unknown", "message": "Telegram delivery outcome is unknown"},
            ) from exc
        if claimed and storage_key is not None:
            await idempotency_store.set(storage_key, "sent", xx=True, ex=604_800)
        return {"ok": True, "deduplicated": False, "delivery_unknown": False}

    return api
