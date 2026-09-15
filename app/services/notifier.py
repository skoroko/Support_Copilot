from __future__ import annotations

import httpx

from app.core.config import Settings


async def notify_user(
    chat_id_tg: int,
    text: str,
    *,
    settings: Settings,
    idempotency_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Send a proactive event from the backend to the bot's internal API."""

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=5.0)
    try:
        payload: dict[str, object] = {"chat_id": chat_id_tg, "text": text}
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        response = await http.post(
            f"{settings.bot_url.rstrip('/')}/notify",
            json=payload,
            headers={"X-Internal-Token": settings.internal_token.get_secret_value()},
        )
        response.raise_for_status()
        result = response.json()
        if result.get("ok") is not True:
            raise RuntimeError("Bot notification was not accepted")
    finally:
        if owns_client:
            await http.aclose()
