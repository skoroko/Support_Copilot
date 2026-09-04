from __future__ import annotations

import httpx

from app.core.config import Settings


async def notify_user(
    chat_id_tg: int,
    text: str,
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Send a proactive event from the backend to the bot's internal API."""

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=5.0)
    try:
        response = await http.post(
            f"{settings.bot_url.rstrip('/')}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": settings.internal_token.get_secret_value()},
        )
        response.raise_for_status()
    finally:
        if owns_client:
            await http.aclose()
