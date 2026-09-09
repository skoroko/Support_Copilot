from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException

from app.deps.providers import SettingsDep


async def require_admin(
    settings: SettingsDep,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    expected = settings.admin_token.get_secret_value()
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=401,
            detail={"code": "admin_unauthorized", "message": "Valid X-Admin-Token is required."},
        )
