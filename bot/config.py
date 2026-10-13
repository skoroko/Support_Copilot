from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    bot_token: SecretStr = Field(validation_alias=AliasChoices("BOT_TOKEN", "TELEGRAM_BOT_TOKEN"))
    backend_url: str = Field(default="http://127.0.0.1:8000", validation_alias=AliasChoices("BACKEND_URL"))
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", validation_alias=AliasChoices("REDIS_URL"))
    bot_admin_ids: Annotated[list[int], NoDecode] = Field(
        default_factory=list, validation_alias=AliasChoices("BOT_ADMIN_IDS")
    )
    bot_allowed_user_ids: Annotated[list[int], NoDecode] = Field(
        default_factory=list, validation_alias=AliasChoices("BOT_ALLOWED_USER_IDS")
    )
    bot_allowed_chat_ids: Annotated[list[int], NoDecode] = Field(
        default_factory=list, validation_alias=AliasChoices("BOT_ALLOWED_CHAT_IDS")
    )
    internal_token: SecretStr = Field(validation_alias=AliasChoices("INTERNAL_TOKEN"))
    admin_token: SecretStr = Field(validation_alias=AliasChoices("ADMIN_TOKEN"))
    bot_api_port: int = Field(default=9000, ge=1, le=65535, validation_alias=AliasChoices("BOT_API_PORT"))
    broadcast_poll_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias=AliasChoices("BROADCAST_POLL_SECONDS"),
    )

    @field_validator("backend_url")
    @classmethod
    def normalize_backend_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("bot_admin_ids", "bot_allowed_user_ids", "bot_allowed_chat_ids", mode="before")
    @classmethod
    def parse_telegram_ids(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                decoded = json.loads(value)
                if not isinstance(decoded, list):
                    raise ValueError("Telegram IDs JSON value must be an array")
                return [int(item) for item in decoded]
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        raise TypeError("Telegram IDs must be integers or a comma-separated list of integers")

    @model_validator(mode="after")
    def validate_privileged_tokens(self) -> "BotSettings":
        internal = self.internal_token.get_secret_value().strip()
        admin = self.admin_token.get_secret_value().strip()
        forbidden = {
            "replace-with-internal-token",
            "replace-with-admin-token",
            "change-me",
            "changeme",
            "secret",
        }
        if len(internal) < 32 or internal.casefold() in forbidden:
            raise ValueError("INTERNAL_TOKEN must be a non-placeholder secret of at least 32 characters")
        if len(admin) < 32 or admin.casefold() in forbidden:
            raise ValueError("ADMIN_TOKEN must be a non-placeholder secret of at least 32 characters")
        if internal == admin:
            raise ValueError("INTERNAL_TOKEN and ADMIN_TOKEN must be different")
        return self


@lru_cache(maxsize=1)
def get_settings() -> BotSettings:
    return BotSettings()
