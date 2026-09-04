from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    bot_admin_ids: list[int] = Field(default_factory=list, validation_alias=AliasChoices("BOT_ADMIN_IDS"))
    internal_token: SecretStr = Field(validation_alias=AliasChoices("INTERNAL_TOKEN"))
    bot_api_port: int = Field(default=9000, ge=1, le=65535, validation_alias=AliasChoices("BOT_API_PORT"))

    @field_validator("backend_url")
    @classmethod
    def normalize_backend_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: Any) -> list[int]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        raise TypeError("BOT_ADMIN_IDS must be a comma-separated list of integers")


@lru_cache(maxsize=1)
def get_settings() -> BotSettings:
    return BotSettings()
