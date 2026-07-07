from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - local environment can run tests without it
    BaseSettings = None
    SettingsConfigDict = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class LLMSettings(BaseModel):
    openai_api_key: SecretStr
    default_model: str = "gpt-5.4-mini"
    request_timeout: float = 30.0


class _SettingsFields(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM__OPENAI_API_KEY"),
    )
    default_model: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices("APP_DEFAULT_MODEL", "OPENAI_MODEL", "LLM__DEFAULT_MODEL"),
    )
    request_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT", "ASYNC_LLM_SDK_TIMEOUT_SEC", "LLM__REQUEST_TIMEOUT"),
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("APP_REDIS_URL", "REDIS_URL"),
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        validation_alias=AliasChoices("APP_CACHE_TTL_SECONDS", "LLM_CACHE_TTL_SECONDS"),
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices("APP_CORS_ORIGINS", "CORS_ORIGINS"),
    )
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "APP_LOG_LEVEL"))
    security_enabled: bool = Field(default=True, validation_alias=AliasChoices("APP_SECURITY_ENABLED"))
    chat_repository: Literal["json", "postgres"] = Field(default="json", validation_alias=AliasChoices("CHAT_REPOSITORY"))
    chat_storage_dir: Path = Field(default=Path("./var/chats"), validation_alias=AliasChoices("CHAT_STORAGE_DIR"))
    chat_database_url: str = Field(
        default="postgresql+asyncpg://support:support@localhost:5432/support_copilot",
        validation_alias=AliasChoices("CHAT_DATABASE_URL", "DATABASE_URL"),
    )
    chat_context_strategy: Literal["sliding", "hybrid"] = Field(
        default="sliding",
        validation_alias=AliasChoices("CHAT_CONTEXT_STRATEGY"),
    )
    chat_context_window: int = Field(default=10, gt=0, validation_alias=AliasChoices("CHAT_CONTEXT_WINDOW"))
    chat_model_context_tokens: int = Field(
        default=128_000,
        gt=0,
        validation_alias=AliasChoices("CHAT_MODEL_CONTEXT_TOKENS"),
    )
    chat_response_tokens: int = Field(default=500, gt=0, validation_alias=AliasChoices("CHAT_RESPONSE_TOKENS"))
    chat_safety_margin_tokens: int = Field(
        default=500,
        ge=0,
        validation_alias=AliasChoices("CHAT_SAFETY_MARGIN_TOKENS"),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return ["*"]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        raise TypeError("cors_origins must be a list, JSON array, or comma-separated string")

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY is required for app.main:app startup")
        return value

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            openai_api_key=self.openai_api_key,
            default_model=self.default_model,
            request_timeout=self.request_timeout,
        )


if BaseSettings is not None:

    class Settings(_SettingsFields, BaseSettings):  # type: ignore[misc]
        model_config = SettingsConfigDict(  # type: ignore[operator]
            env_file=str(DEFAULT_ENV_PATH),
            env_nested_delimiter="__",
            case_sensitive=False,
            extra="ignore",
            populate_by_name=True,
        )

else:

    class Settings(_SettingsFields):
        pass


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    if BaseSettings is not None:
        return Settings()

    _load_env_file(DEFAULT_ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM__OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("OPENAI_API_KEY is required for app.main:app startup")

    return Settings(
        openai_api_key=SecretStr(api_key),
        default_model=os.getenv("APP_DEFAULT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4-mini",
        request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT") or os.getenv("ASYNC_LLM_SDK_TIMEOUT_SEC") or 30.0),
        redis_url=os.getenv("APP_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0",
        cache_ttl_seconds=int(os.getenv("APP_CACHE_TTL_SECONDS") or os.getenv("LLM_CACHE_TTL_SECONDS") or 3600),
        cors_origins=os.getenv("APP_CORS_ORIGINS") or os.getenv("CORS_ORIGINS") or "*",
        log_level=os.getenv("LOG_LEVEL") or os.getenv("APP_LOG_LEVEL") or "INFO",
        security_enabled=(os.getenv("APP_SECURITY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}),
        chat_repository=os.getenv("CHAT_REPOSITORY") or "json",
        chat_storage_dir=Path(os.getenv("CHAT_STORAGE_DIR") or "./var/chats"),
        chat_database_url=(
            os.getenv("CHAT_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or "postgresql+asyncpg://support:support@localhost:5432/support_copilot"
        ),
        chat_context_strategy=os.getenv("CHAT_CONTEXT_STRATEGY") or "sliding",
        chat_context_window=int(os.getenv("CHAT_CONTEXT_WINDOW") or 10),
        chat_model_context_tokens=int(os.getenv("CHAT_MODEL_CONTEXT_TOKENS") or 128_000),
        chat_response_tokens=int(os.getenv("CHAT_RESPONSE_TOKENS") or 500),
        chat_safety_margin_tokens=int(os.getenv("CHAT_SAFETY_MARGIN_TOKENS") or 500),
    )
