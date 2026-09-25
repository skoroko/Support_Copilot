from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - local environment can run tests without it
    BaseSettings = None
    SettingsConfigDict = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
MIN_PRIVILEGED_TOKEN_LENGTH = 32
FORBIDDEN_PRIVILEGED_TOKENS = frozenset(
    {
        "replace-with-internal-token",
        "replace-with-admin-token",
        "change-me",
        "changeme",
        "secret",
    }
)


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
    openai_base_url: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_BASE_URL"))
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
    allowed_models: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("APP_ALLOWED_MODELS"),
    )
    api_rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("APP_RATE_LIMIT_PER_MINUTE"),
    )
    api_max_concurrency_per_principal: int = Field(
        default=4,
        ge=1,
        validation_alias=AliasChoices("APP_MAX_CONCURRENCY_PER_PRINCIPAL"),
    )
    ingestion_queue_capacity: int = Field(
        default=4,
        ge=1,
        le=100,
        validation_alias=AliasChoices("INGESTION_QUEUE_CAPACITY"),
    )
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
    transcription_model: str = Field(default="whisper-1", validation_alias=AliasChoices("TRANSCRIPTION_MODEL"))
    bot_url: str = Field(default="http://bot:9000", validation_alias=AliasChoices("BOT_URL"))
    internal_token: SecretStr = Field(validation_alias=AliasChoices("INTERNAL_TOKEN"))
    admin_token: SecretStr = Field(validation_alias=AliasChoices("ADMIN_TOKEN"))
    chat_media_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=50 * 1024 * 1024,
        validation_alias=AliasChoices("CHAT_MEDIA_MAX_BYTES"),
    )
    moderation_enabled: bool = Field(default=True, validation_alias=AliasChoices("MODERATION_ENABLED"))
    moderation_openai_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("MODERATION_OPENAI_ENABLED"),
    )
    moderation_model: str = Field(
        default="omni-moderation-latest",
        validation_alias=AliasChoices("MODERATION_MODEL"),
    )
    moderation_thresholds: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("MODERATION_THRESHOLDS"),
    )
    moderation_keywords_path: Path = Field(
        default=PROJECT_ROOT / "app" / "moderation" / "moderation_keywords.yaml",
        validation_alias=AliasChoices("MODERATION_KEYWORDS_PATH"),
    )
    qdrant_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_URL"),
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY"),
    )
    rag_enabled: bool = Field(default=False, validation_alias=AliasChoices("RAG_ENABLED"))
    rag_ingest_on_start: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_INGEST_ON_START"),
    )
    rag_allow_external_data_transfer: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAG_ALLOW_EXTERNAL_DATA_TRANSFER"),
    )
    rag_data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        validation_alias=AliasChoices("RAG_DATA_DIR"),
    )
    rag_docstore_path: Path = Field(
        default=PROJECT_ROOT / "var" / "rag_eval_chunk_1024_docstore.json",
        validation_alias=AliasChoices("RAG_DOCSTORE_PATH"),
    )
    rag_llm_model: str = Field(default="gpt-5.4-mini", min_length=1, validation_alias=AliasChoices("RAG_LLM_MODEL"))
    # Answer generation gets a dedicated budget: retrieval (embedding + Qdrant)
    # stays on the shared request_timeout, while a grounded completion over
    # several fragments legitimately needs longer.
    rag_generation_timeout: float = Field(
        default=120.0,
        gt=0,
        validation_alias=AliasChoices("RAG_GENERATION_TIMEOUT"),
    )
    rag_embedding_model: str = Field(
        default="text-embedding-3-large",
        min_length=1,
        validation_alias=AliasChoices("RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    rag_embedding_dimensions: int = Field(
        default=1024,
        gt=0,
        validation_alias=AliasChoices("RAG_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS"),
    )
    rag_chunk_strategy: Literal["fixed", "recursive", "semantic"] = Field(
        default="recursive",
        validation_alias=AliasChoices("RAG_CHUNK_STRATEGY"),
    )
    rag_chunk_size: int = Field(default=1024, gt=0, validation_alias=AliasChoices("RAG_CHUNK_SIZE"))
    rag_chunk_overlap: int = Field(default=64, ge=0, validation_alias=AliasChoices("RAG_CHUNK_OVERLAP"))
    rag_top_k: int = Field(default=10, ge=3, validation_alias=AliasChoices("RAG_TOP_K"))
    rag_product: str = Field(default="gts_nx", min_length=1, validation_alias=AliasChoices("RAG_PRODUCT"))
    rag_product_scope: Literal["strict", "fallback", "all"] = Field(
        default="fallback",
        validation_alias=AliasChoices("RAG_PRODUCT_SCOPE"),
    )
    rag_reranker_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAG_RERANKER_ENABLED"),
    )
    rag_reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        min_length=1,
        validation_alias=AliasChoices("RAG_RERANKER_MODEL"),
    )
    rag_reranker_device: str = Field(default="", validation_alias=AliasChoices("RAG_RERANKER_DEVICE"))
    rag_reranker_batch_size: int = Field(
        default=4,
        gt=0,
        validation_alias=AliasChoices("RAG_RERANKER_BATCH_SIZE"),
    )
    rag_reranker_top_n: int = Field(
        default=5,
        gt=0,
        validation_alias=AliasChoices("RAG_RERANKER_TOP_N"),
    )
    rag_min_score: float = Field(
        default=0.45,
        ge=0,
        le=1,
        validation_alias=AliasChoices("RAG_SCORE_THRESHOLD", "RAG_MIN_SCORE"),
    )
    rag_condense_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_CONDENSE_ENABLED"),
    )
    rag_condense_llm_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_CONDENSE_LLM_ENABLED"),
    )
    rag_lexical_retrieval_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_LEXICAL_RETRIEVAL_ENABLED"),
    )
    rag_lexical_term_weight: float = Field(
        default=0.7,
        ge=0,
        le=1,
        validation_alias=AliasChoices("RAG_LEXICAL_TERM_WEIGHT"),
    )
    rag_lexical_pair_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        validation_alias=AliasChoices("RAG_LEXICAL_PAIR_WEIGHT"),
    )
    rag_collection: str = Field(
        default="rag_eval_chunk_1024",
        min_length=1,
        validation_alias=AliasChoices("RAG_COLLECTION"),
    )
    rag_baremetal_collection: str = Field(
        default="rag_block_03_baremetal",
        min_length=1,
        validation_alias=AliasChoices("RAG_BAREMETAL_COLLECTION"),
    )
    rag_rebuild_on_start: bool = Field(default=False, validation_alias=AliasChoices("RAG_REBUILD_ON_START"))

    @model_validator(mode="after")
    def validate_rag_chunking(self) -> "_SettingsFields":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.rag_collection == self.rag_baremetal_collection:
            raise ValueError("RAG collections must be different")
        return self

    @model_validator(mode="after")
    def validate_privileged_tokens(self) -> "_SettingsFields":
        values = {
            "INTERNAL_TOKEN": self.internal_token.get_secret_value().strip(),
            "ADMIN_TOKEN": self.admin_token.get_secret_value().strip(),
        }
        for name, value in values.items():
            if len(value) < MIN_PRIVILEGED_TOKEN_LENGTH or value.casefold() in FORBIDDEN_PRIVILEGED_TOKENS:
                raise ValueError(f"{name} must be a non-placeholder secret of at least {MIN_PRIVILEGED_TOKEN_LENGTH} characters")
        if values["INTERNAL_TOKEN"] == values["ADMIN_TOKEN"]:
            raise ValueError("INTERNAL_TOKEN and ADMIN_TOKEN must be different")
        return self

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

    @field_validator("allowed_models", mode="before")
    @classmethod
    def parse_allowed_models(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        raise TypeError("APP_ALLOWED_MODELS must be a list, JSON array, or comma-separated string")

    @field_validator("moderation_thresholds", mode="before")
    @classmethod
    def parse_moderation_thresholds(cls, value: Any) -> dict[str, float]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return {str(key): float(score) for key, score in value.items()}
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{"):
                parsed = json.loads(stripped)
                return {str(key): float(score) for key, score in parsed.items()}
            result: dict[str, float] = {}
            for item in stripped.split(","):
                if not item.strip():
                    continue
                name, score = item.split("=", 1)
                result[name.strip()] = float(score.strip())
            return result
        raise TypeError("MODERATION_THRESHOLDS must be JSON or category=score pairs")

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

    @property
    def effective_allowed_models(self) -> frozenset[str]:
        return frozenset(self.allowed_models or [self.default_model])

    @property
    def rag_score_threshold(self) -> float:
        """Public Б5.5 name; ``rag_min_score`` remains for backward compatibility."""

        return self.rag_min_score


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
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        redis_url=os.getenv("APP_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0",
        cache_ttl_seconds=int(os.getenv("APP_CACHE_TTL_SECONDS") or os.getenv("LLM_CACHE_TTL_SECONDS") or 3600),
        cors_origins=os.getenv("APP_CORS_ORIGINS") or os.getenv("CORS_ORIGINS") or "*",
        log_level=os.getenv("LOG_LEVEL") or os.getenv("APP_LOG_LEVEL") or "INFO",
        security_enabled=(os.getenv("APP_SECURITY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}),
        allowed_models=os.getenv("APP_ALLOWED_MODELS") or "",
        api_rate_limit_per_minute=int(os.getenv("APP_RATE_LIMIT_PER_MINUTE") or 60),
        api_max_concurrency_per_principal=int(os.getenv("APP_MAX_CONCURRENCY_PER_PRINCIPAL") or 4),
        ingestion_queue_capacity=int(os.getenv("INGESTION_QUEUE_CAPACITY") or 4),
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
        transcription_model=os.getenv("TRANSCRIPTION_MODEL") or "whisper-1",
        bot_url=os.getenv("BOT_URL") or "http://bot:9000",
        internal_token=SecretStr(os.getenv("INTERNAL_TOKEN") or ""),
        admin_token=SecretStr(os.getenv("ADMIN_TOKEN") or ""),
        chat_media_max_bytes=int(os.getenv("CHAT_MEDIA_MAX_BYTES") or 10 * 1024 * 1024),
        moderation_enabled=(os.getenv("MODERATION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}),
        moderation_openai_enabled=(
            os.getenv("MODERATION_OPENAI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        ),
        moderation_model=os.getenv("MODERATION_MODEL") or "omni-moderation-latest",
        moderation_thresholds=os.getenv("MODERATION_THRESHOLDS") or "{}",
        moderation_keywords_path=Path(
            os.getenv("MODERATION_KEYWORDS_PATH")
            or PROJECT_ROOT / "app" / "moderation" / "moderation_keywords.yaml"
        ),
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=(
            SecretStr(os.environ["QDRANT_API_KEY"])
            if os.getenv("QDRANT_API_KEY")
            else None
        ),
        rag_enabled=(os.getenv("RAG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}),
        rag_ingest_on_start=(
            os.getenv("RAG_INGEST_ON_START", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        rag_allow_external_data_transfer=(
            os.getenv("RAG_ALLOW_EXTERNAL_DATA_TRANSFER", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        rag_data_dir=Path(os.getenv("RAG_DATA_DIR") or PROJECT_ROOT / "data"),
        rag_docstore_path=Path(
            os.getenv("RAG_DOCSTORE_PATH") or PROJECT_ROOT / "var" / "rag_eval_chunk_1024_docstore.json"
        ),
        rag_llm_model=os.getenv("RAG_LLM_MODEL") or "gpt-5.4-mini",
        rag_generation_timeout=float(os.getenv("RAG_GENERATION_TIMEOUT") or 120.0),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-large",
        rag_embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS") or os.getenv("EMBEDDING_DIMENSIONS") or 1024),
        rag_chunk_strategy=os.getenv("RAG_CHUNK_STRATEGY") or "recursive",
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE") or 1024),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP") or 64),
        rag_top_k=int(os.getenv("RAG_TOP_K") or 10),
        rag_product=os.getenv("RAG_PRODUCT") or "gts_nx",
        rag_reranker_enabled=(
            os.getenv("RAG_RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        ),
        rag_reranker_model=os.getenv("RAG_RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3",
        rag_reranker_device=os.getenv("RAG_RERANKER_DEVICE") or "",
        rag_reranker_batch_size=int(os.getenv("RAG_RERANKER_BATCH_SIZE") or 4),
        rag_reranker_top_n=int(os.getenv("RAG_RERANKER_TOP_N") or 5),
        rag_min_score=float(os.getenv("RAG_SCORE_THRESHOLD") or os.getenv("RAG_MIN_SCORE") or 0.45),
        rag_condense_enabled=(
            os.getenv("RAG_CONDENSE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        ),
        rag_condense_llm_enabled=(
            os.getenv("RAG_CONDENSE_LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        ),
        rag_lexical_term_weight=float(os.getenv("RAG_LEXICAL_TERM_WEIGHT") or 0.7),
        rag_lexical_pair_weight=float(os.getenv("RAG_LEXICAL_PAIR_WEIGHT") or 0.3),
        rag_collection=os.getenv("RAG_COLLECTION") or "rag_eval_chunk_1024",
        rag_baremetal_collection=os.getenv("RAG_BAREMETAL_COLLECTION") or "rag_block_03_baremetal",
        rag_rebuild_on_start=(
            os.getenv("RAG_REBUILD_ON_START", "false").strip().lower() in {"1", "true", "yes", "on"}
        ),
    )
