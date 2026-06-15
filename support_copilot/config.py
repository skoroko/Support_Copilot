from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - local fallback keeps offline demos runnable
    BaseSettings = None
    SettingsConfigDict = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - pydantic-settings can read env files itself
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def first_csv_item(value: str) -> str:
    return next((item.strip() for item in value.split(",") if item.strip()), "")


class _SettingsFields(BaseModel):
    llm_timeout_sec: float = 60.0

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_input_price_per_1m: float = 0.0
    openai_output_price_per_1m: float = 0.0

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = ""
    openrouter_input_price_per_1m: float = 0.0
    openrouter_output_price_per_1m: float = 0.0

    ollama_base_url: str = "http://localhost:11434"
    ollama_openai_base_url: str = ""
    ollama_models: str = ""
    ollama_model: str = ""
    ollama_api_key: str = "ollama"

    robust_llm_max_retries: int = 5
    robust_llm_base_backoff_sec: float = 1.0
    robust_llm_max_backoff_sec: float = 16.0
    robust_llm_max_jitter_sec: float = 0.25
    robust_llm_circuit_failures: int = 3
    robust_llm_circuit_cooldown_sec: float = 60.0
    robust_llm_rate_limit_per_minute: int = 0

    async_llm_concurrency: int = 5
    async_llm_sdk_timeout_sec: float = 30.0
    async_llm_business_timeout_sec: float = 15.0
    async_llm_max_retries: int = 3

    llm_cache_ttl_seconds: int = 3600
    llm_cache_prompt_version: str = "support-copilot-prompt-v1"
    llm_cache_service_facts_version: str = "service-facts-v1"

    vision_provider: str = "openai"
    vision_model: str = ""
    openai_vision_model: str = ""
    openrouter_vision_model: str = ""
    ollama_vision_model: str = ""
    vision_timeout_sec: float = 60.0
    vision_max_tokens: int = 1200

    support_copilot_log_path: str = "support_copilot/results/session_log.txt"
    support_copilot_tool_log_path: str = "support_copilot/results/tool_call_log.jsonl"
    tool_call_temperature: float = 0.0
    tool_call_max_tokens: int = 500
    tool_call_log_path: str = "homework_3_1/results/tool_call_demo.jsonl"

    @property
    def selected_ollama_model(self) -> str:
        return self.ollama_model or first_csv_item(self.ollama_models)

    @property
    def ollama_openai_compatible_base_url(self) -> str:
        return self.ollama_openai_base_url or self.ollama_base_url


if BaseSettings is not None:

    class SupportCopilotSettings(_SettingsFields, BaseSettings):  # type: ignore[misc]
        model_config = SettingsConfigDict(  # type: ignore[operator]
            env_file=str(DEFAULT_ENV_PATH),
            case_sensitive=False,
            extra="ignore",
        )

else:

    class SupportCopilotSettings(_SettingsFields):
        pass


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return default if not value else float(value)


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return default if not value else int(value)


def _fallback_settings(env_path: str | os.PathLike[str] | None) -> SupportCopilotSettings:
    if load_dotenv is not None and env_path is not None:
        load_dotenv(env_path, override=False)

    return SupportCopilotSettings(
        llm_timeout_sec=_env_float("LLM_TIMEOUT_SEC", 60.0),
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=_env_str("OPENAI_MODEL"),
        openai_input_price_per_1m=_env_float("OPENAI_INPUT_PRICE_PER_1M", 0.0),
        openai_output_price_per_1m=_env_float("OPENAI_OUTPUT_PRICE_PER_1M", 0.0),
        openrouter_api_key=_env_str("OPENROUTER_API_KEY"),
        openrouter_base_url=_env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=_env_str("OPENROUTER_MODEL"),
        openrouter_input_price_per_1m=_env_float("OPENROUTER_INPUT_PRICE_PER_1M", 0.0),
        openrouter_output_price_per_1m=_env_float("OPENROUTER_OUTPUT_PRICE_PER_1M", 0.0),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_openai_base_url=_env_str("OLLAMA_OPENAI_BASE_URL"),
        ollama_models=_env_str("OLLAMA_MODELS"),
        ollama_model=_env_str("OLLAMA_MODEL"),
        ollama_api_key=_env_str("OLLAMA_API_KEY", "ollama"),
        robust_llm_max_retries=_env_int("ROBUST_LLM_MAX_RETRIES", 5),
        robust_llm_base_backoff_sec=_env_float("ROBUST_LLM_BASE_BACKOFF_SEC", 1.0),
        robust_llm_max_backoff_sec=_env_float("ROBUST_LLM_MAX_BACKOFF_SEC", 16.0),
        robust_llm_max_jitter_sec=_env_float("ROBUST_LLM_MAX_JITTER_SEC", 0.25),
        robust_llm_circuit_failures=_env_int("ROBUST_LLM_CIRCUIT_FAILURES", 3),
        robust_llm_circuit_cooldown_sec=_env_float("ROBUST_LLM_CIRCUIT_COOLDOWN_SEC", 60.0),
        robust_llm_rate_limit_per_minute=_env_int("ROBUST_LLM_RATE_LIMIT_PER_MINUTE", 0),
        async_llm_concurrency=_env_int("ASYNC_LLM_CONCURRENCY", 5),
        async_llm_sdk_timeout_sec=_env_float("ASYNC_LLM_SDK_TIMEOUT_SEC", 30.0),
        async_llm_business_timeout_sec=_env_float("ASYNC_LLM_BUSINESS_TIMEOUT_SEC", 15.0),
        async_llm_max_retries=_env_int("ASYNC_LLM_MAX_RETRIES", 3),
        llm_cache_ttl_seconds=_env_int("LLM_CACHE_TTL_SECONDS", 3600),
        llm_cache_prompt_version=_env_str("LLM_CACHE_PROMPT_VERSION", "support-copilot-prompt-v1"),
        llm_cache_service_facts_version=_env_str("LLM_CACHE_SERVICE_FACTS_VERSION", "service-facts-v1"),
        vision_provider=_env_str("VISION_PROVIDER", "openai"),
        vision_model=_env_str("VISION_MODEL"),
        openai_vision_model=_env_str("OPENAI_VISION_MODEL"),
        openrouter_vision_model=_env_str("OPENROUTER_VISION_MODEL"),
        ollama_vision_model=_env_str("OLLAMA_VISION_MODEL"),
        vision_timeout_sec=_env_float("VISION_TIMEOUT_SEC", _env_float("LLM_TIMEOUT_SEC", 60.0)),
        vision_max_tokens=_env_int("VISION_MAX_TOKENS", 1200),
        support_copilot_log_path=_env_str("SUPPORT_COPILOT_LOG_PATH", "support_copilot/results/session_log.txt"),
        support_copilot_tool_log_path=_env_str(
            "SUPPORT_COPILOT_TOOL_LOG_PATH",
            "support_copilot/results/tool_call_log.jsonl",
        ),
        tool_call_temperature=_env_float("TOOL_CALL_TEMPERATURE", 0.0),
        tool_call_max_tokens=_env_int("TOOL_CALL_MAX_TOKENS", 500),
        tool_call_log_path=_env_str("TOOL_CALL_LOG_PATH", "homework_3_1/results/tool_call_demo.jsonl"),
    )


def load_settings(env_path: str | os.PathLike[str] | None = DEFAULT_ENV_PATH) -> SupportCopilotSettings:
    if BaseSettings is None:
        return _fallback_settings(env_path)

    kwargs: dict[str, Any] = {}
    if env_path is not None:
        kwargs["_env_file"] = str(env_path)
    return SupportCopilotSettings(**kwargs)
