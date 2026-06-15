from __future__ import annotations

import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

try:
    from openai import APIStatusError, APITimeoutError, OpenAI
except ImportError:  # pragma: no cover - fake demo can run without OpenAI SDK
    OpenAI = None

    class APIStatusError(Exception):
        status_code: int | None = None

    class APITimeoutError(Exception):
        pass


SERVICE_UNAVAILABLE_MESSAGE = (
    "Сервис временно недоступен. Передайте обращение инженеру поддержки и повторите запрос позже."
)


class ConfigError(ValueError):
    pass


class ChatClient(Protocol):
    chat: Any


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    timeout_sec: float = 60.0
    max_tokens_param: str = "max_completion_tokens"
    stream_include_usage: bool = True


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        *,
        total_tokens: int | None = None,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.cached_tokens += cached_tokens
        self.reasoning_tokens += reasoning_tokens
        self.cost_usd += cost_usd


@dataclass(frozen=True)
class LLMResult:
    answer: str
    provider: str | None
    model: str | None
    usage: UsageStats
    cost_usd: float
    unavailable: bool = False


@dataclass(frozen=True)
class ClientLogEvent:
    timestamp: str
    provider: str
    attempt: int
    code: str
    message: str
    delay_sec: float | None = None

    def to_line(self) -> str:
        delay = "" if self.delay_sec is None else f" delay={self.delay_sec:.2f}s"
        return (
            f"{self.timestamp} provider={self.provider} attempt={self.attempt} "
            f"code={self.code}{delay} message={self.message}"
        )


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_sec: float = 60.0
    consecutive_failures: int = 0
    opened_until: float = 0.0

    def is_open(self, now_monotonic: float) -> bool:
        return now_monotonic < self.opened_until

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_until = 0.0

    def record_failure(self, now_monotonic: float) -> bool:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.opened_until = now_monotonic + self.cooldown_sec
            return True
        return False


@dataclass
class RateLimiter:
    max_requests_per_minute: int = 0
    timestamps: deque[float] = field(default_factory=deque)

    def acquire(
        self,
        now_fn: Callable[[], float],
        sleep_fn: Callable[[float], None],
    ) -> float:
        if self.max_requests_per_minute <= 0:
            return 0.0

        now = now_fn()
        self._drop_old(now)
        if len(self.timestamps) < self.max_requests_per_minute:
            self.timestamps.append(now)
            return 0.0

        wait_sec = max(0.0, 60.0 - (now - self.timestamps[0]))
        sleep_fn(wait_sec)
        now = now_fn()
        self._drop_old(now)
        self.timestamps.append(now)
        return wait_sec

    def _drop_old(self, now: float) -> None:
        while self.timestamps and now - self.timestamps[0] >= 60.0:
            self.timestamps.popleft()


@dataclass
class ProviderRuntime:
    config: ProviderConfig
    client: ChatClient
    circuit_breaker: CircuitBreaker
    rate_limiter: RateLimiter


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_float(name: str, default: float) -> float:
    value = env_str(name)
    return default if not value else float(value)


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    return default if not value else int(value)


def first_csv_item(value: str) -> str:
    return next((item.strip() for item in value.split(",") if item.strip()), "")


def normalize_openai_base_url(base_url: str, *, provider: str) -> str:
    normalized = base_url.rstrip("/")
    if provider == "ollama" and not normalized.endswith("/v1"):
        return f"{normalized}/v1"
    return normalized


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def value_from(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def sanitize_error(message: str, api_key: str) -> str:
    sanitized = message.replace(api_key, "***") if api_key else message
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-***", sanitized)


def error_code(exc: Exception) -> str:
    status_code = value_from(exc, "status_code")
    if status_code is not None:
        return str(status_code)
    if value_from(exc, "is_timeout", False) or isinstance(exc, APITimeoutError):
        return "timeout"
    return exc.__class__.__name__


def is_retryable_error(exc: Exception) -> bool:
    status_code = value_from(exc, "status_code")
    if status_code == 429:
        return True
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True
    if value_from(exc, "is_timeout", False) or isinstance(exc, APITimeoutError):
        return True
    return False


def calculate_cost(
    usage: UsageStats,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    return (
        usage.prompt_tokens * input_price_per_1m / 1_000_000
        + usage.completion_tokens * output_price_per_1m / 1_000_000
    )


def extract_usage(response: Any) -> UsageStats:
    raw_usage = value_from(response, "usage")
    if raw_usage is None:
        return UsageStats()
    prompt_tokens = int(value_from(raw_usage, "prompt_tokens", value_from(raw_usage, "input_tokens", 0)) or 0)
    completion_tokens = int(
        value_from(raw_usage, "completion_tokens", value_from(raw_usage, "output_tokens", 0)) or 0
    )
    total_tokens = int(value_from(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    prompt_details = value_from(raw_usage, "prompt_tokens_details") or value_from(raw_usage, "input_tokens_details") or {}
    completion_details = (
        value_from(raw_usage, "completion_tokens_details")
        or value_from(raw_usage, "output_tokens_details")
        or {}
    )
    cached_tokens = int(value_from(prompt_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(value_from(completion_details, "reasoning_tokens", 0) or 0)
    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def extract_answer(response: Any) -> str:
    choices = value_from(response, "choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    message = value_from(first_choice, "message", {})
    return str(value_from(message, "content", "") or "").strip()


def extract_delta_content(chunk: Any) -> str:
    choices = value_from(chunk, "choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    delta = value_from(first_choice, "delta", {})
    return str(value_from(delta, "content", "") or "")


def extract_stream_error(chunk: Any) -> str | None:
    error = value_from(chunk, "error")
    if not error:
        return None
    message = value_from(error, "message")
    code = value_from(error, "code")
    if message and code:
        return f"{code}: {message}"
    if message:
        return str(message)
    return str(error)


class RobustLLMClient:
    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        client_factory: Callable[[ProviderConfig], ChatClient] | None = None,
        max_retries: int = 5,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 16.0,
        max_jitter_sec: float = 0.25,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_sec: float = 60.0,
        rate_limit_per_minute: int = 0,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        if not providers:
            raise ConfigError("At least one provider is required.")

        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.max_jitter_sec = max_jitter_sec
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self.random_fn = random_fn
        self.events: list[ClientLogEvent] = []
        self.usage_by_provider: dict[str, UsageStats] = {
            provider.name: UsageStats() for provider in providers
        }

        factory = client_factory or self._openai_client_factory
        self.providers = [
            ProviderRuntime(
                config=provider,
                client=factory(provider),
                circuit_breaker=CircuitBreaker(
                    failure_threshold=circuit_failure_threshold,
                    cooldown_sec=circuit_cooldown_sec,
                ),
                rate_limiter=RateLimiter(rate_limit_per_minute),
            )
            for provider in providers
        ]

    @classmethod
    def from_env(cls, env_path: str | os.PathLike[str] | None = None) -> "RobustLLMClient":
        if load_dotenv is not None and env_path is not None:
            load_dotenv(env_path, override=False)

        timeout_sec = env_float("LLM_TIMEOUT_SEC", 60.0)
        providers: list[ProviderConfig] = []

        openai_api_key = env_str("OPENAI_API_KEY")
        openai_model = env_str("OPENAI_MODEL")
        if openai_api_key and openai_model:
            providers.append(
                ProviderConfig(
                    name="openai",
                    api_key=openai_api_key,
                    base_url=normalize_openai_base_url(
                        env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        provider="openai",
                    ),
                    model=openai_model,
                    input_price_per_1m=env_float("OPENAI_INPUT_PRICE_PER_1M", 0.0),
                    output_price_per_1m=env_float("OPENAI_OUTPUT_PRICE_PER_1M", 0.0),
                    timeout_sec=timeout_sec,
                    max_tokens_param="max_completion_tokens",
                )
            )

        openrouter_api_key = env_str("OPENROUTER_API_KEY")
        openrouter_model = env_str("OPENROUTER_MODEL")
        if openrouter_api_key and openrouter_model:
            providers.append(
                ProviderConfig(
                    name="openrouter",
                    api_key=openrouter_api_key,
                    base_url=normalize_openai_base_url(
                        env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                        provider="openrouter",
                    ),
                    model=openrouter_model,
                    input_price_per_1m=env_float("OPENROUTER_INPUT_PRICE_PER_1M", 0.0),
                    output_price_per_1m=env_float("OPENROUTER_OUTPUT_PRICE_PER_1M", 0.0),
                    timeout_sec=timeout_sec,
                    max_tokens_param="max_tokens",
                )
            )

        ollama_model = env_str("OLLAMA_MODEL") or first_csv_item(env_str("OLLAMA_MODELS"))
        if ollama_model:
            providers.append(
                ProviderConfig(
                    name="ollama",
                    api_key=env_str("OLLAMA_API_KEY", "ollama"),
                    base_url=normalize_openai_base_url(
                        env_str("OLLAMA_OPENAI_BASE_URL") or env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
                        provider="ollama",
                    ),
                    model=ollama_model,
                    input_price_per_1m=0.0,
                    output_price_per_1m=0.0,
                    timeout_sec=timeout_sec,
                    max_tokens_param="max_tokens",
                    stream_include_usage=False,
                )
            )

        if not providers:
            raise ConfigError(
                "Configure at least one provider: OPENAI_API_KEY/OPENAI_MODEL, "
                "OPENROUTER_API_KEY/OPENROUTER_MODEL, or OLLAMA_MODEL."
            )

        return cls(
            providers=providers,
            max_retries=env_int("ROBUST_LLM_MAX_RETRIES", 5),
            base_backoff_sec=env_float("ROBUST_LLM_BASE_BACKOFF_SEC", 1.0),
            max_backoff_sec=env_float("ROBUST_LLM_MAX_BACKOFF_SEC", 16.0),
            max_jitter_sec=env_float("ROBUST_LLM_MAX_JITTER_SEC", 0.25),
            circuit_failure_threshold=env_int("ROBUST_LLM_CIRCUIT_FAILURES", 3),
            circuit_cooldown_sec=env_float("ROBUST_LLM_CIRCUIT_COOLDOWN_SEC", 60.0),
            rate_limit_per_minute=env_int("ROBUST_LLM_RATE_LIMIT_PER_MINUTE", 0),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> LLMResult:
        for runtime in self.providers:
            if runtime.circuit_breaker.is_open(self.monotonic_fn()):
                remaining = runtime.circuit_breaker.opened_until - self.monotonic_fn()
                self._log(
                    runtime.config,
                    attempt=0,
                    code="circuit_open",
                    message=f"provider skipped for {max(0.0, remaining):.1f}s",
                )
                continue

            result = self._try_provider(runtime, messages, temperature, max_tokens)
            if result is not None:
                return result

        self.events.append(
            ClientLogEvent(
                timestamp=utc_timestamp(),
                provider="fallback",
                attempt=0,
                code="service_unavailable",
                message="all providers failed or were skipped",
            )
        )
        return LLMResult(
            answer=SERVICE_UNAVAILABLE_MESSAGE,
            provider=None,
            model=None,
            usage=UsageStats(),
            cost_usd=0.0,
            unavailable=True,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        on_token: Callable[[str], None],
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> LLMResult:
        for runtime in self.providers:
            if runtime.circuit_breaker.is_open(self.monotonic_fn()):
                remaining = runtime.circuit_breaker.opened_until - self.monotonic_fn()
                self._log(
                    runtime.config,
                    attempt=0,
                    code="circuit_open",
                    message=f"provider skipped for {max(0.0, remaining):.1f}s",
                )
                continue

            result = self._try_provider_stream(runtime, messages, temperature, max_tokens, on_token)
            if result is not None:
                return result

        self.events.append(
            ClientLogEvent(
                timestamp=utc_timestamp(),
                provider="fallback",
                attempt=0,
                code="service_unavailable",
                message="all providers failed or were skipped",
            )
        )
        on_token(SERVICE_UNAVAILABLE_MESSAGE)
        return LLMResult(
            answer=SERVICE_UNAVAILABLE_MESSAGE,
            provider=None,
            model=None,
            usage=UsageStats(),
            cost_usd=0.0,
            unavailable=True,
        )

    def _try_provider(
        self,
        runtime: ProviderRuntime,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResult | None:
        config = runtime.config

        for attempt in range(1, self.max_retries + 2):
            waited = runtime.rate_limiter.acquire(self.monotonic_fn, self.sleep_fn)
            if waited > 0:
                self._log(config, attempt=attempt, code="rate_limited", message="client-side wait", delay_sec=waited)

            try:
                response = runtime.client.chat.completions.create(
                    **self._request_params(config, messages, temperature, max_tokens, stream=False)
                )
                usage = extract_usage(response)
                usage.cost_usd = calculate_cost(
                    usage,
                    config.input_price_per_1m,
                    config.output_price_per_1m,
                )
                self.usage_by_provider[config.name].add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost_usd,
                    total_tokens=usage.total_tokens,
                    cached_tokens=usage.cached_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                )
                runtime.circuit_breaker.record_success()
                self._log(config, attempt=attempt, code="success", message="provider returned response")
                return LLMResult(
                    answer=extract_answer(response),
                    provider=config.name,
                    model=config.model,
                    usage=usage,
                    cost_usd=usage.cost_usd,
                )

            except Exception as exc:  # noqa: BLE001 - provider boundary should normalize all errors
                code = error_code(exc)
                retryable = is_retryable_error(exc)
                safe_message = sanitize_error(str(exc), config.api_key)
                self._log(config, attempt=attempt, code=code, message=safe_message)

                if retryable and attempt <= self.max_retries:
                    delay = self._retry_delay(attempt)
                    self._log(
                        config,
                        attempt=attempt,
                        code="retry_scheduled",
                        message=f"retryable error {code}",
                        delay_sec=delay,
                    )
                    self.sleep_fn(delay)
                    continue

                circuit_opened = runtime.circuit_breaker.record_failure(self.monotonic_fn())
                failure_message = "retry budget exhausted" if retryable else "non-retryable error"
                self._log(config, attempt=attempt, code="provider_failed", message=failure_message)
                if circuit_opened:
                    self._log(
                        config,
                        attempt=attempt,
                        code="circuit_opened",
                        message=f"provider skipped for {runtime.circuit_breaker.cooldown_sec:.0f}s",
                    )
                return None

        return None

    def _try_provider_stream(
        self,
        runtime: ProviderRuntime,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        on_token: Callable[[str], None],
    ) -> LLMResult | None:
        config = runtime.config

        for attempt in range(1, self.max_retries + 2):
            waited = runtime.rate_limiter.acquire(self.monotonic_fn, self.sleep_fn)
            if waited > 0:
                self._log(config, attempt=attempt, code="rate_limited", message="client-side wait", delay_sec=waited)

            chunks: list[str] = []
            emitted_chars = 0
            usage = UsageStats()

            try:
                stream = runtime.client.chat.completions.create(
                    **self._request_params(config, messages, temperature, max_tokens, stream=True)
                )

                for chunk in stream:
                    stream_error = extract_stream_error(chunk)
                    if stream_error:
                        raise RuntimeError(f"stream error: {stream_error}")

                    content = extract_delta_content(chunk)
                    if content:
                        emitted_chars += len(content)
                        chunks.append(content)
                        on_token(content)

                    chunk_usage = extract_usage(chunk)
                    if chunk_usage.total_tokens:
                        usage = chunk_usage

                answer = "".join(chunks).strip()
                usage.cost_usd = calculate_cost(
                    usage,
                    config.input_price_per_1m,
                    config.output_price_per_1m,
                )
                self.usage_by_provider[config.name].add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost_usd,
                    total_tokens=usage.total_tokens,
                    cached_tokens=usage.cached_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                )
                runtime.circuit_breaker.record_success()
                self._log(config, attempt=attempt, code="success", message="provider streamed response")
                return LLMResult(
                    answer=answer,
                    provider=config.name,
                    model=config.model,
                    usage=usage,
                    cost_usd=usage.cost_usd,
                )

            except Exception as exc:  # noqa: BLE001 - provider boundary should normalize all errors
                code = error_code(exc)
                retryable = is_retryable_error(exc)
                safe_message = sanitize_error(str(exc), config.api_key)
                self._log(config, attempt=attempt, code=code, message=safe_message)

                if emitted_chars > 0:
                    note = "\n\n[Потоковый ответ прерван провайдером; повторите запрос или выполните эскалацию.]"
                    on_token(note)
                    runtime.circuit_breaker.record_failure(self.monotonic_fn())
                    self._log(config, attempt=attempt, code="stream_interrupted", message="partial answer returned")
                    answer = "".join(chunks).strip() + note
                    return LLMResult(
                        answer=answer,
                        provider=config.name,
                        model=config.model,
                        usage=usage,
                        cost_usd=0.0,
                        unavailable=True,
                    )

                if retryable and attempt <= self.max_retries:
                    delay = self._retry_delay(attempt)
                    self._log(
                        config,
                        attempt=attempt,
                        code="retry_scheduled",
                        message=f"retryable stream error {code}",
                        delay_sec=delay,
                    )
                    self.sleep_fn(delay)
                    continue

                circuit_opened = runtime.circuit_breaker.record_failure(self.monotonic_fn())
                failure_message = "stream retry budget exhausted" if retryable else "non-retryable stream error"
                self._log(config, attempt=attempt, code="provider_failed", message=failure_message)
                if circuit_opened:
                    self._log(
                        config,
                        attempt=attempt,
                        code="circuit_opened",
                        message=f"provider skipped for {runtime.circuit_breaker.cooldown_sec:.0f}s",
                    )
                return None

        return None

    def _retry_delay(self, failed_attempt: int) -> float:
        backoff = min(
            self.base_backoff_sec * (2 ** (failed_attempt - 1)),
            self.max_backoff_sec,
        )
        jitter = self.random_fn() * self.max_jitter_sec
        return backoff + jitter

    def _request_params(
        self,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens > 0:
            params[config.max_tokens_param] = max_tokens
        if stream:
            params["stream"] = True
            if config.stream_include_usage:
                params["stream_options"] = {"include_usage": True}
        return params

    def _openai_client_factory(self, config: ProviderConfig) -> ChatClient:
        if OpenAI is None:
            raise ConfigError("Install OpenAI SDK first: python -m pip install openai")
        return OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
            max_retries=0,
        )

    def _log(
        self,
        config: ProviderConfig,
        *,
        attempt: int,
        code: str,
        message: str,
        delay_sec: float | None = None,
    ) -> None:
        self.events.append(
            ClientLogEvent(
                timestamp=utc_timestamp(),
                provider=config.name,
                attempt=attempt,
                code=code,
                message=message,
                delay_sec=delay_sec,
            )
        )

    def formatted_log(self) -> str:
        return "\n".join(event.to_line() for event in self.events)

    def usage_summary(self) -> dict[str, UsageStats]:
        return {
            provider: UsageStats(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cost_usd=usage.cost_usd,
            )
            for provider, usage in self.usage_by_provider.items()
        }
