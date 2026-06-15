from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from support_copilot.config import load_settings
from support_copilot.infrastructure.cache import LLMCache
from support_copilot.infrastructure.llm import (
    CircuitBreaker,
    ConfigError,
    LLMResult,
    ProviderConfig,
    UsageStats,
    calculate_cost,
    error_code,
    extract_answer,
    extract_delta_content,
    extract_stream_error,
    extract_usage,
    normalize_openai_base_url,
    sanitize_error,
)

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - fake tests do not need OpenAI SDK
    AsyncOpenAI = None


ASYNC_SERVICE_UNAVAILABLE_MESSAGE = (
    "LLM-провайдеры временно недоступны. Передайте обращение инженеру поддержки "
    "и повторите запрос позже."
)


class AsyncLLMUnavailableError(RuntimeError):
    pass


class AsyncChatClient(Protocol):
    chat: Any


@dataclass
class AsyncProviderRuntime:
    config: ProviderConfig
    client: AsyncChatClient
    circuit_breaker: CircuitBreaker


@dataclass(frozen=True)
class AsyncCompletionResult:
    answer: str
    provider: str
    model: str
    usage: UsageStats
    cost_usd: float
    cached: bool = False


async def _await_with_business_timeout(coro: Any, timeout_sec: float) -> Any:
    if hasattr(asyncio, "timeout"):
        async with asyncio.timeout(timeout_sec):  # type: ignore[attr-defined]
            return await coro
    return await asyncio.wait_for(coro, timeout=timeout_sec)


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


class AsyncLLMClient:
    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        client_factory: Callable[[ProviderConfig], AsyncChatClient] | None = None,
        concurrency: int = 5,
        sdk_max_retries: int = 3,
        business_timeout_sec: float = 15.0,
        cache: LLMCache | None = None,
        logger: logging.Logger | None = None,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_sec: float = 60.0,
    ) -> None:
        if not providers:
            raise ConfigError("At least one async provider is required.")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        self.concurrency = concurrency
        self.sdk_max_retries = sdk_max_retries
        self.business_timeout_sec = business_timeout_sec
        self.cache = cache
        self.logger = logger or logging.getLogger(__name__)
        self.usage_by_provider: dict[str, UsageStats] = {
            provider.name: UsageStats() for provider in providers
        }
        self._sem = asyncio.Semaphore(concurrency)

        factory = client_factory or self._openai_client_factory
        self.providers = [
            AsyncProviderRuntime(
                config=provider,
                client=factory(provider),
                circuit_breaker=CircuitBreaker(
                    failure_threshold=circuit_failure_threshold,
                    cooldown_sec=circuit_cooldown_sec,
                ),
            )
            for provider in providers
        ]

    @classmethod
    def from_env(
        cls,
        env_path: str | None = None,
        *,
        concurrency: int | None = None,
        enable_cache: bool = False,
        logger: logging.Logger | None = None,
    ) -> "AsyncLLMClient":
        settings = load_settings(env_path) if env_path is not None else load_settings()
        sdk_timeout_sec = settings.async_llm_sdk_timeout_sec
        providers: list[ProviderConfig] = []

        if settings.openai_api_key and settings.openai_model:
            providers.append(
                ProviderConfig(
                    name="openai",
                    api_key=settings.openai_api_key,
                    base_url=normalize_openai_base_url(settings.openai_base_url, provider="openai"),
                    model=settings.openai_model,
                    input_price_per_1m=settings.openai_input_price_per_1m,
                    output_price_per_1m=settings.openai_output_price_per_1m,
                    timeout_sec=sdk_timeout_sec,
                    max_tokens_param="max_completion_tokens",
                )
            )

        if settings.openrouter_api_key and settings.openrouter_model:
            providers.append(
                ProviderConfig(
                    name="openrouter",
                    api_key=settings.openrouter_api_key,
                    base_url=normalize_openai_base_url(settings.openrouter_base_url, provider="openrouter"),
                    model=settings.openrouter_model,
                    input_price_per_1m=settings.openrouter_input_price_per_1m,
                    output_price_per_1m=settings.openrouter_output_price_per_1m,
                    timeout_sec=sdk_timeout_sec,
                    max_tokens_param="max_tokens",
                )
            )

        ollama_model = settings.selected_ollama_model
        if ollama_model:
            providers.append(
                ProviderConfig(
                    name="ollama",
                    api_key=settings.ollama_api_key,
                    base_url=normalize_openai_base_url(
                        settings.ollama_openai_compatible_base_url,
                        provider="ollama",
                    ),
                    model=ollama_model,
                    timeout_sec=sdk_timeout_sec,
                    max_tokens_param="max_tokens",
                    stream_include_usage=False,
                )
            )

        if not providers:
            raise ConfigError(
                "Configure at least one async provider: OPENAI_API_KEY/OPENAI_MODEL, "
                "OPENROUTER_API_KEY/OPENROUTER_MODEL, or OLLAMA_MODEL."
            )

        cache = None
        if enable_cache:
            cache = LLMCache(
                ttl_seconds=settings.llm_cache_ttl_seconds,
                prompt_version=settings.llm_cache_prompt_version,
                service_facts_version=settings.llm_cache_service_facts_version,
            )

        return cls(
            providers,
            concurrency=concurrency or settings.async_llm_concurrency,
            sdk_max_retries=settings.async_llm_max_retries,
            business_timeout_sec=settings.async_llm_business_timeout_sec,
            cache=cache,
            logger=logger,
            circuit_failure_threshold=settings.robust_llm_circuit_failures,
            circuit_cooldown_sec=settings.robust_llm_circuit_cooldown_sec,
        )

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        start = time.perf_counter()
        status = "error"
        model = self.providers[0].config.model if self.providers else "-"

        try:
            async with self._sem:
                result = await _await_with_business_timeout(
                    self._complete_result(
                        prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    self.business_timeout_sec,
                )
            status = "cache_hit" if result.cached else "success"
            model = result.model
            return result.answer
        except Exception:
            status = "error"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.logger.info(
                "llm.call duration_ms=%.2f model=%s prompt_chars=%d status=%s",
                duration_ms,
                model,
                len(prompt),
                status,
                extra={
                    "duration_ms": duration_ms,
                    "model": model,
                    "prompt_chars": len(prompt),
                    "status": status,
                },
            )

    async def batch_chat(
        self,
        prompts: list[str],
        concurrency: int = 5,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> list[str | Exception]:
        if concurrency != self.concurrency:
            raise ValueError(
                "batch_chat concurrency must match the instance semaphore. "
                f"Create AsyncLLMClient(concurrency={concurrency}) first; "
                f"current concurrency is {self.concurrency}."
            )

        coros = [
            self.complete(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    async def stream_chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> AsyncIterator[str]:
        messages = self._messages(prompt, system_prompt)

        async with self._sem:
            async for delta in self._stream_with_fallback(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield delta

    async def _complete_result(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> AsyncCompletionResult:
        messages = self._messages(prompt, system_prompt)
        cache_model = self._cache_model_id()
        if self.cache is not None:
            cached = self.cache.get(cache_model, messages, temperature)
            if cached is not None:
                return AsyncCompletionResult(
                    answer=str(cached),
                    provider="cache",
                    model=cache_model,
                    usage=UsageStats(),
                    cost_usd=0.0,
                    cached=True,
                )

        errors: list[str] = []
        for runtime in self.providers:
            if runtime.circuit_breaker.is_open(time.monotonic()):
                errors.append(f"{runtime.config.name}: circuit_open")
                continue

            try:
                response = await runtime.client.chat.completions.create(
                    **self._request_params(
                        runtime.config,
                        messages,
                        temperature,
                        max_tokens,
                        stream=False,
                    )
                )
                usage = extract_usage(response)
                usage.cost_usd = calculate_cost(
                    usage,
                    runtime.config.input_price_per_1m,
                    runtime.config.output_price_per_1m,
                )
                self._record_usage(runtime.config, usage)
                runtime.circuit_breaker.record_success()
                answer = extract_answer(response)
                if self.cache is not None:
                    self.cache.set(cache_model, messages, temperature, answer)
                return AsyncCompletionResult(
                    answer=answer,
                    provider=runtime.config.name,
                    model=runtime.config.model,
                    usage=usage,
                    cost_usd=usage.cost_usd,
                )
            except Exception as exc:  # noqa: BLE001 - provider boundary normalizes failures
                runtime.circuit_breaker.record_failure(time.monotonic())
                safe_message = sanitize_error(str(exc), runtime.config.api_key)
                code = error_code(exc)
                errors.append(f"{runtime.config.name}: {code}: {safe_message}")
                self.logger.warning(
                    "llm.provider_failed provider=%s model=%s code=%s message=%s",
                    runtime.config.name,
                    runtime.config.model,
                    code,
                    safe_message,
                )

        joined_errors = "; ".join(errors) if errors else "no providers configured"
        raise AsyncLLMUnavailableError(f"{ASYNC_SERVICE_UNAVAILABLE_MESSAGE} Details: {joined_errors}")

    async def _stream_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        errors: list[str] = []

        for runtime in self.providers:
            if runtime.circuit_breaker.is_open(time.monotonic()):
                errors.append(f"{runtime.config.name}: circuit_open")
                continue

            chunks: list[str] = []
            usage = UsageStats()
            start = time.perf_counter()
            first_token_at: float | None = None

            try:
                stream = await runtime.client.chat.completions.create(
                    **self._request_params(
                        runtime.config,
                        messages,
                        temperature,
                        max_tokens,
                        stream=True,
                    )
                )

                async for chunk in stream:
                    stream_error = extract_stream_error(chunk)
                    if stream_error:
                        raise RuntimeError(f"stream error: {stream_error}")

                    delta = extract_delta_content(chunk)
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(delta)
                        yield delta

                    chunk_usage = extract_usage(chunk)
                    if chunk_usage.total_tokens:
                        usage = chunk_usage

                usage.cost_usd = calculate_cost(
                    usage,
                    runtime.config.input_price_per_1m,
                    runtime.config.output_price_per_1m,
                )
                self._record_usage(runtime.config, usage)
                runtime.circuit_breaker.record_success()
                total_ms = (time.perf_counter() - start) * 1000
                ttft_ms = None if first_token_at is None else (first_token_at - start) * 1000
                self.logger.info(
                    "llm.stream.closed model=%s total_tokens=%d ttft_ms=%s total_ms=%.2f",
                    runtime.config.model,
                    usage.total_tokens,
                    "none" if ttft_ms is None else f"{ttft_ms:.2f}",
                    total_ms,
                    extra={
                        "model": runtime.config.model,
                        "total_tokens": usage.total_tokens,
                        "ttft_ms": ttft_ms,
                        "total_ms": total_ms,
                    },
                )
                return
            except Exception as exc:  # noqa: BLE001 - provider boundary normalizes failures
                runtime.circuit_breaker.record_failure(time.monotonic())
                safe_message = sanitize_error(str(exc), runtime.config.api_key)
                code = error_code(exc)
                errors.append(f"{runtime.config.name}: {code}: {safe_message}")
                self.logger.warning(
                    "llm.stream.provider_failed provider=%s model=%s code=%s message=%s",
                    runtime.config.name,
                    runtime.config.model,
                    code,
                    safe_message,
                )
                if chunks:
                    yield "\n\n[Потоковый ответ прерван провайдером; повторите запрос или выполните эскалацию.]"
                    return

        joined_errors = "; ".join(errors) if errors else "no providers configured"
        raise AsyncLLMUnavailableError(f"{ASYNC_SERVICE_UNAVAILABLE_MESSAGE} Details: {joined_errors}")

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

    def _messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _cache_model_id(self) -> str:
        return "async:" + "->".join(runtime.config.model for runtime in self.providers)

    def _record_usage(self, config: ProviderConfig, usage: UsageStats) -> None:
        self.usage_by_provider[config.name].add(
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cost_usd,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )

    def _openai_client_factory(self, config: ProviderConfig) -> AsyncChatClient:
        if AsyncOpenAI is None:
            raise ConfigError("Install OpenAI SDK first: python -m pip install openai")
        return AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
            max_retries=self.sdk_max_retries,
        )

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


@dataclass(frozen=True)
class _DemoUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class _DemoMessage:
    content: str


@dataclass(frozen=True)
class _DemoChoice:
    message: _DemoMessage


@dataclass(frozen=True)
class _DemoResponse:
    content: str
    usage: _DemoUsage

    @property
    def choices(self) -> list[_DemoChoice]:
        return [_DemoChoice(_DemoMessage(self.content))]


class _DemoAsyncStream:
    def __init__(self, answer: str, usage: dict[str, int], delay_sec: float) -> None:
        self._tokens = answer.split(" ")
        self._usage = usage
        self._delay_sec = delay_sec
        self._index = 0
        self._usage_sent = False

    def __aiter__(self) -> "_DemoAsyncStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index < len(self._tokens):
            await asyncio.sleep(self._delay_sec)
            token = self._tokens[self._index]
            self._index += 1
            suffix = "" if self._index == len(self._tokens) else " "
            return {"choices": [{"delta": {"content": token + suffix}}]}
        if not self._usage_sent:
            self._usage_sent = True
            return {"choices": [{"delta": {"content": ""}}], "usage": self._usage}
        raise StopAsyncIteration


class _DemoAsyncCompletions:
    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = delay_sec

    async def create(self, **params: Any) -> Any:
        messages = params.get("messages", [])
        prompt = _last_user_text(messages)
        if "FORCE_ASYNC_ERROR" in prompt:
            raise RuntimeError("demo async failure")

        answer = (
            "Краткий ответ Support Copilot: зафиксируйте контекст обращения, "
            "уточните версию продукта и передайте инженеру только при признаках блокировки работы."
        )
        prompt_tokens = max(1, len(prompt.split()) * 2)
        completion_tokens = len(answer.split()) * 2
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if params.get("stream"):
            return _DemoAsyncStream(answer, usage, self.delay_sec)

        await asyncio.sleep(self.delay_sec)
        return _DemoResponse(
            content=answer,
            usage=_DemoUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            ),
        )


class _DemoAsyncChat:
    def __init__(self, delay_sec: float) -> None:
        self.completions = _DemoAsyncCompletions(delay_sec)


class _DemoAsyncClient:
    def __init__(self, delay_sec: float) -> None:
        self.chat = _DemoAsyncChat(delay_sec)


def build_demo_async_client(
    *,
    concurrency: int = 5,
    delay_sec: float = 0.1,
    cache: LLMCache | None = None,
    logger: logging.Logger | None = None,
) -> AsyncLLMClient:
    provider = ProviderConfig(
        name="demo",
        api_key="demo",
        base_url="https://example.invalid/v1",
        model="demo-async-model",
        timeout_sec=30,
        max_tokens_param="max_tokens",
    )
    return AsyncLLMClient(
        [provider],
        client_factory=lambda _config: _DemoAsyncClient(delay_sec),
        concurrency=concurrency,
        sdk_max_retries=3,
        business_timeout_sec=15,
        cache=cache,
        logger=logger,
    )
