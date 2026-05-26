"""Robust OpenAI-compatible LLM client with retry, fallback and usage tracking."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0
    timeout_sec: float = 60.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


@dataclass
class LLMResult:
    content: str
    provider: str
    model: str
    usage: LLMUsage
    fallback_used: bool = False
    unavailable: bool = False
    attempts: int = 0


class ChatTransport(Protocol):
    def __call__(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        ...


class RobustLLMClient:
    """OpenAI SDK wrapper with retry only for temporary failures."""

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        max_attempts: int = 5,
        backoff_seconds: tuple[float, ...] = (1, 2, 4, 8, 16),
        jitter_seconds: float = 0.25,
        sleep: Callable[[float], None] | None = None,
        transport: ChatTransport | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.providers = providers
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.jitter_seconds = jitter_seconds
        self.sleep = sleep or time.sleep
        self.transport = transport or self._openai_chat
        self.log_file = Path(log_file) if log_file else None
        self.usage = LLMUsage()
        self.events: list[str] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> LLMResult:
        total_attempts = 0
        configured = [provider for provider in self.providers if provider.is_configured]
        if not configured:
            return self._unavailable("no-configured-provider", total_attempts)

        for provider_index, provider in enumerate(configured):
            for attempt in range(1, self.max_attempts + 1):
                total_attempts += 1
                try:
                    response = self.transport(provider, messages, temperature, max_tokens)
                    content, usage = self._extract_response(response, provider)
                    self.usage.add(usage)
                    self._log(
                        provider=provider.name,
                        attempt=attempt,
                        status="ok",
                        message=f"tokens={usage.total_tokens} cost={usage.cost_usd:.8f}",
                    )
                    return LLMResult(
                        content=content,
                        provider=provider.name,
                        model=provider.model,
                        usage=usage,
                        fallback_used=provider_index > 0,
                        attempts=total_attempts,
                    )
                except Exception as exc:
                    status_code = _status_code(exc)
                    retryable = _is_retryable(exc)
                    self._log(
                        provider=provider.name,
                        attempt=attempt,
                        status=str(status_code or exc.__class__.__name__),
                        message=f"retryable={retryable} error={_short_error(exc)}",
                    )
                    if not retryable or attempt >= self.max_attempts:
                        break
                    self.sleep(self._sleep_for_attempt(attempt))

        return self._unavailable("providers-exhausted", total_attempts)

    def stats(self) -> dict[str, Any]:
        return {
            "usage": self.usage.as_dict(),
            "events": len(self.events),
            "providers": [provider.name for provider in self.providers],
        }

    def _openai_chat(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=provider.timeout_sec,
        )
        token_limit_param = _completion_tokens_parameter(provider.model)
        return client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=temperature,
            **{token_limit_param: max_tokens},
        )

    def _extract_response(self, response: Any, provider: ProviderConfig) -> tuple[str, LLMUsage]:
        if isinstance(response, dict):
            content = response["choices"][0]["message"]["content"]
            raw_usage = response.get("usage", {})
            prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
            completion_tokens = int(raw_usage.get("completion_tokens", 0))
            total_tokens = int(raw_usage.get("total_tokens", prompt_tokens + completion_tokens))
        else:
            content = response.choices[0].message.content
            raw_usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
            total_tokens = int(
                getattr(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0
            )

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=_estimate_cost(provider, prompt_tokens, completion_tokens),
        )
        return str(content), usage

    def _sleep_for_attempt(self, attempt: int) -> float:
        base = self.backoff_seconds[min(attempt - 1, len(self.backoff_seconds) - 1)]
        return base + random.uniform(0, self.jitter_seconds)

    def _unavailable(self, reason: str, attempts: int) -> LLMResult:
        self._log(provider="fallback", attempt=attempts, status=reason, message="unavailable")
        return LLMResult(
            content=(
                "Сервис временно недоступен. Передаю вопрос специалисту поддержки; "
                "попробуйте повторить запрос позже."
            ),
            provider="fallback",
            model="none",
            usage=LLMUsage(),
            fallback_used=True,
            unavailable=True,
            attempts=attempts,
        )

    def _log(self, *, provider: str, attempt: int, status: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = (
            f"[{timestamp}] provider={provider} attempt={attempt} "
            f"status={status} {message}"
        )
        self.events.append(line)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


def _estimate_cost(provider: ProviderConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * provider.input_price_per_1m / 1_000_000
        + completion_tokens * provider.output_price_per_1m / 1_000_000
    )


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "timedout" in name


def _short_error(exc: Exception, limit: int = 220) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _completion_tokens_parameter(model: str) -> str:
    if model.startswith("gpt-5") or model.startswith("o"):
        return "max_completion_tokens"
    return "max_tokens"
