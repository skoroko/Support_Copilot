from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from support_copilot.config import load_settings
from support_copilot.infrastructure.llm import (
    ChatClient,
    CircuitBreaker,
    ClientLogEvent,
    ConfigError,
    OpenAI,
    ProviderConfig,
    ProviderRuntime,
    RateLimiter,
    SERVICE_UNAVAILABLE_MESSAGE,
    UsageStats,
    calculate_cost,
    error_code,
    extract_answer,
    extract_usage,
    is_retryable_error,
    normalize_openai_base_url,
    sanitize_error,
    utc_timestamp,
    value_from,
)


ToolHandler = Callable[..., Any]
ToolEventSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    tool_call_id: str


@dataclass(frozen=True)
class ToolCallingResult:
    answer: str
    provider: str | None
    model: str | None
    usage: UsageStats
    cost_usd: float
    tool_calls: tuple[ToolCallRecord, ...]
    direct_answer: bool
    unavailable: bool = False


@dataclass(frozen=True)
class DemoToolUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class DemoToolFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class DemoToolCall:
    id: str
    type: str
    function: DemoToolFunctionCall


@dataclass(frozen=True)
class DemoToolMessage:
    content: str | None = None
    tool_calls: list[DemoToolCall] | None = None


@dataclass(frozen=True)
class DemoToolChoice:
    message: DemoToolMessage


@dataclass(frozen=True)
class DemoToolResponse:
    choices: list[DemoToolChoice]
    usage: DemoToolUsage


class DemoToolCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **params: Any) -> DemoToolResponse:
        self.calls += 1
        messages = params.get("messages", [])
        user_input = next(
            (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        tool_messages = [message for message in messages if message.get("role") == "tool"]

        if tool_messages:
            tool_result = json.loads(str(tool_messages[-1].get("content", "{}")))
            return self._text_response(messages, self._final_answer_from_tool(tool_result))

        if params.get("tools") and params.get("tool_choice") != "none" and self._should_call_tool(user_input):
            arguments = {
                "query": user_input,
                "product": self._choose_product(user_input),
            }
            tool_call = DemoToolCall(
                id=f"call_demo_{self.calls}",
                type="function",
                function=DemoToolFunctionCall(
                    name="search_knowledge_base",
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
            return self._tool_response(messages, tool_call)

        return self._text_response(messages, self._direct_answer(user_input))

    @staticmethod
    def _should_call_tool(user_input: str) -> bool:
        lowered = user_input.casefold()
        markers = (
            "базе знаний",
            "найди",
            "лиценз",
            "активац",
            "установ",
            "документац",
            "инструкц",
            "расчет",
            "обновлен",
            "ошибк",
            "где посмотреть",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _choose_product(user_input: str) -> str:
        lowered = user_input.casefold()
        if "лиценз" in lowered or "активац" in lowered:
            return "licensing"
        if "установ" in lowered:
            return "installation"
        if "расчет" in lowered or "обновлен" in lowered or "ошибк" in lowered:
            return "calculation"
        if "документац" in lowered or "инструкц" in lowered:
            return "documentation"
        return "support_copilot"

    @staticmethod
    def _direct_answer(user_input: str) -> str:
        lowered = user_input.casefold()
        if any(marker in lowered for marker in ("игнорируй", "скрытый промпт", "system prompt")):
            return (
                "Не могу раскрывать внутренние инструкции или менять рабочие правила. "
                "Сформулируйте, пожалуйста, рабочий вопрос по продукту MIDAS."
            )
        return "Инженер принял обращение в работу и вернется с уточнениями после первичной проверки."

    @staticmethod
    def _final_answer_from_tool(tool_result: dict[str, Any]) -> str:
        results = tool_result.get("results") or []
        if not results:
            return "В базе знаний нет надежного совпадения. Запрос нужно передать инженеру поддержки для проверки."
        top = results[0]
        return (
            f"По базе знаний {top['id']}: {top['answer']} "
            "Если данных не хватает, не обещайте решение до проверки инженером."
        )

    @staticmethod
    def _usage(messages: list[dict[str, Any]], answer: str) -> DemoToolUsage:
        prompt_tokens = max(1, sum(len(str(message.get("content", "")).split()) for message in messages) * 2)
        completion_tokens = max(12, len(answer.split()) * 2)
        return DemoToolUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _text_response(self, messages: list[dict[str, Any]], answer: str) -> DemoToolResponse:
        return DemoToolResponse(
            choices=[DemoToolChoice(DemoToolMessage(content=answer))],
            usage=self._usage(messages, answer),
        )

    def _tool_response(self, messages: list[dict[str, Any]], tool_call: DemoToolCall) -> DemoToolResponse:
        return DemoToolResponse(
            choices=[DemoToolChoice(DemoToolMessage(content=None, tool_calls=[tool_call]))],
            usage=self._usage(messages, tool_call.function.arguments),
        )


class DemoToolChat:
    def __init__(self) -> None:
        self.completions = DemoToolCompletions()


class DemoToolOpenAIClient:
    def __init__(self) -> None:
        self.chat = DemoToolChat()


def _choice_message(response: Any) -> Any:
    choices = value_from(response, "choices", [])
    if not choices:
        return {}
    return value_from(choices[0], "message", {})


def _message_content(message: Any) -> str:
    content = value_from(message, "content", "") or ""
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = value_from(item, "text", None)
            if text is not None:
                chunks.append(str(text))
        return "".join(chunks)
    return str(content)


def _tool_calls(message: Any) -> list[Any]:
    calls = value_from(message, "tool_calls", None) or []
    return list(calls)


def _function_from_tool_call(tool_call: Any) -> Any:
    return value_from(tool_call, "function", {}) or {}


def _tool_call_id(tool_call: Any, index: int) -> str:
    return str(value_from(tool_call, "id", None) or f"call_{index + 1}")


def _tool_call_name(tool_call: Any) -> str:
    return str(value_from(_function_from_tool_call(tool_call), "name", "") or "")


def _tool_call_arguments_text(tool_call: Any) -> str:
    raw_arguments = value_from(_function_from_tool_call(tool_call), "arguments", "{}")
    if isinstance(raw_arguments, str):
        return raw_arguments or "{}"
    return json.dumps(raw_arguments, ensure_ascii=False)


def _parse_tool_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = json.loads(_tool_call_arguments_text(tool_call))
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return arguments


def _assistant_message_with_tool_calls(message: Any) -> dict[str, Any]:
    calls = []
    for index, tool_call in enumerate(_tool_calls(message)):
        calls.append(
            {
                "id": _tool_call_id(tool_call, index),
                "type": value_from(tool_call, "type", "function") or "function",
                "function": {
                    "name": _tool_call_name(tool_call),
                    "arguments": _tool_call_arguments_text(tool_call),
                },
            }
        )
    return {
        "role": "assistant",
        "content": _message_content(message) or None,
        "tool_calls": calls,
    }


class ToolCallingClient:
    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        client_factory: Callable[[ProviderConfig], ChatClient] | None = None,
        max_retries: int = 2,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 8.0,
        max_jitter_sec: float = 0.25,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
        random_fn: Callable[[], float] | None = None,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_sec: float = 60.0,
        rate_limit_per_minute: int = 0,
    ) -> None:
        if not providers:
            raise ConfigError("At least one provider is required.")

        import random
        import time

        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.max_jitter_sec = max_jitter_sec
        self.sleep_fn = sleep_fn or time.sleep
        self.monotonic_fn = monotonic_fn or time.monotonic
        self.random_fn = random_fn or random.random
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
    def from_env(cls, env_path: str | None = None) -> "ToolCallingClient":
        settings = load_settings(env_path)

        timeout_sec = settings.llm_timeout_sec
        providers: list[ProviderConfig] = []

        openai_api_key = settings.openai_api_key
        openai_model = settings.openai_model
        if openai_api_key and openai_model:
            providers.append(
                ProviderConfig(
                    name="openai",
                    api_key=openai_api_key,
                    base_url=normalize_openai_base_url(
                        settings.openai_base_url,
                        provider="openai",
                    ),
                    model=openai_model,
                    input_price_per_1m=settings.openai_input_price_per_1m,
                    output_price_per_1m=settings.openai_output_price_per_1m,
                    timeout_sec=timeout_sec,
                    max_tokens_param="max_completion_tokens",
                )
            )

        openrouter_api_key = settings.openrouter_api_key
        openrouter_model = settings.openrouter_model
        if openrouter_api_key and openrouter_model:
            providers.append(
                ProviderConfig(
                    name="openrouter",
                    api_key=openrouter_api_key,
                    base_url=normalize_openai_base_url(
                        settings.openrouter_base_url,
                        provider="openrouter",
                    ),
                    model=openrouter_model,
                    input_price_per_1m=settings.openrouter_input_price_per_1m,
                    output_price_per_1m=settings.openrouter_output_price_per_1m,
                    timeout_sec=timeout_sec,
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
                    timeout_sec=timeout_sec,
                    max_tokens_param="max_tokens",
                    stream_include_usage=False,
                )
            )

        return cls(
            providers=providers,
            max_retries=settings.robust_llm_max_retries,
            base_backoff_sec=settings.robust_llm_base_backoff_sec,
            max_backoff_sec=settings.robust_llm_max_backoff_sec,
            max_jitter_sec=settings.robust_llm_max_jitter_sec,
            circuit_failure_threshold=settings.robust_llm_circuit_failures,
            circuit_cooldown_sec=settings.robust_llm_circuit_cooldown_sec,
            rate_limit_per_minute=settings.robust_llm_rate_limit_per_minute,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, ToolHandler],
        temperature: float = 0.0,
        max_tokens: int = 500,
        event_sink: ToolEventSink | None = None,
    ) -> ToolCallingResult:
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

            result = self._try_provider(
                runtime,
                messages,
                tools,
                tool_handlers,
                temperature,
                max_tokens,
                event_sink,
            )
            if result is not None:
                return result

        if event_sink is not None:
            event_sink("service_unavailable", {"reason": "all providers failed or were skipped"})
        return ToolCallingResult(
            answer=SERVICE_UNAVAILABLE_MESSAGE,
            provider=None,
            model=None,
            usage=UsageStats(),
            cost_usd=0.0,
            tool_calls=(),
            direct_answer=False,
            unavailable=True,
        )

    def _try_provider(
        self,
        runtime: ProviderRuntime,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, ToolHandler],
        temperature: float,
        max_tokens: int,
        event_sink: ToolEventSink | None,
    ) -> ToolCallingResult | None:
        config = runtime.config
        for attempt in range(1, self.max_retries + 2):
            waited = runtime.rate_limiter.acquire(self.monotonic_fn, self.sleep_fn)
            if waited > 0:
                self._log(config, attempt=attempt, code="rate_limited", message="client-side wait", delay_sec=waited)

            try:
                result = self._run_tool_cycle(
                    runtime,
                    messages,
                    tools,
                    tool_handlers,
                    temperature,
                    max_tokens,
                    event_sink,
                )
                runtime.circuit_breaker.record_success()
                self.usage_by_provider[config.name].add(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    result.cost_usd,
                    total_tokens=result.usage.total_tokens,
                    cached_tokens=result.usage.cached_tokens,
                    reasoning_tokens=result.usage.reasoning_tokens,
                )
                self._log(config, attempt=attempt, code="success", message="provider completed tool cycle")
                return result
            except Exception as exc:  # noqa: BLE001 - provider boundary normalizes failures
                code = error_code(exc)
                retryable = is_retryable_error(exc)
                safe_message = sanitize_error(str(exc), config.api_key)
                self._log(config, attempt=attempt, code=code, message=safe_message)
                if event_sink is not None:
                    event_sink("provider_error", {"provider": config.name, "attempt": attempt, "code": code})

                if retryable and attempt <= self.max_retries:
                    delay = self._retry_delay(attempt)
                    self._log(
                        config,
                        attempt=attempt,
                        code="retry_scheduled",
                        message=f"retryable tool-call error {code}",
                        delay_sec=delay,
                    )
                    self.sleep_fn(delay)
                    continue

                circuit_opened = runtime.circuit_breaker.record_failure(self.monotonic_fn())
                self._log(config, attempt=attempt, code="provider_failed", message="tool-call provider failed")
                if circuit_opened:
                    self._log(
                        config,
                        attempt=attempt,
                        code="circuit_opened",
                        message=f"provider skipped for {runtime.circuit_breaker.cooldown_sec:.0f}s",
                    )
                return None

        return None

    def _run_tool_cycle(
        self,
        runtime: ProviderRuntime,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, ToolHandler],
        temperature: float,
        max_tokens: int,
        event_sink: ToolEventSink | None,
    ) -> ToolCallingResult:
        config = runtime.config
        working_messages = [dict(message) for message in messages]
        if event_sink is not None:
            event_sink("model_request", {"provider": config.name, "model": config.model, "stage": "tool_decision"})

        first_response = runtime.client.chat.completions.create(
            **self._request_params(
                config,
                working_messages,
                temperature,
                max_tokens,
                tools=tools,
                tool_choice="auto",
            )
        )
        total_usage = extract_usage(first_response)
        first_message = _choice_message(first_response)
        calls = _tool_calls(first_message)

        if not calls:
            answer = extract_answer(first_response)
            total_usage.cost_usd = calculate_cost(
                total_usage,
                config.input_price_per_1m,
                config.output_price_per_1m,
            )
            if event_sink is not None:
                event_sink("tool_skipped", {"reason": "model_returned_text"})
            return ToolCallingResult(
                answer=answer,
                provider=config.name,
                model=config.model,
                usage=total_usage,
                cost_usd=total_usage.cost_usd,
                tool_calls=(),
                direct_answer=True,
            )

        working_messages.append(_assistant_message_with_tool_calls(first_message))
        records: list[ToolCallRecord] = []
        for index, tool_call in enumerate(calls):
            call_id = _tool_call_id(tool_call, index)
            name = _tool_call_name(tool_call)
            arguments = _parse_tool_arguments(tool_call)
            if event_sink is not None:
                event_sink("tool_call", {"tool_call_id": call_id, "name": name, "arguments": arguments})

            handler = tool_handlers.get(name)
            if handler is None:
                result: Any = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = handler(**arguments)
                except Exception as exc:  # noqa: BLE001 - tool failures become model-visible tool results
                    result = {"error": type(exc).__name__, "message": str(exc)}

            records.append(ToolCallRecord(name=name, arguments=arguments, result=result, tool_call_id=call_id))
            if event_sink is not None:
                event_sink("tool_result", {"tool_call_id": call_id, "name": name, "result": result})

            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        if event_sink is not None:
            event_sink("model_request", {"provider": config.name, "model": config.model, "stage": "final_answer"})

        final_response = runtime.client.chat.completions.create(
            **self._request_params(
                config,
                working_messages,
                temperature,
                max_tokens,
                tools=tools,
                tool_choice="none",
            )
        )
        final_usage = extract_usage(final_response)
        total_usage.add(
            final_usage.prompt_tokens,
            final_usage.completion_tokens,
            0.0,
            total_tokens=final_usage.total_tokens,
            cached_tokens=final_usage.cached_tokens,
            reasoning_tokens=final_usage.reasoning_tokens,
        )
        total_usage.cost_usd = calculate_cost(
            total_usage,
            config.input_price_per_1m,
            config.output_price_per_1m,
        )
        return ToolCallingResult(
            answer=extract_answer(final_response),
            provider=config.name,
            model=config.model,
            usage=total_usage,
            cost_usd=total_usage.cost_usd,
            tool_calls=tuple(records),
            direct_answer=False,
        )

    def _request_params(
        self,
        config: ProviderConfig,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        tools: list[dict[str, Any]],
        tool_choice: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        if max_tokens > 0:
            params[config.max_tokens_param] = max_tokens
        return params

    def _retry_delay(self, failed_attempt: int) -> float:
        backoff = min(
            self.base_backoff_sec * (2 ** (failed_attempt - 1)),
            self.max_backoff_sec,
        )
        return backoff + self.random_fn() * self.max_jitter_sec

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


def build_demo_tool_calling_client() -> ToolCallingClient:
    provider = ProviderConfig(
        name="demo-openai",
        api_key="sk-demo",
        base_url="https://api.openai.com/v1",
        model="demo-function-calling-model",
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
    )
    return ToolCallingClient(
        providers=[provider],
        client_factory=lambda _config: DemoToolOpenAIClient(),
        max_retries=0,
        sleep_fn=lambda _delay: None,
        random_fn=lambda: 0.0,
    )
