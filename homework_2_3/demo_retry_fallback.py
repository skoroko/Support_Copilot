from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_llm_client import ConfigError, ProviderConfig, RobustLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

DEMO_MESSAGES = [
    {
        "role": "system",
        "content": "Ты Support Copilot. Отвечай на русском и не выдумывай факты MIDAS.",
    },
    {
        "role": "user",
        "content": "Пользователь пишет: после обновления расчет падает с ошибкой. Подготовь черновик ответа.",
    },
]


class FakeProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, is_timeout: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_timeout = is_timeout


@dataclass(frozen=True)
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class FakeMessage:
    content: str


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeResponse:
    content: str
    usage: FakeUsage

    @property
    def choices(self) -> list[FakeChoice]:
        return [FakeChoice(FakeMessage(self.content))]


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_params: Any) -> Any:
        self.calls += 1
        if not self.outcomes:
            raise FakeProviderError("fake provider has no configured outcomes", status_code=500)

        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    def __init__(self, outcomes: list[Any]) -> None:
        self.completions = FakeCompletions(outcomes)


class FakeOpenAIClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.chat = FakeChat(outcomes)


def fake_success(text: str, prompt_tokens: int, completion_tokens: int) -> FakeResponse:
    return FakeResponse(
        content=text,
        usage=FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def build_demo_client() -> RobustLLMClient:
    providers = [
        ProviderConfig(
            name="openai",
            api_key="sk-demo-openai",
            base_url="https://api.openai.com/v1",
            model="env-openai-model",
            input_price_per_1m=0.75,
            output_price_per_1m=4.50,
        ),
        ProviderConfig(
            name="openrouter",
            api_key="sk-demo-openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="env-openrouter-model",
            input_price_per_1m=0.20,
            output_price_per_1m=0.60,
            max_tokens_param="max_tokens",
        ),
    ]

    scripted_outcomes: dict[str, list[Any]] = {
        "openai": [
            FakeProviderError("rate limit", status_code=429),
            FakeProviderError("internal server error", status_code=500),
            FakeProviderError("request timed out", is_timeout=True),
            FakeProviderError("bad gateway", status_code=502),
            FakeProviderError("rate limit again", status_code=429),
            FakeProviderError("service unavailable", status_code=503),
        ],
        "openrouter": [
            FakeProviderError("gateway timeout", status_code=504),
            fake_success(
                "Категория: technical_issue\n"
                "Приоритет: medium\n"
                "Черновик ответа: запросите версию продукта, текст ошибки, лог расчета и шаги воспроизведения.\n"
                "Эскалация: нет, пока не собраны базовые диагностические данные.",
                prompt_tokens=420,
                completion_tokens=86,
            ),
        ],
    }

    def factory(config: ProviderConfig) -> FakeOpenAIClient:
        return FakeOpenAIClient(scripted_outcomes[config.name])

    return RobustLLMClient(
        providers=providers,
        client_factory=factory,
        max_retries=5,
        max_jitter_sec=0.50,
        random_fn=lambda: 0.20,
        sleep_fn=lambda _delay: None,
        rate_limit_per_minute=0,
    )


def build_circuit_demo_client() -> RobustLLMClient:
    provider = ProviderConfig(
        name="openai",
        api_key="sk-demo-openai",
        base_url="https://api.openai.com/v1",
        model="env-openai-model",
    )
    outcomes = [
        FakeProviderError("server error A", status_code=500),
        FakeProviderError("server error B", status_code=500),
        FakeProviderError("server error C", status_code=500),
    ]

    return RobustLLMClient(
        providers=[provider],
        client_factory=lambda _config: FakeOpenAIClient(outcomes),
        max_retries=0,
        sleep_fn=lambda _delay: None,
        circuit_failure_threshold=3,
        circuit_cooldown_sec=60,
    )


def print_result(client: RobustLLMClient) -> None:
    result = client.chat(DEMO_MESSAGES, temperature=0.2, max_tokens=300)

    print("Retry + fallback demo")
    print("=" * 80)
    print(f"Provider used: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Unavailable: {result.unavailable}")
    print(f"Prompt tokens: {result.usage.prompt_tokens}")
    print(f"Completion tokens: {result.usage.completion_tokens}")
    print(f"Cost USD: {result.cost_usd:.8f}")
    print()
    print("Answer")
    print("-" * 80)
    print(result.answer)
    print()
    print("Retry log")
    print("-" * 80)
    print(client.formatted_log())
    print()
    print("Usage summary")
    print("-" * 80)
    for provider, usage in client.usage_summary().items():
        print(
            f"{provider}: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}, cost_usd={usage.cost_usd:.8f}"
        )


def print_circuit_breaker_demo() -> None:
    client = build_circuit_demo_client()
    for _ in range(4):
        client.chat(DEMO_MESSAGES, temperature=0.2, max_tokens=120)

    print()
    print("Circuit breaker demo")
    print("=" * 80)
    print(client.formatted_log())


def run_fake_demo(include_circuit_demo: bool) -> int:
    print_result(build_demo_client())
    if include_circuit_demo:
        print_circuit_breaker_demo()
    return 0


def run_real_demo() -> int:
    try:
        client = RobustLLMClient.from_env(DEFAULT_ENV_PATH)
        print_result(client)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 2.3 retry/fallback demo.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real OpenRouter/OpenAI/Ollama provider chain from .env instead of the offline fake demo.",
    )
    parser.add_argument(
        "--no-circuit-demo",
        action="store_true",
        help="Skip the offline circuit breaker demonstration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.real:
        return run_real_demo()
    return run_fake_demo(include_circuit_demo=not args.no_circuit_demo)


if __name__ == "__main__":
    raise SystemExit(main())
