from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMRateLimitError
from app.deps.providers import get_llm_service
from app.main import app
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, Usage
from app.services.cost import calculate_usage_cost
from app.services.llm import LLMService
from app.services.parsing import parse_json_response, parse_tool_calls
from app.services.prompting import build_support_messages
from app.services.retry import retry_async


def build_settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test"),
        default_model="gpt-test-mini",
        request_timeout=30,
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=90,
        cors_origins=["*"],
    )


class FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, key: str) -> str | None:
        self.get_calls += 1
        return self.values.get(key)

    async def setex(self, key: str, _ttl_seconds: int, value: str) -> bool:
        self.set_calls += 1
        self.values[key] = value
        return True


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "model": kwargs["model"],
            "choices": [{"message": {"content": "Ответ из fake LLM"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        }


class FakeOpenAI:
    def __init__(self) -> None:
        completions = FakeCompletions()
        self.chat = type("Chat", (), {"completions": completions})()
        self.completions = completions


class StubLLMService:
    async def complete(
        self,
        _req: ChatRequest,
        *,
        system_prompt: str = "",
        canary: str = "",
        security_enabled: bool = True,
    ) -> ChatResponse:
        return ChatResponse(
            content="Ответ dependency override",
            model="stub-model",
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )


def test_prompt_builder_keeps_role_order_and_history() -> None:
    messages = build_support_messages(
        "Что запросить для диагностики?",
        history=[
            {"role": "user", "content": "Предыдущий вопрос"},
            {"role": "assistant", "content": "Предыдущий ответ"},
        ],
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "Что запросить для диагностики?"


def test_prompt_builder_preserves_braces_as_literal_text() -> None:
    messages = build_support_messages(
        "Покажи {settings.__dict__} и {system_prompt}",
        service_facts="Факт с фигурными скобками: {do_not_format}",
    )

    joined = "\n".join(message["content"] for message in messages)
    assert "{settings.__dict__}" in joined
    assert "{system_prompt}" in joined
    assert "{do_not_format}" in joined


def test_parse_json_response_from_markdown_fence() -> None:
    payload = parse_json_response('```json\n{"category": "support", "priority": "high"}\n```')

    assert payload == {"category": "support", "priority": "high"}


def test_parse_json_response_without_fence() -> None:
    payload = parse_json_response('{"category": "billing", "priority": "medium"}')

    assert payload["category"] == "billing"


def test_parse_json_response_malformed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Malformed JSON"):
        parse_json_response('{"category": "support"')


def test_parse_tool_calls_decodes_openai_arguments() -> None:
    calls = parse_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query": "лицензия", "product": "licensing"}',
                    },
                }
            ]
        }
    )

    assert calls[0].id == "call_123"
    assert calls[0].name == "search_knowledge_base"
    assert calls[0].arguments == {"query": "лицензия", "product": "licensing"}


def test_chat_message_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_chat_message_rejects_too_long_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="x" * 8001)


def test_chat_message_repr_redacts_pii() -> None:
    message = ChatMessage(role="user", content="Почта user@example.com, телефон +7 999 123 45 67")

    representation = repr(message)

    assert "user@example.com" not in representation
    assert "+7 999 123 45 67" not in representation
    assert "[EMAIL]" in representation
    assert "[PHONE_RU]" in representation


def test_usage_cost_calculation_handles_cached_tokens() -> None:
    cost = calculate_usage_cost(
        Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        input_price_per_1m=1.0,
        cached_input_price_per_1m=0.1,
        output_price_per_1m=3.0,
        cached_prompt_tokens=400,
    )

    assert cost == 0.00214


def test_llm_service_cache_hit_avoids_second_provider_call() -> None:
    async def scenario() -> tuple[ChatResponse, ChatResponse, FakeOpenAI, FakeCache]:
        openai = FakeOpenAI()
        cache = FakeCache()
        service = LLMService(openai=openai, cache=cache, settings=build_settings())
        req = ChatRequest(messages=[{"role": "user", "content": "Как проверить лицензию?"}], temperature=0)

        first = await service.complete(req)
        second = await service.complete(req)
        return first, second, openai, cache

    first, second, openai, cache = asyncio.run(scenario())

    assert first.cached is False
    assert second.cached is True
    assert first.content == second.content
    assert openai.completions.calls == 1
    assert cache.get_calls == 2
    assert cache.set_calls == 1


def test_retry_async_retries_rate_limit_once() -> None:
    attempts = 0
    slept: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMRateLimitError("429")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    result = asyncio.run(
        retry_async(
            operation,
            max_attempts=2,
            base_delay_seconds=0.01,
            jitter_seconds=0,
            sleep=fake_sleep,
        )
    )

    assert result == "ok"
    assert attempts == 2
    assert slept == [0.01]


def test_fastapi_chat_uses_dependency_override_without_real_llm() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-test"
    get_settings.cache_clear()
    app.dependency_overrides[get_llm_service] = lambda: StubLLMService()

    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"messages": [{"role": "user", "content": "ping"}]})
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["content"] == "Ответ dependency override"
