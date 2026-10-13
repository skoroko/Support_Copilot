from __future__ import annotations

from types import SimpleNamespace

from support_copilot.cli import SupportAssistant
from support_copilot.infrastructure.cache import LLMCache


class FakeClient:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        self.calls.append(messages)
        product = next(
            message["content"]
            for message in messages
            if message["role"] == "user" and message["content"].startswith("Product ")
        )
        return SimpleNamespace(
            answer=f"Instructions for {product}",
            provider="fake",
            model="fake-model",
            usage=None,
            unavailable=False,
        )


class FakeLogger:
    def write(self, _line: str) -> None:
        pass


def make_assistant(client: FakeClient, cache: LLMCache, product: str, *, max_tokens: int = 100) -> SupportAssistant:
    assistant = SupportAssistant(
        client,
        cache,
        mode_name="test",
        real=False,
        stream=False,
        vision_provider=None,
        model_cache_id="fake-model",
        temperature=0,
        max_tokens=max_tokens,
        history_limit=10,
        logger=FakeLogger(),
    )
    assistant.history = [
        {"role": "user", "content": f"Product {product}"},
        {"role": "assistant", "content": "Context noted"},
    ]
    return assistant


def test_cli_cache_uses_the_same_history_as_the_provider() -> None:
    cache = LLMCache()
    client = FakeClient()

    answer_a = make_assistant(client, cache, "A").answer("How do I install it?")
    answer_b = make_assistant(client, cache, "B").answer("How do I install it?")
    repeated_a = make_assistant(client, cache, "A").answer("How do I install it?")

    assert "Product A" in answer_a
    assert "Product B" in answer_b
    assert "Product A" in repeated_a
    assert "Источник: cache" in repeated_a
    assert len(client.calls) == 2
    assert client.calls[0][1]["content"] == "Product A"
    assert client.calls[1][1]["content"] == "Product B"


def test_cli_cache_separates_max_token_settings() -> None:
    cache = LLMCache()
    client = FakeClient()

    make_assistant(client, cache, "A", max_tokens=100).answer("How do I install it?")
    make_assistant(client, cache, "A", max_tokens=25).answer("How do I install it?")

    assert len(client.calls) == 2
