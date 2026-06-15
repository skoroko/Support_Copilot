from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_cache import LLMCache, RedisLLMCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MESSAGES = [
    {
        "role": "system",
        "content": "Ты Support Copilot. Отвечай по-русски и не выдумывай факты MIDAS.",
    },
    {
        "role": "user",
        "content": "После обновления расчет модели завершается ошибкой. Что запросить у пользователя?",
    },
]

CACHED_RESPONSE = {
    "answer": (
        "Запросите версию продукта, текст ошибки, лог расчета и шаги воспроизведения. "
        "До проверки инженером не обещайте исправление."
    ),
    "category": "technical_issue",
    "priority": "medium",
}


class ManualClock:
    def __init__(self, current: float = 1_000.0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def run_memory_demo() -> None:
    clock = ManualClock()
    cache = LLMCache(ttl_seconds=5, clock=clock)
    model = "support-demo-model"
    temperature = 0.2

    print("In-memory LLMCache demo")
    print("=" * 80)

    first = cache.get(model, MESSAGES, temperature)
    print(f"1. First read before set: {first}")
    print(f"   stats: {pretty(cache.stats())}")

    key = cache.set(model, MESSAGES, temperature, CACHED_RESPONSE)
    print(f"2. Stored key: {key}")

    second = cache.get(model, MESSAGES, temperature)
    print(f"3. Second read after set: {pretty(second)}")
    print(f"   stats: {pretty(cache.stats())}")

    different_temperature = cache.get(model, MESSAGES, 0.7)
    print(f"4. Read with different temperature: {different_temperature}")
    print(f"   stats: {pretty(cache.stats())}")

    different_prompt_version = cache.get(
        model,
        MESSAGES,
        temperature,
        prompt_version="support-copilot-prompt-v2",
    )
    print(f"5. Read with different prompt version: {different_prompt_version}")
    print(f"   stats: {pretty(cache.stats())}")

    clock.advance(6)
    expired = cache.get(model, MESSAGES, temperature)
    print(f"6. Read after TTL expiry: {expired}")
    print(f"   stats: {pretty(cache.stats())}")


def run_redis_demo(redis_url: str) -> int:
    cache = RedisLLMCache(redis_url=redis_url, ttl_seconds=60)
    model = "support-demo-model"
    temperature = 0.2
    print("RedisLLMCache demo")
    print("=" * 80)
    print(f"Before set: {cache.get(model, MESSAGES, temperature)}")
    key = cache.set(model, MESSAGES, temperature, CACHED_RESPONSE)
    print(f"Stored key: {key}")
    print(f"After set: {pretty(cache.get(model, MESSAGES, temperature))}")
    print(f"Stats: {pretty(cache.stats())}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 2.4 LLM cache demo.")
    parser.add_argument(
        "--redis-url",
        help="Run RedisLLMCache demo with this Redis URL. Omit for in-memory demo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.redis_url:
        return run_redis_demo(args.redis_url)
    run_memory_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
