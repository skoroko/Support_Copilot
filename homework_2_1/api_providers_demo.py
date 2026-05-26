"""Homework 2.1: OpenAI SDK, temperature experiment, streaming and providers."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from support_copilot.config import env_float, env_int, load_env_file


SYSTEM_MESSAGE = (
    "Ты Support Copilot, внутренний помощник техподдержки MIDAS. "
    "Отвечай по-русски, не выдумывай факты и явно проси эскалацию, если данных мало."
)
USER_MESSAGE = (
    "Пользователь пишет: после обновления появляется ошибка 503. "
    "Подготовь короткий черновик ответа инженеру поддержки."
)

# Наблюдения по temperature:
# - temperature=0 обычно дает самый стабильный и повторяемый ответ.
# - temperature=0.7 добавляет вариативность формулировок без сильной потери контроля.
# - temperature=1.5 может быть креативнее, но для техподдержки повышает риск лишних деталей.
OBSERVATIONS = {
    0.0: "стабильный, сухой и хорошо контролируемый ответ",
    0.7: "более естественные формулировки, умеренная вариативность",
    1.5: "больше разнообразия, но выше риск лишних предположений",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo for homework 2.1")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--provider", default=None, help="openai, openrouter or value from LLM_PROVIDER")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)

    provider_name = (args.provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    max_tokens = args.max_tokens or env_int("DEMO_MAX_TOKENS", 260)

    lines: list[str] = []
    _emit(lines, f"provider={provider_name}")
    _emit(lines, "Base request: system + user messages")

    for temperature in (0.0, 0.7, 1.5):
        started = time.perf_counter()
        client, model = _client_for_provider(provider_name)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE},
            ],
            temperature=temperature,
            **{_completion_tokens_parameter(model): max_tokens},
        )
        answer = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - started
        _emit(lines, f"\n--- temperature={temperature} elapsed={elapsed:.3f}s")
        _emit(lines, f"Observation: {OBSERVATIONS[temperature]}")
        _emit(lines, answer)

    _emit(lines, "\n--- streaming demo")
    client, model = _client_for_provider(provider_name)
    started = time.perf_counter()
    first_token_at: float | None = None
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ],
        temperature=0.2,
        stream=True,
        **{_completion_tokens_parameter(model): max_tokens},
    )
    for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta and first_token_at is None:
            first_token_at = time.perf_counter()
        if delta:
            _emit(lines, f"chunk={delta}")
    total = time.perf_counter() - started
    ttft = (first_token_at - started) if first_token_at else total
    _emit(lines, f"TTFT={ttft:.3f}s total={total:.3f}s")

    if args.log_file:
        Path(args.log_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _client_for_provider(provider_name: str) -> tuple[OpenAI, str]:
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "")
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "")

    if not api_key or not model:
        raise RuntimeError(
            f"Provider {provider_name} is not configured: set API key, base_url and model."
        )
    return OpenAI(api_key=api_key, base_url=base_url), model


def _emit(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def _completion_tokens_parameter(model: str) -> str:
    if model.startswith("gpt-5") or model.startswith("o"):
        return "max_completion_tokens"
    return "max_tokens"


if __name__ == "__main__":
    raise SystemExit(main())
