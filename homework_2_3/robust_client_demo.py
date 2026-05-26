"""Homework 2.3: live RobustLLMClient demo with retry, fallback and usage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.config import env_float, env_int, load_env_file
from support_copilot.infrastructure import ProviderConfig, RobustLLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live RobustLLMClient demo")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--log-file", default="homework_2_3/retry_fallback_log.txt")
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)

    log_path = Path(args.log_file)
    if log_path.exists():
        log_path.unlink()

    providers = _providers_from_env()
    client = RobustLLMClient(providers, log_file=log_path)
    result = client.chat(
        [
            {"role": "system", "content": "Ты помощник техподдержки MIDAS."},
            {
                "role": "user",
                "content": (
                    "После обновления появляется ошибка 503. "
                    "Дай краткий план диагностики и условия эскалации."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=args.max_tokens or env_int("DEMO_MAX_TOKENS", 350),
    )

    print(result.content)
    print(f"provider={result.provider} fallback_used={result.fallback_used}")
    print(f"usage={result.usage.as_dict()}")
    print(f"log={log_path}")
    return 0


def _providers_from_env() -> list[ProviderConfig]:
    return [
        ProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", ""),
            input_price_per_1m=env_float("OPENAI_INPUT_PRICE_PER_1M", 0.0),
            output_price_per_1m=env_float("OPENAI_OUTPUT_PRICE_PER_1M", 0.0),
            timeout_sec=env_float("LLM_TIMEOUT_SEC", 60.0),
        ),
        ProviderConfig(
            name="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL", ""),
            input_price_per_1m=env_float("OPENROUTER_INPUT_PRICE_PER_1M", 0.0),
            output_price_per_1m=env_float("OPENROUTER_OUTPUT_PRICE_PER_1M", 0.0),
            timeout_sec=env_float("LLM_TIMEOUT_SEC", 60.0),
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
