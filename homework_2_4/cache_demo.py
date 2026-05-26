"""Homework 2.4: LLMCache demo with TTL and stats."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.infrastructure import LLMCache


MESSAGES = [
    {"role": "system", "content": "Ты помощник техподдержки MIDAS."},
    {"role": "user", "content": "Как установить лицензию?"},
]
MODEL = "gpt-5.4-mini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLMCache demo")
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lines: list[str] = []
    cache = LLMCache(ttl_sec=2)

    _emit(lines, "First get: expected miss")
    _emit(lines, f"value={cache.get(MODEL, MESSAGES, 0.2)}")
    cache.set(MODEL, MESSAGES, 0.2, "cached answer")

    _emit(lines, "Second get: expected hit")
    _emit(lines, f"value={cache.get(MODEL, MESSAGES, 0.2)}")

    _emit(lines, "Different temperature: expected miss")
    _emit(lines, f"value={cache.get(MODEL, MESSAGES, 0.7)}")

    _emit(lines, f"stats={cache.stats()}")

    time.sleep(2.1)
    _emit(lines, "After TTL: expected miss")
    _emit(lines, f"value={cache.get(MODEL, MESSAGES, 0.2)}")
    _emit(lines, f"stats={cache.stats()}")

    if args.log_file:
        Path(args.log_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _emit(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


if __name__ == "__main__":
    raise SystemExit(main())
