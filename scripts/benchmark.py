from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.infrastructure.async_llm import AsyncLLMClient, build_demo_async_client  # noqa: E402
from support_copilot.infrastructure.llm import RobustLLMClient  # noqa: E402


PROMPTS = [
    (
        "Объясни одним абзацем концепцию #{index}: как инженеру поддержки MIDAS "
        "быстро понять суть обращения, проверить факты и подготовить безопасный черновик ответа."
    ).format(index=index)
    for index in range(1, 21)
]


class DemoSyncLLMClient:
    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = delay_sec

    def complete(self, prompt: str) -> str:
        time.sleep(self.delay_sec)
        return f"Ответ на {len(prompt)} символов"


def run_sync_demo(prompts: list[str], delay_sec: float) -> tuple[float, int]:
    client = DemoSyncLLMClient(delay_sec)
    start = time.perf_counter()
    for prompt in prompts:
        client.complete(prompt)
    return time.perf_counter() - start, 0


def run_sync_real(prompts: list[str], max_tokens: int) -> tuple[float, int]:
    client = RobustLLMClient.from_env()
    errors = 0
    start = time.perf_counter()
    for prompt in prompts:
        result = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
        if result.unavailable:
            errors += 1
    return time.perf_counter() - start, errors


async def run_async_demo(prompts: list[str], concurrency: int, delay_sec: float) -> tuple[float, int]:
    client = build_demo_async_client(concurrency=concurrency, delay_sec=delay_sec)
    start = time.perf_counter()
    results = await client.batch_chat(prompts, concurrency=concurrency)
    elapsed = time.perf_counter() - start
    errors = sum(1 for item in results if isinstance(item, Exception))
    return elapsed, errors


async def run_async_real(prompts: list[str], concurrency: int, max_tokens: int) -> tuple[float, int]:
    client = AsyncLLMClient.from_env(concurrency=concurrency)
    start = time.perf_counter()
    results = await client.batch_chat(prompts, concurrency=concurrency, max_tokens=max_tokens)
    elapsed = time.perf_counter() - start
    errors = sum(1 for item in results if isinstance(item, Exception))
    return elapsed, errors


async def run_benchmark(args: argparse.Namespace) -> str:
    prompts = PROMPTS[: args.requests]
    lines = [
        "# Async benchmark results",
        "",
        f"Mode: `{args.mode}`",
        f"Requests: `{len(prompts)}`",
        f"Demo delay: `{args.delay_sec:.3f}s`",
        "",
        "| Mode | Concurrency | Total seconds | Errors | Speedup vs sync |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    if args.mode == "real":
        sync_time, sync_errors = run_sync_real(prompts, args.max_tokens)
    else:
        sync_time, sync_errors = run_sync_demo(prompts, args.delay_sec)

    lines.append(f"| sync sequential | 1 | {sync_time:.3f} | {sync_errors} | 1.00x |")

    for concurrency in (1, 5, 10):
        if args.mode == "real":
            async_time, async_errors = await run_async_real(prompts, concurrency, args.max_tokens)
        else:
            async_time, async_errors = await run_async_demo(prompts, concurrency, args.delay_sec)
        speedup = sync_time / async_time if async_time > 0 else 0.0
        lines.append(
            f"| async batch_chat | {concurrency} | {async_time:.3f} | {async_errors} | {speedup:.2f}x |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- Demo mode uses `time.sleep` only in the old sync baseline and `asyncio.sleep` in the async fake provider.",
            "- Real mode uses provider settings from `.env`; rate limits can reduce async speedup.",
            "- `AsyncLLMClient` keeps concurrency in the instance semaphore `self._sem`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sync vs async Support Copilot LLM calls.")
    parser.add_argument("--mode", choices=("demo", "real"), default="demo")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--delay-sec", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--output", default="scripts/benchmark_results.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_benchmark(args))
    print(report)
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
