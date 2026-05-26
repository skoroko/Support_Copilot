from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def monthly_cost(
    requests_per_day: float,
    input_tokens: float,
    output_tokens: float,
    input_price: float,
    cached_input_price: float,
    output_price: float,
    cache_hit_rate: float,
) -> dict[str, float]:
    monthly_requests = requests_per_day * 30
    monthly_input = monthly_requests * input_tokens
    monthly_output = monthly_requests * output_tokens
    no_cache = (monthly_input / 1_000_000 * input_price) + (monthly_output / 1_000_000 * output_price)

    cached_input = monthly_input * cache_hit_rate
    uncached_input = monthly_input - cached_input
    with_cache = (
        (uncached_input / 1_000_000 * input_price)
        + (cached_input / 1_000_000 * cached_input_price)
        + (monthly_output / 1_000_000 * output_price)
    )
    return {
        "monthly_requests": round(monthly_requests, 0),
        "monthly_input_tokens": round(monthly_input, 0),
        "monthly_output_tokens": round(monthly_output, 0),
        "cloud_cost_no_cache_usd": round(no_cache, 2),
        "cloud_cost_with_cache_usd": round(with_cache, 2),
        "cache_savings_usd": round(no_cache - with_cache, 2),
    }


def main() -> int:
    load_env(PROJECT_ROOT / ".env")
    report = {
        "cloud_model": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        "local_model_primary": "qwen3:1.7b",
        "assumptions": {
            "requests_per_day": env_float("REQUESTS_PER_DAY", 300),
            "avg_input_tokens": env_float("AVG_INPUT_TOKENS", 900),
            "avg_output_tokens": env_float("AVG_OUTPUT_TOKENS", 350),
            "cache_hit_rate": env_float("CACHE_HIT_RATE", 0.35),
            "openai_input_price_per_1m": env_float("OPENAI_INPUT_PRICE_PER_1M", 0.75),
            "openai_cached_input_price_per_1m": env_float("OPENAI_CACHED_INPUT_PRICE_PER_1M", 0.075),
            "openai_output_price_per_1m": env_float("OPENAI_OUTPUT_PRICE_PER_1M", 4.50),
        },
    }
    report["monthly_cost"] = monthly_cost(
        requests_per_day=report["assumptions"]["requests_per_day"],
        input_tokens=report["assumptions"]["avg_input_tokens"],
        output_tokens=report["assumptions"]["avg_output_tokens"],
        input_price=report["assumptions"]["openai_input_price_per_1m"],
        cached_input_price=report["assumptions"]["openai_cached_input_price_per_1m"],
        output_price=report["assumptions"]["openai_output_price_per_1m"],
        cache_hit_rate=report["assumptions"]["cache_hit_rate"],
    )
    output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cost_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
