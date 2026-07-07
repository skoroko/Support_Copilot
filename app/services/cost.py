from __future__ import annotations

from app.schemas.chat import Usage


def calculate_usage_cost(
    usage: Usage,
    *,
    input_price_per_1m: float,
    output_price_per_1m: float,
    cached_input_price_per_1m: float | None = None,
    cached_prompt_tokens: int = 0,
) -> float:
    cached_tokens = max(0, min(cached_prompt_tokens, usage.prompt_tokens))
    regular_prompt_tokens = max(0, usage.prompt_tokens - cached_tokens)
    cached_price = input_price_per_1m if cached_input_price_per_1m is None else cached_input_price_per_1m

    cost = (
        regular_prompt_tokens * input_price_per_1m
        + cached_tokens * cached_price
        + usage.completion_tokens * output_price_per_1m
    ) / 1_000_000
    return round(cost, 8)
