from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    display_name: str
    input_price_per_1m: float
    cached_input_price_per_1m: float
    output_price_per_1m: float
    source: str


# Checked against https://openai.com/api/pricing/ on 2026-06-16.
OPENAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.5",
        display_name="GPT-5.5",
        input_price_per_1m=5.00,
        cached_input_price_per_1m=0.50,
        output_price_per_1m=30.00,
        source="https://openai.com/api/pricing/",
    ),
    ModelInfo(
        id="gpt-5.4",
        display_name="GPT-5.4",
        input_price_per_1m=2.50,
        cached_input_price_per_1m=0.25,
        output_price_per_1m=15.00,
        source="https://openai.com/api/pricing/",
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 mini",
        input_price_per_1m=0.75,
        cached_input_price_per_1m=0.075,
        output_price_per_1m=4.50,
        source="https://openai.com/api/pricing/",
    ),
]
