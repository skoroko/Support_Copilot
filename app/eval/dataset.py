from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenCase:
    user_input: str
    reference: str
    reference_contexts: tuple[str, ...]


def load_golden_dataset(path: Path, *, minimum_cases: int = 30) -> list[GoldenCase]:
    """Load the manually curated RAGAS dataset with a strict stable schema."""

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) < minimum_cases:
        raise ValueError(f"golden dataset must contain at least {minimum_cases} cases")

    cases: list[GoldenCase] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"golden item {index} must be an object")
        user_input = _required_text(item.get("user_input"), "user_input", index)
        reference = _required_text(item.get("reference"), "reference", index)
        contexts = item.get("reference_contexts")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"golden item {index} must contain reference_contexts")
        normalized_contexts = tuple(
            _required_text(context, "reference_contexts", index) for context in contexts
        )
        normalized_key = " ".join(user_input.casefold().split())
        if normalized_key in seen_questions:
            raise ValueError(f"golden item {index} duplicates user_input")
        seen_questions.add(normalized_key)
        cases.append(GoldenCase(user_input, reference, normalized_contexts))
    return cases


def _required_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"golden item {index} has an invalid {field}")
    return value.strip()

