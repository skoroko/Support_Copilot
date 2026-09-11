from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval.dataset import load_golden_dataset
from app.eval.metrics import aggregate_metrics, eval_row


class FakeMetric:
    def __init__(self, value: float | str) -> None:
        self.value = value
        self.calls: list[dict] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value, reason="checked")


class FailingMetric:
    async def ascore(self, **kwargs):
        raise RuntimeError("judge unavailable")


def test_golden_dataset_requires_thirty_complete_unique_cases(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "user_input": f"Вопрос {index}",
                    "reference": f"Ответ {index}",
                    "reference_contexts": [f"Контекст {index}"],
                }
                for index in range(30)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_golden_dataset(path)

    assert len(cases) == 30
    assert cases[0].reference_contexts == ("Контекст 0",)


def test_golden_dataset_rejects_empty_reference_contexts(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "user_input": f"Вопрос {index}",
                    "reference": f"Ответ {index}",
                    "reference_contexts": [] if index == 2 else ["Контекст"],
                }
                for index in range(30)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference_contexts"):
        load_golden_dataset(path)


@pytest.mark.asyncio
async def test_eval_row_uses_metric_specific_named_arguments() -> None:
    metrics = {
        "faithfulness": FakeMetric(0.8),
        "answer_relevancy": FakeMetric(0.9),
        "context_precision": FakeMetric(0.7),
        "context_recall": FakeMetric(0.6),
        "has_citation": FakeMetric("yes"),
    }

    row = await eval_row(
        metrics,
        user_input="Вопрос",
        response="Ответ [1]",
        retrieved_contexts=["Полный контекст"],
        reference="Эталон",
    )

    assert row["has_citation"] == 1.0
    assert set(metrics["faithfulness"].calls[0]) == {
        "user_input",
        "response",
        "retrieved_contexts",
    }
    assert set(metrics["context_recall"].calls[0]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }
    assert set(metrics["context_precision"].calls[0]) == {
        "user_input",
        "reference",
        "retrieved_contexts",
    }


def test_aggregate_metrics_ignores_missing_cells() -> None:
    result = aggregate_metrics(
        [
            {"faithfulness": 0.5, "has_citation": 1.0, "latency_ms": 10.0},
            {"faithfulness": 1.0, "has_citation": 0.0, "latency_ms": 20.0},
        ]
    )

    assert result["faithfulness"] == 0.75
    assert result["has_citation"] == 0.5
    assert result["latency_ms"] == 15.0
    assert result["answer_relevancy"] == 0.0


@pytest.mark.asyncio
async def test_eval_row_preserves_other_scores_when_one_metric_fails() -> None:
    metrics = {
        "faithfulness": FakeMetric(0.8),
        "answer_relevancy": FailingMetric(),
        "context_precision": FakeMetric(0.7),
        "context_recall": FakeMetric(0.6),
        "has_citation": FakeMetric("yes"),
    }

    row = await eval_row(
        metrics,
        user_input="Вопрос",
        response="Ответ [1]",
        retrieved_contexts=["Контекст"],
        reference="Эталон",
    )

    assert row["faithfulness"] == 0.8
    assert row["answer_relevancy"] is None
    assert row["answer_relevancy_error"] == "RuntimeError: judge unavailable"
