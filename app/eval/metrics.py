from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "has_citation",
)


class CitationVerdict(BaseModel):
    has_citation: bool
    reason: str


@dataclass(frozen=True)
class EvaluationModels:
    llm: Any
    embeddings: Any
    client: Any


def build_judge(model: str, embedding_model: str, *, api_key: str | None = None) -> EvaluationModels:
    """Build native RAGAS 0.4 judge and embedding adapters lazily."""

    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(model, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
    return EvaluationModels(llm=llm, embeddings=embeddings, client=client)


def make_has_citation(llm: Any) -> Any:
    """Create the required judge-backed binary citation metric."""

    from ragas.metrics import MetricResult, discrete_metric

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(*, response: str) -> Any:
        prompt = (
            "Содержит ли ответ ссылку на источник: маркер вида '[1]' или '[doc_id]', "
            "имя файла, либо фразу 'согласно ...'? Оцени только наличие ссылки, "
            "не качество ответа.\n\nОтвет:\n"
            f"{response}"
        )
        verdict = await llm.agenerate(prompt, response_model=CitationVerdict)
        return MetricResult(
            value="yes" if verdict.has_citation else "no",
            reason=verdict.reason,
        )

    return has_citation


def build_metrics(models: EvaluationModels) -> dict[str, Any]:
    """Create the four RAGAS collection metrics and custom citation metric."""

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=models.llm),
        "answer_relevancy": AnswerRelevancy(llm=models.llm, embeddings=models.embeddings),
        "context_precision": ContextPrecision(llm=models.llm),
        "context_recall": ContextRecall(llm=models.llm),
        "has_citation": make_has_citation(models.llm),
    }


async def eval_row(
    metrics: Mapping[str, Any],
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict[str, Any]:
    """Evaluate one row using only named RAGAS 0.4 ``ascore`` arguments."""

    calls = {
        "faithfulness": {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        },
        "answer_relevancy": {
            "user_input": user_input,
            "response": response,
        },
        "context_precision": {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
        },
        "context_recall": {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
        },
        "has_citation": {"response": response},
    }
    row: dict[str, Any] = {}
    for name in METRIC_NAMES:
        try:
            result = await metrics[name].ascore(**calls[name])
            value = getattr(result, "value", result)
            if value is None:
                raise ValueError(getattr(result, "reason", None) or "metric returned no value")
            if name == "has_citation":
                value = 1.0 if str(value).casefold() == "yes" else 0.0
            row[name] = _finite_float(value)
            reason = getattr(result, "reason", None)
            if reason:
                row[f"{name}_reason"] = str(reason)
        except Exception as exc:  # one judge failure must not discard the full run
            row[name] = None
            row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return row


def aggregate_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """Mean metric values, ignoring failed or missing cells explicitly."""

    aggregates: dict[str, float] = {}
    for name in (*METRIC_NAMES, "latency_ms"):
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))
        ]
        aggregates[name] = sum(values) / len(values) if values else 0.0
    return aggregates


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric returned a non-finite value")
    return result
