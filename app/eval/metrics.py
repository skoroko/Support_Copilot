from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from pydantic import create_model


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "has_citation",
)


async def score_context_precision_batch(
    llm: Any,
    *,
    user_input: str,
    reference: str,
    retrieved_contexts: list[str],
    calculate: Callable[[list[int]], float],
) -> float:
    """Judge ordered contexts in one structured call and apply RAGAS AP."""

    if not user_input:
        raise ValueError("user_input cannot be empty")
    if not reference:
        raise ValueError("reference cannot be empty")
    if not retrieved_contexts:
        raise ValueError("retrieved_contexts cannot be empty")

    response_model = create_model(
        "ContextPrecisionBatchVerdict",
        **{
            f"verdict_{index}": (Literal[0, 1], ...)
            for index in range(1, len(retrieved_contexts) + 1)
        },
    )
    prompt = (
        "Given a question, reference answer and an ordered list of contexts, "
        "verify each context independently. Return verdict 1 when that context "
        "is useful for arriving at the reference answer, otherwise 0. Return "
        "every required verdict_N field and no explanations.\n\n"
        + json.dumps(
            {
                "question": user_input,
                "reference_answer": reference,
                "contexts": {
                    f"context_{index}": context
                    for index, context in enumerate(retrieved_contexts, start=1)
                },
            },
            ensure_ascii=False,
        )
    )
    result = await llm.agenerate(
        prompt,
        response_model=response_model,
    )
    verdicts = [
        int(getattr(result, f"verdict_{index}"))
        for index in range(1, len(retrieved_contexts) + 1)
    ]
    return float(calculate(verdicts))


@dataclass(frozen=True)
class EvaluationModels:
    llm: Any
    embeddings: Any
    client: Any


def build_judge(model: str, embedding_model: str, *, api_key: str | None = None) -> EvaluationModels:
    """Build native RAGAS 0.4 judge and embedding adapters lazily."""

    import httpx
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(600.0, connect=30.0),
        max_retries=2,
    )
    llm = llm_factory(model, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
    return EvaluationModels(llm=llm, embeddings=embeddings, client=client)


def make_has_citation(llm: Any) -> Any:
    """Create the required deterministic binary citation metric."""

    from ragas.metrics import MetricResult, discrete_metric

    del llm  # The common metric factory signature is kept for build_metrics().

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(*, response: str) -> Any:
        present = bool(
            re.search(r"\[(?:\d+|[A-Za-z][\w.-]*)\]", response)
            or re.search(r"\b[^\n]{1,160}\.(?:pdf|docx?|chm|html?)\b", response, re.I)
            or re.search(r"\bсогласно\s+\S+", response, re.I)
        )
        return MetricResult(
            value="yes" if present else "no",
            reason="citation marker present" if present else "citation marker absent",
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
    from ragas.metrics import MetricResult

    class BatchedContextPrecision(ContextPrecision):
        """Keep RAGAS average precision while judging all contexts in one call."""

        async def ascore(
            self,
            user_input: str,
            reference: str,
            retrieved_contexts: list[str],
        ) -> Any:
            score = await score_context_precision_batch(
                self.llm,
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
                calculate=self._calculate_average_precision,
            )
            return MetricResult(value=score)

    answer_relevancy_strictness = int(
        os.environ.get("RAG_EVAL_ANSWER_RELEVANCY_STRICTNESS", "1")
    )
    if answer_relevancy_strictness < 1:
        raise ValueError("RAG_EVAL_ANSWER_RELEVANCY_STRICTNESS must be positive")

    return {
        "faithfulness": Faithfulness(llm=models.llm),
        "answer_relevancy": AnswerRelevancy(
            llm=models.llm,
            embeddings=models.embeddings,
            strictness=answer_relevancy_strictness,
        ),
        "context_precision": BatchedContextPrecision(llm=models.llm),
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
    cached_values: Mapping[str, float] | None = None,
    metric_semaphores: Mapping[str, asyncio.Semaphore] | None = None,
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
        if cached_values is not None and name in cached_values:
            row[name] = _finite_float(cached_values[name])
            row[f"{name}_cached"] = True
            continue
        try:
            timeout_seconds = float(os.environ.get("RAG_EVAL_METRIC_TIMEOUT_SECONDS", "180"))
            metric_semaphore = (metric_semaphores or {}).get(name)
            if metric_semaphore is None:
                result = await asyncio.wait_for(
                    metrics[name].ascore(**calls[name]),
                    timeout=timeout_seconds,
                )
            else:
                async with metric_semaphore:
                    result = await asyncio.wait_for(
                        metrics[name].ascore(**calls[name]),
                        timeout=timeout_seconds,
                    )
            value = getattr(result, "value", result)
            if value is None:
                raise ValueError(getattr(result, "reason", None) or "metric returned no value")
            if name == "has_citation":
                value = 1.0 if str(value).casefold() == "yes" else 0.0
            row[name] = _finite_float(value)
            row[f"{name}_cached"] = False
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
