from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    relevant_doc_ids: tuple[str, ...]


def load_retrieval_dataset(path: Path, *, minimum_cases: int = 20) -> list[RetrievalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < minimum_cases:
        raise ValueError(f"retrieval dataset must contain at least {minimum_cases} cases")

    cases: list[RetrievalCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"dataset item {index} must be an object")
        question = item.get("question")
        doc_ids = item.get("relevant_doc_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"dataset item {index} has an invalid question")
        if not isinstance(doc_ids, list) or not 1 <= len(doc_ids) <= 3:
            raise ValueError(f"dataset item {index} must contain 1-3 relevant_doc_ids")
        normalized = tuple(str(doc_id).strip() for doc_id in doc_ids)
        if any(not doc_id for doc_id in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError(f"dataset item {index} contains invalid relevant_doc_ids")
        cases.append(RetrievalCase(question=question.strip(), relevant_doc_ids=normalized))
    return cases


def retrieval_metrics(
    cases: Sequence[RetrievalCase],
    rankings: Sequence[Sequence[str]],
) -> dict[str, float]:
    if not cases or len(cases) != len(rankings):
        raise ValueError("cases and rankings must be non-empty and have equal length")

    hit_rate = 0.0
    reciprocal_rank = 0.0
    recall = 0.0
    for case, ranking in zip(cases, rankings, strict=True):
        relevant = set(case.relevant_doc_ids)
        hit_rate += float(any(doc_id in relevant for doc_id in ranking[:5]))
        reciprocal_rank += next(
            (1.0 / rank for rank, doc_id in enumerate(ranking[:10], start=1) if doc_id in relevant),
            0.0,
        )
        recall += len(relevant.intersection(ranking[:10])) / len(relevant)

    count = len(cases)
    return {
        "hit_rate@5": hit_rate / count,
        "mrr@10": reciprocal_rank / count,
        "recall@10": recall / count,
    }


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    retrieve: Callable[[str, int], Sequence[str]],
) -> dict[str, float]:
    """Run one retriever over a golden dataset and return all required metrics."""

    rankings = [list(retrieve(case.question, 10)) for case in cases]
    return retrieval_metrics(cases, rankings)
