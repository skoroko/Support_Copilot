from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidencePredicate:
    """One acceptable chunk-level evidence target for a golden case."""

    doc_id: str
    contains_all: tuple[str, ...]
    page: int | None = None
    printed_page: int | None = None


@dataclass(frozen=True)
class RetrievalHistoryItem:
    role: str
    content: str


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    relevant_doc_ids: tuple[str, ...]
    history: tuple[RetrievalHistoryItem, ...] = ()
    relevant_evidence: tuple[EvidencePredicate, ...] = ()
    expected_answer_contains_all: tuple[str, ...] = ()


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
        history = _load_history(item.get("history", []), index)
        evidence = _load_evidence(item.get("relevant_evidence", []), normalized, index)
        answer_terms = _load_string_list(
            item.get("expected_answer_contains_all", []),
            field="expected_answer_contains_all",
            index=index,
        )
        cases.append(
            RetrievalCase(
                question=question.strip(),
                relevant_doc_ids=normalized,
                history=history,
                relevant_evidence=evidence,
                expected_answer_contains_all=answer_terms,
            )
        )
    return cases


def _load_string_list(value: Any, *, field: str, index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"dataset item {index} has an invalid {field}")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"dataset item {index} contains duplicate {field}")
    return normalized


def _load_history(value: Any, index: int) -> tuple[RetrievalHistoryItem, ...]:
    if not isinstance(value, list):
        raise ValueError(f"dataset item {index} has invalid history")
    history: list[RetrievalHistoryItem] = []
    for turn in value:
        if not isinstance(turn, dict):
            raise ValueError(f"dataset item {index} has invalid history")
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError(f"dataset item {index} has invalid history")
        history.append(RetrievalHistoryItem(role=role, content=content.strip()))
    return tuple(history)


def _load_evidence(
    value: Any,
    relevant_doc_ids: tuple[str, ...],
    index: int,
) -> tuple[EvidencePredicate, ...]:
    if not isinstance(value, list):
        raise ValueError(f"dataset item {index} has invalid relevant_evidence")
    evidence: list[EvidencePredicate] = []
    for predicate in value:
        if not isinstance(predicate, dict):
            raise ValueError(f"dataset item {index} has invalid relevant_evidence")
        doc_id = predicate.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in relevant_doc_ids:
            raise ValueError(f"dataset item {index} evidence references a non-relevant doc_id")
        contains_all = _load_string_list(
            predicate.get("contains_all", []),
            field="relevant_evidence.contains_all",
            index=index,
        )
        if not contains_all:
            raise ValueError(f"dataset item {index} evidence must contain text predicates")
        page = _optional_positive_int(predicate.get("page"), "page", index)
        printed_page = _optional_positive_int(predicate.get("printed_page"), "printed_page", index)
        evidence.append(EvidencePredicate(doc_id, contains_all, page, printed_page))
    return tuple(evidence)


def _optional_positive_int(value: Any, field: str, index: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"dataset item {index} has invalid evidence {field}")
    return value


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


def evidence_metrics(
    cases: Sequence[RetrievalCase],
    rankings: Sequence[Sequence[Mapping[str, Any] | Any]],
) -> dict[str, float]:
    """Score exact supporting chunks, including optional physical/printed pages.

    Cases without ``relevant_evidence`` are excluded from the denominator so
    legacy document-only datasets and reports remain backward compatible.
    """

    if not cases or len(cases) != len(rankings):
        raise ValueError("cases and rankings must be non-empty and have equal length")
    evaluated = [(case, ranking) for case, ranking in zip(cases, rankings, strict=True) if case.relevant_evidence]
    if not evaluated:
        return {"evidence_hit_rate@5": 0.0, "evidence_mrr@10": 0.0, "evidence_recall@10": 0.0}

    hit_rate = 0.0
    reciprocal_rank = 0.0
    recall = 0.0
    for case, ranking in evaluated:
        matched_predicates: set[int] = set()
        first_rank = 0
        for rank, candidate in enumerate(ranking[:10], start=1):
            for predicate_index, predicate in enumerate(case.relevant_evidence):
                if _matches_evidence(candidate, predicate):
                    matched_predicates.add(predicate_index)
                    if not first_rank:
                        first_rank = rank
        hit_rate += float(any(_matches_any(candidate, case.relevant_evidence) for candidate in ranking[:5]))
        reciprocal_rank += 1.0 / first_rank if first_rank else 0.0
        recall += len(matched_predicates) / len(case.relevant_evidence)

    count = len(evaluated)
    return {
        "evidence_hit_rate@5": hit_rate / count,
        "evidence_mrr@10": reciprocal_rank / count,
        "evidence_recall@10": recall / count,
    }


def answer_evidence_metrics(
    cases: Sequence[RetrievalCase],
    answers: Sequence[str],
) -> dict[str, float]:
    """Measure whether generated answers contain all case-level answer facts."""

    if not cases or len(cases) != len(answers):
        raise ValueError("cases and answers must be non-empty and have equal length")
    evaluated = [
        (case, answer)
        for case, answer in zip(cases, answers, strict=True)
        if case.expected_answer_contains_all
    ]
    if not evaluated:
        return {"answer_evidence_rate": 0.0}
    passed = sum(
        all(term.casefold() in answer.casefold() for term in case.expected_answer_contains_all)
        for case, answer in evaluated
    )
    return {"answer_evidence_rate": passed / len(evaluated)}


def citation_provenance_metrics(
    cases: Sequence[RetrievalCase],
    source_rankings: Sequence[Sequence[Mapping[str, Any] | Any]],
) -> dict[str, float]:
    """Score whether cited sources point to the expected document and pages.

    This complements chunk-text evidence metrics for the end-to-end answer
    path, where the public source contract intentionally contains only a short
    preview.  A generated answer passes only when its returned citation points
    to an accepted document, physical PDF page and printed manual page.
    """

    if not cases or len(cases) != len(source_rankings):
        raise ValueError("cases and source_rankings must be non-empty and have equal length")
    evaluated = [
        (case, ranking)
        for case, ranking in zip(cases, source_rankings, strict=True)
        if case.relevant_evidence
    ]
    if not evaluated:
        return {"citation_provenance_rate": 0.0}
    passed = sum(
        any(_matches_provenance(source, predicate) for source in ranking for predicate in case.relevant_evidence)
        for case, ranking in evaluated
    )
    return {"citation_provenance_rate": passed / len(evaluated)}


def _matches_any(candidate: Mapping[str, Any] | Any, predicates: Sequence[EvidencePredicate]) -> bool:
    return any(_matches_evidence(candidate, predicate) for predicate in predicates)


def _matches_evidence(candidate: Mapping[str, Any] | Any, predicate: EvidencePredicate) -> bool:
    if not _matches_provenance(candidate, predicate):
        return False
    text = str(_candidate_value(candidate, "text", _candidate_value(candidate, "snippet", ""))).casefold()
    if not all(term.casefold() in text for term in predicate.contains_all):
        return False
    return True


def _matches_provenance(candidate: Mapping[str, Any] | Any, predicate: EvidencePredicate) -> bool:
    doc_id = str(_candidate_value(candidate, "doc_id", _candidate_value(candidate, "file_name", "")))
    if doc_id != predicate.doc_id:
        return False
    if predicate.page is not None and _candidate_value(candidate, "page") != predicate.page:
        return False
    return not (
        predicate.printed_page is not None
        and _candidate_value(candidate, "printed_page") != predicate.printed_page
    )


def _candidate_value(candidate: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)
