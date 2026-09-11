from __future__ import annotations

import json
from pathlib import Path

from eval.check_thresholds import check_thresholds
from eval.run_evaluation import build_judge_messages, calculate_aggregates, load_golden, normalize_judge_result
from app.services.retrieval_eval import load_retrieval_dataset
from scripts.evaluate_rag_quality import evaluate as evaluate_rag_quality


ROOT = Path(__file__).resolve().parents[2]


def test_golden_dataset_meets_homework_shape() -> None:
    golden = load_golden(ROOT / "eval" / "golden_dataset.json")
    items = golden["items"]

    categories = {item["category"] for item in items}
    hard_items = [item for item in items if item["difficulty"] == "hard"]

    assert golden["version"] == 1
    assert len(items) >= 20
    assert len(categories) >= 3
    assert len(hard_items) >= 3
    assert all({"id", "question", "expected_answer", "expected_keywords", "category", "difficulty", "source"} <= set(item) for item in items)


def test_judge_prompt_requires_reasoning_before_scores() -> None:
    item = {
        "question": "Как проверить лицензию?",
        "expected_answer": "Нужно запросить версию и идентификатор лицензии.",
        "expected_keywords": ["версия", "лицензия"],
        "must_not_contain": ["гарантирую"],
    }

    messages = build_judge_messages(item, "Ответ кандидата")
    system = messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])

    assert "reason-then-score" in system
    assert system.index("reasoning") < system.index("scores")
    assert list(user_payload["required_json_shape"]) == ["reasoning", "scores", "explanation"]


def test_normalize_judge_result_validates_score_range() -> None:
    normalized = normalize_judge_result(
        {
            "reasoning": "ok",
            "scores": {"relevance": 5, "correctness": 4, "completeness": 3},
            "explanation": "fine",
        }
    )

    assert normalized["scores"]["correctness"] == 4


def test_calculate_aggregates_matches_required_fields() -> None:
    aggregates = calculate_aggregates(
        [
            {"scores": {"relevance": 5, "correctness": 4, "completeness": 5}},
            {"scores": {"relevance": 3, "correctness": 2, "completeness": 4}},
        ]
    )

    assert aggregates == {
        "relevance_avg": 4.0,
        "correctness_avg": 3.0,
        "completeness_avg": 4.5,
        "min_correctness": 2,
    }


def test_check_thresholds_reports_each_failed_metric() -> None:
    failures = check_thresholds(
        {"aggregates": {"correctness_avg": 3.8, "min_correctness": 1}},
        {"correctness_avg": 4.0, "min_correctness": 2.0},
    )

    assert failures == ["correctness_avg: 3.8 < 4.0", "min_correctness: 1.0 < 2.0"]


def test_rag_quality_eval_checks_answers_and_citation_pages() -> None:
    cases = load_retrieval_dataset(ROOT / "tests" / "eval" / "retrieval_dataset.json")
    by_question = {case.question: case for case in cases}

    class FakeRAGService:
        def answer(self, question: str, *, history):
            case = by_question[question]
            source = {"file_name": case.relevant_doc_ids[0], "page": None, "printed_page": None}
            answer = "Проверяемый ответ [1]."
            if case.relevant_evidence:
                predicate = case.relevant_evidence[0]
                source.update(page=predicate.page, printed_page=predicate.printed_page)
                answer = " ".join(case.expected_answer_contains_all) + " [1]."
            return {"answer": answer, "top_score": 0.9, "sources": [source]}

    payload = evaluate_rag_quality(
        FakeRAGService(),  # type: ignore[arg-type]
        ROOT / "tests" / "eval" / "retrieval_dataset.json",
    )

    assert payload["case_count"] == len(cases)
    assert payload["metrics"]["hit_rate@5"] == 1.0
    assert payload["metrics"]["citation_provenance_rate"] == 1.0
    assert payload["metrics"]["answer_evidence_rate"] == 1.0
    assert payload["metrics"]["answer_focus_rate"] == 1.0
