from __future__ import annotations

import json
from pathlib import Path

from eval.check_thresholds import check_thresholds
from eval.run_evaluation import build_judge_messages, calculate_aggregates, load_golden, normalize_judge_result


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
