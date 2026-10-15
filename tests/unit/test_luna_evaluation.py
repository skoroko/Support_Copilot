from __future__ import annotations

import pytest

from scripts.compare_luna import Budget, BudgetExceeded, extract_cases, usage_cost
from scripts.judge_luna import blind_case


def test_billing_counts_read_write_and_reasoning_exactly_once():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 800,
        "input_token_details": {"cache_read": 300, "cache_creation": 200},
        "output_token_details": {"reasoning": 600},
    }
    expected = (500 * 0.1 + 300 * 0.01 + 200 * 0.125 + 800 * 0.5) / 1e6
    assert usage_cost("gpt-6-luna", usage) == pytest.approx(expected)
    with pytest.raises(ValueError, match="Missing usage"):
        usage_cost("gpt-6-luna", {})


def test_budget_reserves_concurrent_calls_and_releases_unused_cap():
    budget = Budget(1, spent=0.2)
    budget.reserve(0.5)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.4)
    budget.settle(0.5, 0.1)
    budget.reserve(0.6)
    assert budget.spent == pytest.approx(0.3)


def test_evidence_is_taken_from_each_turn_and_all_searches_not_latest_thread():
    import json

    def tool(number, text):
        return {
            "type": "tool",
            "name": "search_knowledge_base",
            "content": json.dumps(
                {
                    "status": "ok",
                    "evidence": [
                        {
                            "citation_number": number,
                            "text": text,
                            "source": {"id": str(number)},
                        }
                    ],
                }
            ),
        }

    run = {
        "scenarios": [{"id": "seed", "reference": "SCORING ONLY"}, {"id": "followup"}]
    }
    cp = {
        "threads": {
            "a": {
                "scenario_ids": ["seed", "followup"],
                "messages": [
                    {"type": "human"},
                    tool(1, "first"),
                    tool(2, "second"),
                    {"type": "human"},
                    tool(1, "followup evidence"),
                    {"type": "human"},
                    tool(1, "expansion must not overwrite seed"),
                ],
            }
        }
    }
    cases = extract_cases(run, cp)
    assert [r["text"] for r in cases[0]["records"]] == ["first", "second"]
    assert [r["text"] for r in cases[1]["records"]] == ["followup evidence"]


def test_blind_judge_hides_names_and_maps_public_citations_to_exact_chunks():
    row = {
        "case_id": "golden-01",
        "status": "completed",
        "answer": "Факт [1].",
        "answer_info": {"incomplete": False},
        "sources": [{"chunk_id": "chunk-b"}],
        "evidence": [{"source": {"chunk_id": "chunk-b"}, "text": "Подтверждение"}],
    }
    payload = {
        "cases": [
            {
                "id": "golden-01",
                "conversation": "one",
                "question": "Вопрос",
                "group": "golden",
                "records": [],
                "reference": "Ориентир",
            }
        ],
        "variants": {n: {"results": [row]} for n in ("mini", "luna-high")},
    }
    mapping, data = blind_case(payload, "golden-01")
    assert set(mapping.values()) == {"mini", "luna-high"}
    assert all(c["citations"]["1"].startswith("chunk-b@") for c in data["candidates"])
    assert all(
        data["documents"][c["citations"]["1"]] == "Подтверждение"
        for c in data["candidates"]
    )
    assert not any(n in str(data) for n in payload["variants"])


def test_report_rejects_ratings_for_different_answer_or_tampered_input(tmp_path):
    import hashlib
    import json

    from scripts.report_luna import merge_reviews

    answer = {
        "case_id": "one",
        "answer": "Первый ответ",
        "sources": [],
        "evidence": [],
        "answer_info": {},
    }
    packet = {"variants": {"mini": {"results": [answer]}}}
    source, review_path = tmp_path / "input.json", tmp_path / "review.json"
    source.write_text(json.dumps(packet), encoding="utf-8")
    review = {
        "signature": {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "scores": [{"variant": "mini", "case_id": "one"}],
        "calls": [],
    }
    review_path.write_text(json.dumps(review), encoding="utf-8")
    assert len(merge_reviews(packet, [review_path], [source])[0]) == 1
    changed = {
        "variants": {"mini": {"results": [{**answer, "answer": "Другой ответ"}]}}
    }
    with pytest.raises(ValueError, match="differs"):
        merge_reviews(changed, [review_path], [source])
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_reviews(packet, [review_path], [source])
