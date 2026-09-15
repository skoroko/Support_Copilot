"""Offline contract tests for HW 6.5; no provider or RAG service is contacted."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from experiments.multi_agent_langgraph import build_multi_agent_app, run_comparison, run_multi_agent_case
from experiments.shared import (
    EXPERIMENT_CASES,
    HANDOFF_BUDGET,
    judge_answers_batched,
    parse_judge_scores,
    pending_results_payload,
    search_knowledge_base,
    search_knowledge_base_documents,
    usage_from_ai_messages,
)
from experiments.single_agent_baseline import build_single_agent, run_single_agent_case


class ToolCapableFakeModel(FakeMessagesListChatModel):
    """LangChain's fake model with the one missing capability create_agent needs."""

    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any) -> "ToolCapableFakeModel":
        return self


def _usage_message(content: str, *, tool_call: bool = False) -> AIMessage:
    tool_calls = []
    if tool_call:
        tool_calls = [
            {
                "id": "call-kb",
                "name": "search_knowledge_base",
                "args": {"query": "лицензия"},
            }
        ]
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )


def test_common_mock_tool_is_structured_and_returns_an_explicit_out_of_base_result() -> None:
    docs = search_knowledge_base_documents("Ошибка активации лицензии")
    assert len(docs) == 3
    assert all({"content", "source", "score"} <= set(document) for document in docs)
    assert search_knowledge_base.name == "search_knowledge_base"

    absent = search_knowledge_base_documents("квантовый телепорт MIDAS-QX9")
    assert len(absent) == 2
    assert all(document["score"] == 0.0 for document in absent)
    assert absent[1]["id"] == "kb-escalation"


def test_single_agent_measures_all_model_turns_and_has_no_handoffs() -> None:
    fake = ToolCapableFakeModel(
        responses=[
            _usage_message("", tool_call=True),
            _usage_message("Попросите код ошибки [1].\nИсточники: [1] KB/MIDAS-Licensing/activation.md."),
        ]
    )
    record = run_single_agent_case(build_single_agent(fake), EXPERIMENT_CASES[0])

    assert record["run_status"] == "completed"
    assert record["llm_calls"] == 2
    assert record["total_tokens"] == 26
    assert record["handoff_count"] == 0
    assert "[1]" in record["answer"]


def test_manual_supervisor_streams_three_visible_handoffs_and_sums_all_agent_messages() -> None:
    fake = ToolCapableFakeModel(
        responses=[
            _usage_message("", tool_call=True),
            _usage_message("- Запросить код ошибки.\nИсточники: [1] KB/MIDAS-Licensing/activation.md."),
            _usage_message("Попросите код ошибки [1].\nИсточники: [1] KB/MIDAS-Licensing/activation.md."),
        ]
    )
    app = build_multi_agent_app(fake)
    updates: list[dict[str, Any]] = []
    record = run_multi_agent_case(app, EXPERIMENT_CASES[0], on_update=updates.append)

    assert record["run_status"] == "completed"
    assert record["llm_calls"] == 3  # tool-call turn, researcher facts, writer answer
    assert record["total_tokens"] == 39
    assert record["handoff_count"] == HANDOFF_BUDGET == 3
    assert record["handoff_log"] == [
        "supervisor -> researcher",
        "researcher -> supervisor",
        "supervisor -> writer",
    ]
    assert "[1]" in record["answer"]
    assert {"supervisor", "researcher", "writer"} <= {node for update in updates for node in update}
    mermaid = app.get_graph().draw_mermaid()
    assert all(node in mermaid for node in ("supervisor", "researcher", "writer"))


def test_usage_counts_tool_call_ai_messages_and_zero_usage_honestly() -> None:
    measured = usage_from_ai_messages([_usage_message("", tool_call=True), AIMessage(content="no metadata")])
    assert measured == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13, "llm_calls": 2}


def test_one_batched_judge_has_unambiguous_ids_and_a_safe_parser_fallback() -> None:
    rows = [
        {"implementation": "single_agent_baseline", "case_id": "01", "question": "q1", "answer": "a1"},
        {"implementation": "multi_agent_langgraph", "case_id": "01", "question": "q1", "answer": "a2"},
    ]
    fake = ToolCapableFakeModel(
        responses=[
            _usage_message(
                json.dumps(
                    {
                        "scores": [
                            {"case_id": "single_agent_baseline:01", "score": 4, "reason": "ok"},
                            {"case_id": "multi_agent_langgraph:01", "score": 5, "reason": "ok"},
                        ]
                    }
                )
            )
        ]
    )

    scores, judge_usage = judge_answers_batched(fake, rows)
    assert scores == {"single_agent_baseline:01": 4.0, "multi_agent_langgraph:01": 5.0}
    assert judge_usage["llm_calls"] == 1
    assert judge_usage["total_tokens"] == 13
    assert parse_judge_scores("not json", ["only"]) == {"only": None}


def test_pending_template_is_exactly_ten_honest_non_live_records() -> None:
    payload = pending_results_payload()
    assert payload["status"] == "pending_live_run"
    assert len(payload["records"]) == 10
    assert {record["case_id"] for record in payload["records"]} == {case["id"] for case in EXPERIMENT_CASES}
    assert all(record["total_tokens"] is None and record["latency_ms"] is None for record in payload["records"])
    assert {record["handoff_count"] for record in payload["records"] if record["implementation"] == "single_agent_baseline"} == {0}
    assert {record["handoff_count"] for record in payload["records"] if record["implementation"] == "multi_agent_langgraph"} == {3}


def test_checked_in_live_results_have_ten_completed_cited_records_and_real_metrics() -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments" / "results.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "completed_live_run"
    assert payload["judge"]["status"] == "completed_one_batched_call"
    assert len(payload["records"]) == 10
    for record in payload["records"]:
        assert {"total_tokens", "llm_calls", "latency_ms", "handoff_count", "answer", "quality_score"} <= set(record)
        assert record["run_status"] == "completed"
        assert isinstance(record["total_tokens"], int) and record["total_tokens"] > 0
        assert isinstance(record["llm_calls"], int) and record["llm_calls"] > 0
        assert isinstance(record["latency_ms"], float) and record["latency_ms"] > 0
        assert isinstance(record["quality_score"], float) and 1.0 <= record["quality_score"] <= 5.0
        assert "[1]" in record["answer"]

    single = [record for record in payload["records"] if record["implementation"] == "single_agent_baseline"]
    multi = [record for record in payload["records"] if record["implementation"] == "multi_agent_langgraph"]
    assert len(single) == len(multi) == 5
    assert sum(record["total_tokens"] for record in single) / 5 == 1100.2
    assert sum(record["total_tokens"] for record in multi) / 5 == 1470.2
    assert median(record["latency_ms"] for record in single) == 2248.48
    assert median(record["latency_ms"] for record in multi) == 3987.97
    assert {record["llm_calls"] for record in single} == {2}
    assert {record["llm_calls"] for record in multi} == {3}
    assert {record["handoff_count"] for record in single} == {0}
    assert {record["handoff_count"] for record in multi} == {3}
    assert {record["quality_score"] for record in single} == {5.0}
    assert [record["quality_score"] for record in multi] == [1.0, 4.0, 4.0, 1.0, 5.0]


def test_comparison_uses_five_by_two_records_and_excludes_judge_when_skipped() -> None:
    # Per case: single tool-call + final, then researcher tool-call + facts + writer.
    responses: list[AIMessage] = []
    for _case in EXPERIMENT_CASES:
        responses.extend(
            [
                _usage_message("", tool_call=True),
                _usage_message("Single answer [1]."),
                _usage_message("", tool_call=True),
                _usage_message("- fact [1]."),
                _usage_message("Multi answer [1]."),
            ]
        )
    payload = run_comparison(model=ToolCapableFakeModel(responses=responses), score_with_judge=False)

    assert len(payload["records"]) == 10
    assert payload["judge"]["status"] == "skipped"
    assert [record["implementation"] for record in payload["records"]] == [
        implementation
        for _case in EXPERIMENT_CASES
        for implementation in ("single_agent_baseline", "multi_agent_langgraph")
    ]
