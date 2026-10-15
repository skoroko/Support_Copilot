import json
from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from app.services.retrieval_eval import EvidencePredicate, RetrievalCase
from bot.services.backend_client import BackendClient
from scripts import evaluate_telegram_agent as evaluation
from scripts.export_agent_eval_evidence import evaluation_threads, message_record

CHAT = UUID("00000000-0000-0000-0000-000000000001")
BRIEF = "00000000-0000-0000-0000-000000000002"
EXPANDED = "00000000-0000-0000-0000-000000000003"
OWNER = "eval-offline-scenario"


def definition(ident="one", group="golden", **extra):
    return {"id": ident, "group": group, "conversation": ident, "golden_index": 1,
            "question": "Вопрос?", "retrieval_contract": None, **extra}


def transport(*, stream_error=False, missing_history=False, source_mismatch=False):
    source = {"file_name": "Manual.pdf", "page": 32, "printed_page": 45}
    brief = {"id": BRIEF, "role": "assistant", "content": "Итог [1].", "latency_ms": 15,
             "rag_sources": [] if source_mismatch else [source], "media_refs": {"answer": {"expandable": True}}}
    history = [] if missing_history else [brief]
    calls = {"confirm": 0, "generation": 0}
    result = {"answer": "Подробный ответ [1].", "sources": [source], "message_id": EXPANDED,
              "answer_info": {"mode": "expanded", "expandable": False}}

    def handler(request):
        path = request.url.path
        assert request.headers["X-User-ID"] == OWNER
        assert request.headers["X-User-Interface"] == "telegram"
        if path == "/chats":
            return httpx.Response(200, json={"chat_id": str(CHAT)})
        if request.method == "GET" and path.endswith("/messages"):
            return httpx.Response(200, json=history)
        if request.method == "POST" and path.endswith("/messages"):
            events = [{"type": "token", "delta": "Неверный черновик"}, {"type": "replace", "text": brief["content"]}]
            body = "".join("data: " + json.dumps(item) + "\n\n" for item in events)
            if stream_error:
                body += 'event: error\ndata: {"type":"error","code":"chat_stream_failed","message":"private body"}\n\n'
            else:
                body += "data: " + json.dumps({"type": "message", "message_id": BRIEF, "answer_info": {"expandable": True}}) + "\n\n"
                body += 'data: {"type":"done"}\n\n'
                body += "event: sources\ndata: " + json.dumps({"sources": [source]}) + "\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.method == "GET" and path.endswith("/expansion"):
            return httpx.Response(200, json={"available": True, "max_output_tokens": 1500})
        if path.endswith("/expand"):
            body = json.loads(request.content)
            if body["confirmed"] is False:
                return httpx.Response(422, json={"detail": "private validation body"})
            assert body == {"confirmed": True}
            calls["confirm"] += 1
            if calls["confirm"] == 1:
                calls["generation"] += 1
                history.extend([{"id": "user-expansion", "role": "user", "content": "Раскрой подробнее."},
                                {"id": EXPANDED, "role": "assistant", "content": result["answer"], "rag_sources": [source]}])
            return httpx.Response(200, json=result)
        raise AssertionError((request.method, path))

    return httpx.MockTransport(handler), calls


@pytest.mark.asyncio
async def test_sse_reads_sources_after_done_and_replaces_draft():
    mock, _ = transport()
    async with httpx.AsyncClient(base_url="http://offline", transport=mock) as http:
        backend = BackendClient("http://offline", client=http, internal_token="offline", max_connect_attempts=1)
        chat = await backend.get_or_create_chat(OWNER, "telegram")
        row = {}
        await evaluation.collect_answer(backend, chat, "Вопрос?", row)
    assert row["status"] == "completed"
    assert row["response"] == "Итог [1]."
    assert row["sources"][0]["page"] == 32
    assert row["message_id"] == BRIEF
    assert row["transport_matches_persisted"] and row["sources_match_persisted"]
    assert row["persisted"]["latency_ms"] == 15 and row["wall_latency_ms"] >= row["ttft_ms"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("options,error_type", [({"stream_error": True}, "BackendStreamError"), ({"missing_history": True}, "ValueError")])
async def test_partial_or_missing_persistence_is_failed_and_safe(options, error_type):
    mock, _ = transport(**options)
    async with httpx.AsyncClient(base_url="http://offline", transport=mock) as http:
        backend = BackendClient("http://offline", client=http, internal_token="offline", max_connect_attempts=1)
        chat = await backend.get_or_create_chat(OWNER, "telegram")
        row = {}
        await evaluation.collect_answer(backend, chat, "Вопрос?", row)
    assert row["status"] == "failed" and row["error"] == {"type": error_type}
    assert row["response"] == "Итог [1]." and "private" not in json.dumps(row)


@pytest.mark.asyncio
async def test_source_transport_mismatch_is_recorded():
    mock, _ = transport(source_mismatch=True)
    async with httpx.AsyncClient(base_url="http://offline", transport=mock) as http:
        backend = BackendClient("http://offline", client=http, internal_token="offline")
        chat = await backend.get_or_create_chat(OWNER, "telegram")
        row = {}
        await evaluation.collect_answer(backend, chat, "Вопрос?", row)
    assert row["sources_match_persisted"] is False


def test_structural_metrics_separate_failures_groups_and_page_mismatch():
    case = RetrievalCase("Вопрос?", ("Manual.pdf",), relevant_evidence=(EvidencePredicate("Manual.pdf", (), 32, 45),))
    definitions = [definition(retrieval_contract=asdict(case)), definition("failed", "followup", retrieval_contract=asdict(case)), definition("unbound", "clarification")]
    rows = [{"id": "one", "status": "completed", "response": "Ответ [1].", "sources": [{"file_name": "Manual.pdf", "page": 31, "printed_page": 45}]},
            {"id": "failed", "status": "failed"}]
    result = evaluation.structural_proxies(definitions, rows)
    assert result["metrics"]["documents"]["denominator"] == 1
    assert result["metrics"]["documents"]["values"]["hit_rate@5"] == 1
    assert result["metrics"]["provenance"]["values"]["citation_provenance_rate"] == 0
    assert result["groups"]["followup"]["failed"] == 1
    assert result["groups"]["clarification"]["other"] == 1
    assert result["metrics"]["abstention"] == {"denominator": 0, "values": {}}
    assert result["semantic_review"] == "pending"


@pytest.mark.asyncio
async def test_confirmed_expansion_checks_preview_rejection_and_idempotent_replay():
    mock, calls = transport()
    async with httpx.AsyncClient(base_url="http://offline", transport=mock) as http:
        backend = BackendClient("http://offline", client=http, internal_token="offline")
        chat = await backend.get_or_create_chat(OWNER, "telegram")
        row = {"message_id": BRIEF}
        headers = {"X-Internal-Token": "offline", "X-User-ID": OWNER, "X-User-Interface": "telegram"}
        checkpoints = []
        await evaluation.collect_expansion(backend, http, headers, chat, OWNER, row, lambda: checkpoints.append(row["expansion"]["status"]))
    assert calls == {"confirm": 2, "generation": 1}
    assert row["expansion"]["status"] == "completed"
    assert row["expansion"]["false_confirmation_http_status"] == 422
    assert row["expansion"]["preview_preserves_history"]
    assert row["expansion"]["false_confirmation_preserves_history"]
    assert row["expansion"]["replay_equal"] and row["expansion"]["replay_preserves_history"]
    assert row["expansion"]["replay_message_id"] == EXPANDED
    assert "confirmation_started" in checkpoints


def test_scenario_groups_bind_by_question_with_one_explicit_equivalence():
    definitions = evaluation.scenarios()
    assert len(definitions) == 43
    golden = [item for item in definitions if item["group"] == "golden"]
    assert len(golden) == 34 and golden[33]["retrieval_binding"].startswith("intentional_index_34")
    assert golden[30]["retrieval_binding"] == "unmatched"
    assert sum(item["group"] == "followup" for item in definitions) == 3
    assert sum(item["group"] == "clarification" for item in definitions) == 1
    assert sum(item["group"] == "refusal" for item in definitions) == 2


@pytest.mark.asyncio
async def test_guard_before_configuration_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "get_settings", lambda: pytest.fail("configuration read before guard"))
    with pytest.raises(ValueError, match="allow-external-data-transfer"):
        await evaluation.run(Namespace(allow_external_data_transfer=False))


@pytest.mark.asyncio
async def test_resume_never_retries_saved_failed_or_interrupted_requests(tmp_path, monkeypatch):
    definitions = [definition("failed"), definition("started")]
    monkeypatch.setattr(evaluation, "scenarios", lambda: definitions)
    monkeypatch.setattr(evaluation, "get_settings", lambda: SimpleNamespace(internal_token=SimpleNamespace(get_secret_value=lambda: "offline")))
    monkeypatch.setattr(evaluation.BackendClient, "get_or_create_chat", lambda *args: pytest.fail("resume replayed a billed request"))
    output = tmp_path / "run.json"
    evaluation.save(output, {"run_id": "offline", "options": {"api_url": "http://offline", "expansion_indices": [1], "no_expansions": False},
                             "scenarios": definitions, "results": [{"id": "failed", "status": "failed"}, {"id": "started", "status": "started"}]})
    args = Namespace(allow_external_data_transfer=True, limit=None, expansion_indices="1", api_url="http://offline", no_expansions=False,
                     output=output, resume_safe=True, runtime_manifest=None)
    result = await evaluation.run(args)
    assert [row["status"] for row in result["results"]] == ["failed", "started"]


def test_evidence_export_refuses_other_owners_and_deduplicates_eval_chat():
    run_id = "a" * 32
    row = {"owner": f"eval-{run_id}-followup", "chat_id": str(CHAT), "id": "seed"}
    threads = evaluation_threads({"run_id": run_id, "results": [row, {**row, "id": "followup"}]})
    assert len(threads) == 1
    assert next(iter(threads.values()))["scenario_ids"] == ["seed", "followup"]
    with pytest.raises(ValueError, match="non-evaluation owner"):
        evaluation_threads({"run_id": run_id, "results": [{**row, "owner": "real-user"}]})


def test_evidence_message_export_omits_unrelated_provider_metadata():
    from langchain_core.messages import AIMessage

    message = AIMessage(content="Ответ [1].", response_metadata={"finish_reason": "stop", "unrelated": "omit"})
    record = message_record(message)
    assert record["content"] == "Ответ [1]." and record["finish_reason"] == "stop"
    assert "response_metadata" not in record and "omit" not in json.dumps(record)


def test_checkpoint_retries_only_local_replace_when_windows_reader_blocks(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.replace
    attempts = []

    def blocked_once(path, destination):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("temporary local sharing lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", blocked_once)
    monkeypatch.setattr(evaluation.time, "sleep", lambda _delay: None)
    output = tmp_path / "run.json"
    evaluation.save(output, {"scenarios": [], "results": []})
    assert len(attempts) == 2 and json.loads(output.read_text())["results"] == []
