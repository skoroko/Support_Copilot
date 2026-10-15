"""Collect real Telegram-agent answers through the authenticated HTTP facade.

No Telegram delivery is requested. Principals deliberately cannot be Telegram
chat IDs. Structural metrics are proxies; semantic review remains a separate step.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.services.rag import NO_ANSWER
from app.services.retrieval_eval import (
    RetrievalCase,
    abstention_metrics,
    answer_evidence_metrics,
    answer_focus_metrics,
    citation_focus_metrics,
    citation_provenance_metrics,
    load_retrieval_dataset,
    retrieval_metrics,
)
from bot.services.backend_client import BackendClient, StreamReplace


def normalized(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def scenarios() -> list[dict]:
    golden = json.loads((ROOT / "tests/eval/golden_dataset.json").read_text(encoding="utf-8"))
    retrieval = load_retrieval_dataset(ROOT / "tests/eval/retrieval_dataset.json")
    by_question = {normalized(case.question): case for case in retrieval}
    result = []
    for index, item in enumerate(golden, 1):
        case = by_question.get(normalized(item["user_input"]))
        binding = "normalized_question" if case else "unmatched"
        if index == 34:
            # Deliberate equivalence: explicit plane-strain question in golden;
            # contextual DOF follow-up in retrieval. Do not inject its history.
            case, binding = retrieval[33], "intentional_index_34_same_plane_strain_target_no_history_injection"
        result.append({"id": f"golden-{index:02}", "group": "golden", "conversation": f"golden-{index:02}",
                       "golden_index": index, "question": item["user_input"],
                       "reference": item["reference"], "reference_contexts": item["reference_contexts"],
                       "retrieval_binding": binding, "retrieval_contract": asdict(case) if case else None,
                       "expected_behavior": "answer_or_documented_abstention"})
    for index, followup in ((1, "А какие из них используются для плоской деформации?"),
                            (10, "А какие у неё ограничения?"), (16, "А как они проверяют сходимость?")):
        seed = dict(result[index - 1])
        conversation = f"followup-{index:02}"
        seed.update(id=conversation + "-seed", group="followup_seed", conversation=conversation, golden_index=None)
        result.append(seed)
        case = retrieval[33] if index == 1 else None
        result.append({"id": conversation + "-question", "group": "followup", "conversation": conversation,
                       "question": followup, "depends_on": seed["id"], "golden_index": None,
                       "reference": golden[33]["reference"] if index == 1 else None,
                       "reference_contexts": golden[33]["reference_contexts"] if index == 1 else [],
                       "retrieval_binding": "intentional_plane_strain_followup" if case else "semantic_review_only",
                       "retrieval_contract": asdict(case) if case else None,
                       "expected_behavior": "resolve_history_and_search_for_supporting_evidence"})
    for ident, group, question, expectation in (
        ("ambiguous-cold", "clarification", "Какие параметры нужно задать?", "ask_for_product_task_or_model_before_specific_parameters"),
        ("refusal-discount", "refusal", "Какая точная цена лицензии GTS NX и какую скидку MIDAS гарантирует мне сегодня?", "abstain_from_unverified_price_and_discount"),
        ("refusal-sla", "refusal", "Какой гарантированный SLA MIDAS по моему договору и за сколько часов вы обязаны исправить ошибку?", "abstain_from_unverified_contract_and_sla"),
    ):
        case = RetrievalCase(question, (), expected_no_answer=True, max_cited_sources=0) if group == "refusal" else None
        result.append({"id": ident, "group": group, "conversation": ident, "question": question,
                       "golden_index": None, "reference": None, "reference_contexts": [],
                       "retrieval_binding": "supplemental_behavior_contract", "retrieval_contract": asdict(case) if case else None,
                       "expected_behavior": expectation})
    return result


def contract(raw: dict) -> RetrievalCase:
    from app.services.retrieval_eval import EvidencePredicate, RetrievalHistoryItem
    values = dict(raw)
    values["relevant_evidence"] = tuple(EvidencePredicate(**item) for item in values.get("relevant_evidence", []))
    values["history"] = tuple(RetrievalHistoryItem(**item) for item in values.get("history", []))
    return RetrievalCase(**values)


def structural_proxies(definitions: list[dict], rows: list[dict], *, include_scopes: bool = True) -> dict:
    completed = {row["id"]: row for row in rows if row["status"] == "completed"}
    bound = [(contract(item["retrieval_contract"]), completed[item["id"]]) for item in definitions
             if item.get("retrieval_contract") and item["id"] in completed]
    specifications = (
        ("documents", lambda case: not case.expected_no_answer, lambda cases, answers, sources: retrieval_metrics(cases, [[str(s.get("file_name") or "") for s in ss] for ss in sources])),
        ("provenance", lambda case: bool(case.relevant_evidence), lambda cases, answers, sources: citation_provenance_metrics(cases, sources)),
        ("answer_terms", lambda case: bool(case.expected_answer_contains_all), lambda cases, answers, sources: answer_evidence_metrics(cases, answers)),
        ("excluded_terms", lambda case: bool(case.expected_answer_excludes_all), lambda cases, answers, sources: answer_focus_metrics(cases, answers)),
        ("abstention", lambda case: case.expected_no_answer, lambda cases, answers, sources: abstention_metrics(cases, answers, sources, no_answer_text=NO_ANSWER)),
        ("citation_count", lambda case: case.max_cited_sources is not None, lambda cases, answers, sources: citation_focus_metrics(cases, sources)),
    )
    metrics = {}
    for name, predicate, measure in specifications:
        selected = [(case, row) for case, row in bound if predicate(case)]
        metrics[name] = {"denominator": len(selected), "values": measure(
            [case for case, _ in selected], [row["response"] for _, row in selected],
            [row["sources"] for _, row in selected]) if selected else {}}
    groups = {}
    for definition in definitions:
        counts = groups.setdefault(definition["group"], {"planned": 0, "completed": 0, "failed": 0, "other": 0})
        counts["planned"] += 1
        row = next((item for item in rows if item["id"] == definition["id"]), None)
        status = row["status"] if row else "other"
        counts[status if status in {"completed", "failed"} else "other"] += 1
    result = {"label": "structural_proxies_not_correctness_or_faithfulness", "groups": groups, "metrics": metrics,
              "semantic_review": "pending", "failed_requests_excluded_from_metric_denominators": True}
    if include_scopes:
        golden = [item for item in definitions if item["group"] == "golden"]
        golden_ids = {item["id"] for item in golden}
        result["golden_only"] = structural_proxies(golden, [row for row in rows if row["id"] in golden_ids], include_scopes=False)
    return result


def safe_error(exc: Exception) -> dict:
    result = {"type": type(exc).__name__}
    if isinstance(exc, httpx.HTTPStatusError):
        result["http_status"] = exc.response.status_code
    return result


def save(path: Path, payload: dict) -> None:
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["structural_proxies"] = structural_proxies(payload["scenarios"], payload["results"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers (for example a concurrent artifact export) may briefly
    # deny replacement. Retry this local file operation, never the billed call.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


async def collect_answer(backend: BackendClient, chat_id, question: str, row: dict) -> None:
    started = time.perf_counter()
    row.update(response="", ttft_ms=None, sources=[], answer_info={})
    try:
        async for delta in backend.send_message(chat_id, question):
            if row["ttft_ms"] is None and (isinstance(delta, StreamReplace) or delta):
                row["ttft_ms"] = (time.perf_counter() - started) * 1000
            row["response"] = delta.text if isinstance(delta, StreamReplace) else row["response"] + str(delta)
        row.update(sources=backend.last_response_sources(chat_id), answer_info=backend.last_answer_info(chat_id),
                   message_id=str(backend.last_response_message_id(chat_id) or ""), approval=backend.last_approval(chat_id))
        history = await backend.list_messages(chat_id, limit=2)
        persisted = next((item for item in history if str(item["id"]) == row["message_id"]), None)
        if not persisted:
            raise ValueError("canonical persisted answer missing")
        row["persisted"] = persisted
        row["transport_matches_persisted"] = row["response"] == persisted["content"]
        row["sources_match_persisted"] = row["sources"] == persisted.get("rag_sources", [])
        row["response"] = persisted["content"]
        row["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - persist safe failure at evaluation boundary
        row.update(status="failed", error=safe_error(exc))
    finally:
        row["wall_latency_ms"] = (time.perf_counter() - started) * 1000


async def collect_expansion(backend, http, headers, chat_id, owner, row, checkpoint) -> None:
    details = row.setdefault("expansion", {"status": "started"})
    checkpoint()
    started = time.perf_counter()
    try:
        message_id = row["message_id"]
        before = await backend.list_messages(chat_id, limit=50)
        details["preview"] = await backend.preview_expansion(chat_id, message_id, owner_external_id=owner)
        after_preview = await backend.list_messages(chat_id, limit=50)
        details["preview_preserves_history"] = before == after_preview
        response = await http.post(f"/chats/{chat_id}/messages/{message_id}/expand", headers=headers, json={"confirmed": False})
        details["false_confirmation_http_status"] = response.status_code
        details["false_confirmation_preserves_history"] = before == await backend.list_messages(chat_id, limit=50)
        if response.status_code != 422 or not details["preview_preserves_history"] or not details["false_confirmation_preserves_history"]:
            raise ValueError("expansion consent contract failed")
        details["status"] = "confirmation_started"
        checkpoint()
        confirmed_at = time.perf_counter()
        details["result"] = await backend.expand_response(chat_id, message_id, owner_external_id=owner)
        details["confirmed_latency_ms"] = (time.perf_counter() - confirmed_at) * 1000
        checkpoint()
        history = await backend.list_messages(chat_id, limit=50)
        details["persisted"] = next((item for item in history if str(item["id"]) == details["result"]["message_id"]), None)
        replay = await backend.expand_response(chat_id, message_id, owner_external_id=owner)
        details["replay_equal"] = replay == details["result"]
        details["replay_preserves_history"] = history == await backend.list_messages(chat_id, limit=50)
        details["replay_message_id"] = replay.get("message_id")
        if not details["replay_equal"] or not details["replay_preserves_history"] or not details["persisted"]:
            raise ValueError("expansion replay contract failed")
        details["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - persist safe failure at evaluation boundary
        details.update(status="failed", error=safe_error(exc))
    finally:
        details["wall_latency_ms"] = (time.perf_counter() - started) * 1000
        checkpoint()


async def run(args) -> dict:
    if not args.allow_external_data_transfer:
        raise ValueError("External data transfer requires --allow-external-data-transfer")
    definitions = scenarios()
    if args.limit is not None:
        definitions = definitions[:args.limit]
    indices = [int(item) for item in args.expansion_indices.split(",") if item.strip()]
    options = {"api_url": args.api_url.rstrip("/"), "expansion_indices": indices, "no_expansions": args.no_expansions}
    if args.output.exists():
        if not args.resume_safe:
            raise ValueError("Output exists; use --resume-safe or a new output path")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload["scenarios"] != definitions or payload["options"] != options:
            raise ValueError("Resume configuration differs from the saved run")
    else:
        payload = {"schema_version": 1, "run_id": uuid4().hex, "options": options, "scenarios": definitions,
                   "runtime_manifest": json.loads(args.runtime_manifest.read_text(encoding="utf-8")) if args.runtime_manifest else None,
                   "results": [], "notes": ["No real Telegram sends; nonnumeric isolated owners.",
                   "TTFT is first HTTP content event; this agent buffers full generation before emitting it.",
                   "Resume never retries any saved request, including failed or interrupted requests; use a new run for fresh retries.",
                   "Only index 34 is deliberately paired by index; other mismatched questions remain unbound."]}
    checkpoint = lambda: save(args.output, payload)
    checkpoint()
    token = get_settings().internal_token.get_secret_value()
    async with httpx.AsyncClient(base_url=options["api_url"], timeout=httpx.Timeout(180.0)) as http:
        backend = BackendClient(options["api_url"], client=http, internal_token=token, max_connect_attempts=1)
        for definition in definitions:
            # Durable pre-call records make resume safe even after a lost HTTP response.
            if any(row["id"] == definition["id"] for row in payload["results"]):
                continue
            owner = f"eval-{payload['run_id']}-{definition['conversation']}"
            row = {"id": definition["id"], "group": definition["group"], "owner": owner, "status": "started"}
            payload["results"].append(row)
            checkpoint()
            if definition.get("depends_on") and not any(item["id"] == definition["depends_on"] and item["status"] == "completed" for item in payload["results"]):
                row.update(status="failed", error={"type": "DependencyNotCompleted"})
                checkpoint()
                continue
            try:
                chat_id = await backend.get_or_create_chat(owner, "telegram")
                row["chat_id"] = str(chat_id)
                checkpoint()
                await collect_answer(backend, chat_id, definition["question"], row)
                checkpoint()
                if row["status"] == "completed" and not args.no_expansions and definition.get("golden_index") in indices:
                    if row.get("answer_info", {}).get("expandable"):
                        headers = {"X-Internal-Token": token, "X-User-ID": owner, "X-User-Interface": "telegram"}
                        await collect_expansion(backend, http, headers, chat_id, owner, row, checkpoint)
                    else:
                        row["expansion"] = {"status": "unavailable", "reason": "answer_info_not_expandable"}
            except Exception as exc:  # noqa: BLE001 - isolate one evaluation request
                row.update(status="failed", error=safe_error(exc))
            checkpoint()
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum total scenarios, including seeds and supplemental turns")
    parser.add_argument("--resume-safe", action="store_true")
    parser.add_argument("--no-expansions", action="store_true")
    parser.add_argument("--expansion-indices", default="1,10,16,24,32")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main():
    try:
        payload = asyncio.run(run(parse_args()))
    except Exception as exc:  # noqa: BLE001 - never print credential-bearing exception bodies
        print(json.dumps({"status": "failed", "error": safe_error(exc)}))
        raise SystemExit(1) from None
    print(json.dumps({"run_id": payload["run_id"], "groups": payload["structural_proxies"]["groups"]}))


if __name__ == "__main__":
    main()
