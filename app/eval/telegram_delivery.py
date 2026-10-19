"""Real Telegram ACK measurement around the unchanged candidate bot handlers.

Inbound Update/callbacks are injected in process. Telegram polling, native input
latency and native callback acknowledgements are explicitly outside this probe.
No raw message content, recipient ID, callback payload or exception text is logged.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import Chat, Message, Update, User
from aiohttp import ThreadedResolver

from app.eval.candidate_identity import source_identity
from bot.admission import AdmissionMiddleware
from bot.handlers import _burst, router
from bot.handlers._common import RESPONSE_PLACEHOLDER
from bot.handlers.expansion import expansion
from bot.handlers.feedback import show_all_sources
from bot.services.backend_client import BackendClient


def fingerprint(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def resolve_private_recipient(settings, expected_fingerprint: str) -> int:
    values = set(settings.bot_allowed_user_ids)
    if len(values) != 1:
        raise ValueError("exactly one allowed private recipient required")
    recipient = values.pop()
    if recipient <= 0 or fingerprint(recipient) != expected_fingerprint:
        raise ValueError("confirmed recipient fingerprint mismatch")
    return recipient


def validate_candidate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme, parsed.hostname, parsed.port, parsed.path) != (
        "http", "127.0.0.1", 18001, ""
    ) or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("only isolated loopback candidate API :18001 is allowed")


def write_private_result(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class DeliveryRecorder:
    recipient: int
    started: float = field(default_factory=time.monotonic)
    phase: str = "answer"
    events: list[dict] = field(default_factory=list)
    # Objects/payloads stay only in memory, for owner-bound source/expansion callbacks.
    messages: dict[int, Message] = field(default_factory=dict, repr=False)
    callbacks: dict[str, tuple[str, Message]] = field(default_factory=dict, repr=False)

    async def __call__(self, make_request, bot, method):
        name = method.__api_method__
        if name not in {"getMe", "sendMessage", "editMessageText", "deleteMessage",
                        "sendChatAction", "editMessageReplyMarkup"}:
            raise RuntimeError("Telegram method outside delivery probe allowlist")
        target = getattr(method, "chat_id", None)
        if name != "getMe" and target != self.recipient:
            raise RuntimeError("Telegram delivery recipient mismatch")
        text = getattr(method, "text", None)
        body = name in {"sendMessage", "editMessageText"} and text != RESPONSE_PLACEHOLDER
        event = {"method": name, "phase": self.phase, "body": bool(body),
                 "request_ms": round((time.monotonic() - self.started) * 1000, 3),
                 "status": "in_flight"}
        if isinstance(text, str):
            event.update(text_chars=len(text), text_sha256=fingerprint(text))
        markup = getattr(method, "reply_markup", None)
        buttons = [button.callback_data for row in getattr(markup, "inline_keyboard", [])
                   for button in row if button.callback_data]
        event["button_kinds"] = sorted({item.split(":", 1)[0] for item in buttons})
        self.events.append(event)
        try:
            result = await make_request(bot, method)
        except BaseException as exc:
            event.update(status="failed", error_type=type(exc).__name__,
                         finished_ms=round((time.monotonic() - self.started) * 1000, 3))
            raise
        event.update(status="ack", ack_ms=round((time.monotonic() - self.started) * 1000, 3))
        if isinstance(result, Message):
            self.messages[result.message_id] = result.as_(bot)
            for item in buttons:
                kind = item.split(":", 1)[0]
                if kind != "exp" or item.startswith(("exp:a:", "exp:y:")):
                    self.callbacks[kind] = (item, result.as_(bot))
        elif name == "editMessageReplyMarkup":
            message_id = getattr(method, "message_id", None)
            if message_id in self.messages:
                for item in buttons:
                    self.callbacks[item.split(":", 1)[0]] = (item, self.messages[message_id])
        return result

    def metrics(self) -> dict:
        bodies = [event["ack_ms"] for event in self.events
                  if event["status"] == "ack" and event["body"] and event["phase"] == "answer"]
        deliveries = [event["ack_ms"] for event in self.events
                      if event["status"] == "ack" and event["method"] in {
                          "sendMessage", "editMessageText", "editMessageReplyMarkup"}]
        return {"receipt_to_first_body_ack_ms": min(bodies) if bodies else None,
                "receipt_to_last_answer_ack_ms": max(bodies) if bodies else None,
                "receipt_to_last_delivery_ack_ms": max(deliveries) if deliveries else None,
                "telegram_failures": sum(event["status"] == "failed" for event in self.events),
                "acknowledged_body_operations": len(bodies)}


class InjectedCallback:
    """Actual callback handler and real message transport, no fake Telegram ACK."""
    def __init__(self, data: str, message: Message, user: User):
        self.data, self.message, self.from_user = data, message, user
        self.callback_ack_requested = False

    async def answer(self, *args, **kwargs):
        self.callback_ack_requested = True


class ObservedBackend(BackendClient):
    """Retain safely typed failures even when UI handlers consume HTTP errors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures: list[dict] = []
        self.expansion_completed = False

    async def send_message(self, *args, **kwargs):
        try:
            async for value in super().send_message(*args, **kwargs):
                yield value
        except BaseException as exc:
            self.failures.append({"error_type": type(exc).__name__,
                                  "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            raise

    async def expand_response(self, *args, **kwargs):
        try:
            result = await super().expand_response(*args, **kwargs)
            self.expansion_completed = bool(result.get("message_id"))
            return result
        except BaseException as exc:
            self.failures.append({"operation": "expansion", "error_type": type(exc).__name__,
                                  "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            raise


async def run_delivery(*, settings, api_url: str, question: str, scenario_id: str,
                       recipient_fingerprint: str, expected_identity: str,
                       source_root: Path, output: Path, timeout: float = 240,
                       expanded: bool = False, with_sources: bool = True) -> dict:
    validate_candidate_url(api_url)
    recipient = resolve_private_recipient(settings, recipient_fingerprint)
    canonical_path = output.with_suffix(".canonical.json")
    if output.exists() or canonical_path.exists():
        raise ValueError("output already exists; uncertain/failed attempts are never repeated automatically")
    report = {"schema_version": 1, "scenario_id": scenario_id,
              "created_at": datetime.now(UTC).isoformat(), "status": "preflight",
              "recipient_sha256": recipient_fingerprint, "question_sha256": fingerprint(question),
              "candidate_identity_sha256": expected_identity,
              "measurement": "in_process_injected_Update_to_real_Telegram_API_ack",
              "native_polling_measured": False, "native_callback_ack_measured": False,
              "cost_accounting": "shared_candidate_provider_budget_ledger",
              "events": [], "failures": []}
    # Durable checkpoint exists before the first request that could become billed.
    write_private_result(output, report)
    recorder = DeliveryRecorder(recipient)
    backend = None
    session = AiohttpSession()
    session._connector_init["resolver"] = ThreadedResolver()
    session.middleware(recorder)
    try:
        bot = Bot(settings.bot_token.get_secret_value(), session=session)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as http:
            identity_response = await http.get("/__candidate_identity", headers={
                "X-Internal-Token": settings.internal_token.get_secret_value()})
            identity_response.raise_for_status()
            identity = identity_response.json()
            if identity.get("candidate_identity_sha256") != expected_identity:
                raise ValueError("candidate runtime identity mismatch")
            if identity.get("source_sha256") != source_identity(source_root):
                raise ValueError("local handlers differ from candidate runtime bytes")
            if identity.get("loaded_source_sha256") != identity.get("source_sha256"):
                raise ValueError("candidate must attest its loaded source bytes after restart")
            if not identity.get("budget_enforced", False):
                raise ValueError("candidate must attest shared provider budget enforcement")
            report["budget_before"] = identity.get("budget")
            me = await bot.get_me()  # Read only, excluded from receipt timing.
            report["bot_sha256"] = fingerprint(me.id)
            backend = ObservedBackend(api_url, client=http, max_connect_attempts=1,
                                      internal_token=settings.internal_token.get_secret_value(),
                                      admin_token=settings.admin_token.get_secret_value())
            # This backend has already attested the isolated candidate identity.
            # Reset its separate evaluation history before the receipt clock;
            # the working backend and visible Telegram messages are untouched.
            probe_chat = await backend.get_or_create_chat(str(recipient), "telegram")
            await backend.clear_messages(probe_chat)
            if await backend.list_messages(probe_chat, limit=1):
                raise RuntimeError("candidate evaluation history reset failed")
            report["history_reset"] = "isolated_candidate_before_receipt_timer"
            dp = Dispatcher()
            dp.update.outer_middleware(AdmissionMiddleware(user_ids=settings.bot_allowed_user_ids,
                                                            chat_ids=settings.bot_allowed_chat_ids))
            dp["backend"] = backend
            dp["bot_admin_ids"] = settings.bot_admin_ids
            dp.include_router(router)
            user = User(id=recipient, is_bot=False, first_name="Acceptance")
            message = Message(message_id=1, date=datetime.now(UTC),
                              chat=Chat(id=recipient, type="private"), from_user=user, text=question)
            recorder.events.clear()
            recorder.started = time.monotonic()
            report["status"] = "in_flight"
            write_private_result(output, report)
            async with asyncio.timeout(timeout):
                await dp.feed_update(bot, Update(update_id=1, message=message))
                batch = _burst._pending.get(str(recipient))
                if batch is None or batch.task is None:
                    raise RuntimeError("injected update not admitted to text handler")
                await batch.task  # Includes real debounce, streaming, edits and final sends.
                chat_id = await backend.get_or_create_chat(str(recipient), "telegram")
                message_id = backend.last_response_message_id(chat_id)
                if message_id is None or backend.failures:
                    raise RuntimeError("backend response did not complete successfully")
                # Burst handler intentionally consumes UI failures; HTTP success
                # alone is insufficient. Final controls are attached only to the
                # final rendered chunk, so their ACK witnesses the entire send loop.
                if not any(kind in recorder.callbacks for kind in ("fb", "ap")):
                    raise RuntimeError("final answer controls were not acknowledged by Telegram")
                canonical = await backend.list_messages(chat_id, limit=2)
                assistant = next((item for item in reversed(canonical) if item.get("role") == "assistant"), None)
                if assistant is None:
                    raise RuntimeError("canonical persisted assistant answer missing")
                report["canonical_answer_sha256"] = fingerprint(assistant.get("content") or "")
                sources = backend.last_response_sources(chat_id)
                canonical_evidence = {"scenario_id": scenario_id, "question_sha256": fingerprint(question),
                    "candidate_identity_sha256": expected_identity, "brief": {
                        "answer": assistant.get("content") or "", "sources": sources,
                        "message_id": str(message_id), "answer_sha256": report["canonical_answer_sha256"]}}
                write_private_result(canonical_path, canonical_evidence)
                report["canonical_evidence_file"] = canonical_path.name
                report["source_count"] = len(sources)
                report["approval_pending"] = bool(backend.last_approval(chat_id))
                report["answer_info"] = {key: value for key, value in backend.last_answer_info(chat_id).items()
                                         if key in {"mode", "expandable", "response_path"}}
                if sources and not backend.last_approval(chat_id) and "src" not in recorder.callbacks:
                    raise RuntimeError("source controls were not acknowledged by Telegram")
                if backend.last_approval(chat_id) and "ap" not in recorder.callbacks:
                    raise RuntimeError("approval controls were not acknowledged by Telegram")
                if expanded:
                    if "exp" not in recorder.callbacks:
                        raise RuntimeError("expansion controls absent")
                    data, anchor = recorder.callbacks["exp"]
                    recorder.phase = "expansion_preview"
                    await expansion(InjectedCallback(data, anchor, user), backend)
                    # Separate explicit CLI consent, not automatic action approval.
                    data, anchor = recorder.callbacks["exp"]
                    if not data.startswith("exp:y:"):
                        raise RuntimeError("expansion confirmation controls absent")
                    recorder.phase = "expanded_answer"
                    report["expanded_confirmation_at_ms"] = (time.monotonic() - recorder.started) * 1000
                    await expansion(InjectedCallback(data, anchor, user), backend)
                    if not backend.expansion_completed or backend.failures:
                        raise RuntimeError("expanded backend response failed")
                    expanded_messages = await backend.list_messages(chat_id, limit=2)
                    expanded_answer = next((item for item in reversed(expanded_messages)
                                            if item.get("role") == "assistant"), None)
                    if expanded_answer is None:
                        raise RuntimeError("canonical expanded answer missing")
                    canonical_evidence["expanded"] = {"answer": expanded_answer.get("content") or "",
                        "sources": expanded_answer.get("rag_sources", []),
                        "answer_sha256": fingerprint(expanded_answer.get("content") or "")}
                    write_private_result(canonical_path, canonical_evidence)
                if with_sources and "src" in recorder.callbacks:
                    data, anchor = recorder.callbacks["src"]
                    recorder.phase = "documents"
                    before = len(recorder.events)
                    await show_all_sources(InjectedCallback(data, anchor, user), backend)
                    if not any(event["status"] == "ack" and event["body"] for event in recorder.events[before:]):
                        raise RuntimeError("documents panel delivery was not acknowledged")
                report["status"] = "delivered"
    except BaseException as exc:
        report["status"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        report["failures"].append({"error_type": type(exc).__name__,
                                    "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        # Never leave a delayed batch that could generate/send after failure.
        batch = _burst._pending.pop(str(recipient), None)
        if batch and batch.task:
            batch.task.cancel()
            await asyncio.gather(batch.task, return_exceptions=True)
        report.update(recorder.metrics(), events=recorder.events)
        expanded_acks = [event["ack_ms"] for event in recorder.events if event["status"] == "ack"
                         and event["body"] and event["phase"] == "expanded_answer"]
        if expanded_acks and "expanded_confirmation_at_ms" in report:
            report["expanded_confirmation_to_last_body_ack_ms"] = max(expanded_acks) - report["expanded_confirmation_at_ms"]
        if canonical_path.is_file():
            report["canonical_evidence_sha256"] = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
        if backend:
            report["backend_failures"] = backend.failures
        # Best effort read-only snapshot also on failed/timeout generation. The
        # durable provider ledger remains authoritative when this GET fails.
        try:
            async with httpx.AsyncClient(base_url=api_url, timeout=3) as accounting:
                response = await accounting.get("/__candidate_identity", headers={
                    "X-Internal-Token": settings.internal_token.get_secret_value()})
                response.raise_for_status()
                report["budget_after"] = response.json().get("budget")
        except Exception as exc:  # noqa: BLE001 - snapshot failure must not disclose authorization headers
            report["cost_snapshot_error_type"] = type(exc).__name__
        write_private_result(output, report)
        await session.close()
    return report
