from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, User
from pydantic import SecretStr

from app.eval.candidate_identity import (
    source_identity,
    validate_isolation,
)
from app.eval.telegram_delivery import (
    DeliveryRecorder,
    fingerprint,
    resolve_private_recipient,
    validate_candidate_url,
)


def test_recipient_must_be_exact_unique_confirmed_private_user():
    assert resolve_private_recipient(SimpleNamespace(bot_allowed_user_ids=[42]), fingerprint(42)) == 42
    for values, expected in (([], fingerprint(42)), ([42, 43], fingerprint(42)),
                             ([42], fingerprint(43)), ([-42], fingerprint(-42))):
        with pytest.raises(ValueError):
            resolve_private_recipient(SimpleNamespace(bot_allowed_user_ids=values), expected)


@pytest.mark.parametrize("url", ["http://127.0.0.1:18000", "http://localhost:18001",
    "https://127.0.0.1:18001", "http://user:pass@127.0.0.1:18001", "http://127.0.0.1:18001/path"])
def test_production_and_ambiguous_urls_rejected(url):
    with pytest.raises(ValueError):
        validate_candidate_url(url)


def test_source_identity_covers_relative_names_and_exact_bytes(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "bot").mkdir()
    path = tmp_path / "app" / "example.py"
    path.write_text("x=1", encoding="utf-8")
    previous = source_identity(tmp_path)
    path.write_text("x = 1", encoding="utf-8")
    assert source_identity(tmp_path) != previous
    previous = source_identity(tmp_path)
    path.rename(tmp_path / "bot" / "example.py")
    assert source_identity(tmp_path) != previous


def test_isolation_rejects_production_collection_and_ingestion():
    settings = SimpleNamespace(rag_collection="s0005_candidate_v1",
        qdrant_url="http://host.docker.internal:16333", chat_database_url="postgresql+asyncpg://candidate:candidate@postgres:5432/candidate",
        redis_url="redis://redis:6379/0", rag_ingest_on_start=False, rag_rebuild_on_start=False,
        bot_url="http://127.0.0.1:9")
    validate_isolation(settings)
    settings.rag_collection = "rag_eval_chunk_1024"
    with pytest.raises(ValueError):
        validate_isolation(settings)
    settings.rag_collection = "s0005_candidate_v1"
    settings.rag_ingest_on_start = True
    with pytest.raises(ValueError):
        validate_isolation(settings)


@pytest.mark.asyncio
async def test_recorder_requires_actual_ack_and_never_logs_sensitive_values():
    recorder = DeliveryRecorder(42)
    method = SendMessage(chat_id=42, text="Private text")
    async def fail(bot, method):
        raise RuntimeError("sensitive token and raw response")
    with pytest.raises(RuntimeError):
        await recorder(fail, None, method)
    report = json.dumps(recorder.events)
    assert "Private text" not in report and "sensitive" not in report
    assert recorder.metrics()["receipt_to_first_body_ack_ms"] is None
    assert recorder.metrics()["telegram_failures"] == 1
    async def ack(bot, method):
        return True
    with pytest.raises(RuntimeError):
        await recorder(ack, None, SendMessage(chat_id=43, text="wrong recipient"))
    await recorder(ack, None, method)
    assert recorder.metrics()["receipt_to_first_body_ack_ms"] is not None


@pytest.mark.asyncio
async def test_real_handler_offline_transport_contract_all_chunks_documents_and_no_action_approval(monkeypatch, tmp_path):
    """Offline only: actual Dispatcher, burst, SSE parser, rendering and source handler.

    This validates instrumentation; it does not claim real Telegram delivery.
    """
    from app.eval import telegram_delivery as module
    from bot.handlers import _burst
    chat_id, message_id = UUID(int=100), UUID(int=200)
    answer = "Точный ответ [1]. " * 230
    sources = [{"citation_index": 1, "filename": "manual.pdf", "page": 3, "text": "evidence"}]
    transport_calls = []
    sent = False
    async def response(request):
        nonlocal sent
        transport_calls.append((request.method, request.url.path))
        if request.url.path == "/__candidate_identity":
            return httpx.Response(200, json={"candidate_identity_sha256": "candidate", "source_sha256": "local", "loaded_source_sha256": "local",
                "budget_enforced": True, "budget": {"limit_usd": 10, "accounted_upper_usd": 0}})
        if request.url.path == "/chats":
            return httpx.Response(200, json={"chat_id": str(chat_id)})
        if request.method == "DELETE" and request.url.path.endswith("/messages"):
            return httpx.Response(204)
        if request.method == "POST" and request.url.path.endswith("/messages"):
            sent = True
            values = [{"type": "token", "delta": answer},
                      {"type": "message", "message_id": str(message_id), "answer_info": {"expandable": True}},
                      {"type": "done", "message_id": str(message_id)}]
            stream = "".join("data: " + json.dumps(value) + "\n\n" for value in values)
            stream += "event: sources\ndata: " + json.dumps(sources) + "\n\n"
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})
        if request.url.path.endswith("/sources"):
            return httpx.Response(200, json=sources)
        if request.url.path.endswith("/messages"):
            if not sent:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[{"role": "user", "content": "test"}, {"role": "assistant", "content": answer}])
        raise AssertionError("unexpected HTTP request")
    real_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(response)))
    monkeypatch.setattr(module, "source_identity", lambda _: "local")
    monkeypatch.setattr(_burst, "TEXT_DEBOUNCE_SECONDS", 0.01)
    class OfflineSession(BaseSession):
        def __init__(self):
            super().__init__()
            self._connector_init = {}
            self.counter = 0
        async def close(self):
            pass
        async def make_request(self, bot, method, timeout=None):
            if method.__api_method__ == "getMe":
                return User(id=12345, is_bot=True, first_name="Bot")
            if method.__api_method__ in {"sendMessage", "editMessageText"}:
                self.counter += 1
                return Message(message_id=getattr(method, "message_id", None) or self.counter,
                    date=datetime.now(UTC), chat=Chat(id=42, type="private"), text=method.text,
                    reply_markup=getattr(method, "reply_markup", None)).as_(bot)
            return True
        async def stream_content(self, *args, **kwargs):
            yield b""
    monkeypatch.setattr(module, "AiohttpSession", OfflineSession)
    settings = SimpleNamespace(bot_allowed_user_ids=[42], bot_allowed_chat_ids=[], bot_admin_ids=[42],
        bot_token=SecretStr("12345:offline-only-fake-token"), internal_token=SecretStr("internal"), admin_token=SecretStr("admin"))
    output = tmp_path / "delivery.json"
    report = await module.run_delivery(settings=settings, api_url="http://127.0.0.1:18001", question="test",
        scenario_id="offline", recipient_fingerprint=fingerprint(42), expected_identity="candidate",
        source_root=tmp_path, output=output)
    assert report["status"] == "delivered", report
    assert report["native_polling_measured"] is False
    assert report["native_callback_ack_measured"] is False
    assert report["history_reset"] == "isolated_candidate_before_receipt_timer"
    evidence = json.loads(output.with_suffix(".canonical.json").read_text(encoding="utf-8"))
    assert evidence["brief"]["answer"] == answer and evidence["brief"]["sources"] == sources
    assert report["canonical_answer_sha256"] == evidence["brief"]["answer_sha256"]
    assert report["receipt_to_last_delivery_ack_ms"] >= report["receipt_to_last_answer_ack_ms"]
    assert report["receipt_to_last_answer_ack_ms"] >= report["receipt_to_first_body_ack_ms"]
    answer_sends = [e for e in report["events"] if e["body"] and e["phase"] == "answer" and e["status"] == "ack"]
    assert len(answer_sends) >= 3
    assert any("src" in e["button_kinds"] and "exp" in e["button_kinds"] for e in answer_sends)
    assert any(e["phase"] == "documents" and e["status"] == "ack" for e in report["events"])
    assert not any("approvals" in path or path.endswith("/expand") for _, path in transport_calls)
    serialized = output.read_text(encoding="utf-8")
    assert answer not in serialized and "offline-only-fake-token" not in serialized
    with pytest.raises(ValueError, match="already exists"):
        await module.run_delivery(settings=settings, api_url="http://127.0.0.1:18001", question="test",
            scenario_id="offline", recipient_fingerprint=fingerprint(42), expected_identity="candidate",
            source_root=tmp_path, output=output)
