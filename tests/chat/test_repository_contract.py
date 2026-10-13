from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import update

from app.chat.domain import ChatMessage
from app.chat.feedback import MessageFeedback
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_models import Base, BroadcastQueueRow
from app.chat.repositories.pg_repo import PostgresChatRepository
from app.chat.repository import ChatRepository


@pytest_asyncio.fixture(params=["json", "postgres"])
async def repository(request, tmp_path) -> AsyncIterator[ChatRepository]:
    if request.param == "json":
        yield JsonChatRepository(tmp_path)
        return

    database_url = os.getenv("CHAT_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set CHAT_TEST_DATABASE_URL to run Postgres repository contract tests")

    try:
        import asyncpg  # noqa: F401
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    except ImportError as exc:
        pytest.skip(f"Postgres test dependency is not installed: {exc}")

    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield PostgresChatRepository(session)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_chat_and_read_back(repository: ChatRepository) -> None:
    chat = await repository.create_chat("telegram-123", "telegram", "system")

    loaded = await repository.get_chat(chat.id)

    assert loaded == chat


@pytest.mark.asyncio
async def test_append_message_and_list_messages_are_chronological(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "cli")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = ChatMessage(chat_id=chat.id, role="user", content="first", created_at=started)
    second = ChatMessage(chat_id=chat.id, role="assistant", content="second", created_at=started + timedelta(seconds=1))

    await repository.append_message(chat.id, first)
    await repository.append_message(chat.id, second)

    messages = await repository.list_messages(chat.id)

    assert [message.content for message in messages] == ["first", "second"]


@pytest.mark.asyncio
async def test_independent_json_repositories_serialize_concurrent_appends(tmp_path) -> None:
    first = JsonChatRepository(tmp_path)
    second = JsonChatRepository(tmp_path)
    chat = await first.create_chat("owner", "cli")

    await asyncio.gather(
        *(
            (first if index % 2 == 0 else second).append_message(
                chat.id,
                ChatMessage(chat_id=chat.id, role="user", content=f"message-{index}"),
            )
            for index in range(40)
        )
    )

    messages = await first.list_messages(chat.id, limit=100)
    assert len(messages) == 40
    assert {message.content for message in messages} == {f"message-{index}" for index in range(40)}


@pytest.mark.asyncio
async def test_message_media_refs_round_trip(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "telegram")
    media_refs = {
        "mime": "image/png",
        "size": 3,
        "filename": "screen.png",
        "part": {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
    }
    await repository.append_message(
        chat.id,
        ChatMessage(chat_id=chat.id, role="user", content="image", media_refs=media_refs),
    )

    messages = await repository.list_messages(chat.id)

    assert messages[0].media_refs == media_refs


@pytest.mark.asyncio
async def test_message_rag_sources_round_trip(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "telegram")
    sources = [{"id": "manual-7", "file_name": "manual.pdf", "page": 7, "score": 0.91}]
    await repository.append_message(
        chat.id,
        ChatMessage(chat_id=chat.id, role="assistant", content="Ответ [1].", rag_sources=sources),
    )

    messages = await repository.list_messages(chat.id)

    assert messages[0].rag_sources == sources


@pytest.mark.asyncio
async def test_list_messages_limit_returns_last_n(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "web")
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(3):
        await repository.append_message(
            chat.id,
            ChatMessage(
                chat_id=chat.id,
                role="user",
                content=f"message-{index}",
                created_at=started + timedelta(seconds=index),
            ),
        )

    messages = await repository.list_messages(chat.id, limit=2)

    assert [message.content for message in messages] == ["message-1", "message-2"]


@pytest.mark.asyncio
async def test_soft_delete_hides_old_messages_but_keeps_new_ones(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "cli")
    await repository.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="old"))

    await repository.soft_delete_messages(chat.id)

    assert await repository.list_messages(chat.id) == []

    await repository.append_message(chat.id, ChatMessage(chat_id=chat.id, role="user", content="new"))

    messages = await repository.list_messages(chat.id)
    assert [message.content for message in messages] == ["new"]


@pytest.mark.asyncio
async def test_get_unknown_chat_returns_none(repository: ChatRepository) -> None:
    assert await repository.get_chat(uuid4()) is None


@pytest.mark.asyncio
async def test_feedback_is_unique_per_owner_and_message(repository: ChatRepository) -> None:
    chat = await repository.create_chat("owner", "telegram")
    message = ChatMessage(chat_id=chat.id, role="assistant", content="answer", latency_ms=25.0)
    await repository.append_message(chat.id, message)

    await repository.save_feedback(MessageFeedback(message_id=message.id, owner_external_id="owner", value="up"))
    await repository.save_feedback(MessageFeedback(message_id=message.id, owner_external_id="owner", value="down"))

    stats = await repository.get_admin_stats()
    assert stats.total_messages == 1
    assert stats.active_users == 1
    assert stats.feedback_up_ratio == 0.0
    assert stats.avg_latency_ms == 25.0


@pytest.mark.asyncio
async def test_broadcast_queue_lifecycle(repository: ChatRepository) -> None:
    await repository.create_chat("42", "telegram")
    item = await repository.enqueue_broadcast("Maintenance", "telegram")

    pending = await repository.list_pending_broadcasts()
    assert [queued.id for queued in pending] == [item.id]
    claimed = await repository.claim_broadcasts()
    assert len(claimed) == 1
    lease = claimed[0].lease_id
    assert lease is not None
    assert [recipient.chat_id for recipient in claimed[0].recipients] == [42]
    assert await repository.claim_broadcasts() == []
    assert await repository.list_pending_broadcasts() == []

    attempt = uuid4()
    assert await repository.finish_broadcast(item.id, lease) is None
    assert await repository.update_broadcast_recipient(item.id, uuid4(), 42, attempt, "in_flight") is None
    started = await repository.update_broadcast_recipient(item.id, lease, 42, attempt, "in_flight")
    assert started is not None and started.recipients[0].status == "in_flight"
    assert await repository.update_broadcast_recipient(item.id, lease, 42, uuid4(), "sent") is None
    delivered = await repository.update_broadcast_recipient(item.id, lease, 42, attempt, "sent")
    assert delivered is not None and delivered.recipients[0].status == "sent"
    finished = await repository.finish_broadcast(item.id, lease)
    assert finished is not None and finished.status == "sent"
    assert await repository.claim_broadcasts() == []


@pytest.mark.asyncio
async def test_broadcast_reclaim_preserves_progress_and_quarantines_uncertain_delivery(
    repository: ChatRepository,
) -> None:
    for owner in ("41", "42", "43:7", "43:8"):
        await repository.create_chat(owner, "telegram")
    item = await repository.enqueue_broadcast("Maintenance", "telegram")
    first = (await repository.claim_broadcasts())[0]
    assert [recipient.chat_id for recipient in first.recipients] == [41, 42, 43]
    assert first.lease_id is not None

    sent_attempt = uuid4()
    uncertain_attempt = uuid4()
    await repository.update_broadcast_recipient(item.id, first.lease_id, 41, sent_attempt, "in_flight")
    await repository.update_broadcast_recipient(item.id, first.lease_id, 41, sent_attempt, "sent")
    await repository.update_broadcast_recipient(item.id, first.lease_id, 42, uncertain_attempt, "in_flight")

    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    if isinstance(repository, JsonChatRepository):
        path = repository._broadcast_file()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["lease_expires_at"] = expired.isoformat()
        await repository._write_jsonl(path, rows)
    else:
        await repository.session.execute(
            update(BroadcastQueueRow)
            .where(BroadcastQueueRow.id == item.id)
            .values(lease_expires_at=expired)
        )
        await repository.session.commit()

    reclaimed = (await repository.claim_broadcasts())[0]
    assert reclaimed.lease_id != first.lease_id
    assert [recipient.status for recipient in reclaimed.recipients] == ["sent", "unknown", "pending"]
    assert reclaimed.review_required is True
    assert [review.id for review in await repository.list_review_broadcasts()] == [item.id]
    assert (await repository.get_broadcast(item.id)).recipients[1].status == "unknown"
    assert await repository.finish_broadcast(item.id, first.lease_id) is None
    assert await repository.update_broadcast_recipient(
        item.id, first.lease_id, 42, uncertain_attempt, "sent",
    ) is None
    assert reclaimed.lease_id is not None
    final_attempt = uuid4()
    await repository.update_broadcast_recipient(item.id, reclaimed.lease_id, 43, final_attempt, "in_flight")
    await repository.update_broadcast_recipient(item.id, reclaimed.lease_id, 43, final_attempt, "sent")
    finished = await repository.finish_broadcast(item.id, reclaimed.lease_id)
    assert finished is not None and finished.status == "failed" and finished.review_required


@pytest.mark.asyncio
async def test_legacy_json_processing_broadcast_requires_manual_review(tmp_path) -> None:
    repository = JsonChatRepository(tmp_path)
    item = await repository.enqueue_broadcast("Maintenance", "telegram")
    path = repository._broadcast_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["status"] = "processing"
    await repository._write_jsonl(path, rows)

    assert await repository.claim_broadcasts() == []
    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][0]
    assert stored["id"] == str(item.id)
    assert stored["status"] == "failed"
    assert stored["review_required"] is True
    assert [review.id for review in await repository.list_review_broadcasts()] == [item.id]


@pytest.mark.asyncio
async def test_broadcast_review_listing_is_bounded_and_paginated(repository: ChatRepository) -> None:
    await repository.create_chat("unparseable-owner", "telegram")
    first = await repository.enqueue_broadcast("First", "telegram")
    second = await repository.enqueue_broadcast("Second", "telegram")
    claims = await repository.claim_broadcasts(limit=2)
    assert [claim.id for claim in claims] == [first.id, second.id]
    for claim in claims:
        assert claim.lease_id is not None
        assert claim.review_required is True
        finished = await repository.finish_broadcast(claim.id, claim.lease_id)
        assert finished is not None and finished.status == "failed"

    assert [item.id for item in await repository.list_review_broadcasts(limit=1)] == [first.id]
    assert [item.id for item in await repository.list_review_broadcasts(limit=1, offset=1)] == [second.id]
    assert await repository.list_review_broadcasts(limit=1, offset=2) == []
    assert (await repository.get_broadcast(second.id)).invalid_recipient_count == 1
    assert await repository.get_broadcast(uuid4()) is None


@pytest.mark.asyncio
async def test_json_broadcast_claim_does_not_silently_skip_corrupt_chat(tmp_path) -> None:
    repository = JsonChatRepository(tmp_path)
    chat = await repository.create_chat("42", "telegram")
    item = await repository.enqueue_broadcast("Maintenance", "telegram")
    repository._chat_file(chat.id).write_text("not valid JSON", encoding="utf-8")

    with pytest.raises(ValueError):
        await repository.claim_broadcasts()
    assert [queued.id for queued in await repository.list_pending_broadcasts()] == [item.id]
