from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from app.chat.domain import ChatMessage
from app.chat.feedback import MessageFeedback
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_models import Base
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
    item = await repository.enqueue_broadcast("Maintenance", "telegram")

    pending = await repository.list_pending_broadcasts()
    updated = await repository.update_broadcast_status(item.id, "sent")

    assert [queued.id for queued in pending] == [item.id]
    assert updated is not None and updated.status == "sent"
    assert await repository.list_pending_broadcasts() == []
