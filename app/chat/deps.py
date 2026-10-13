from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated
from weakref import WeakValueDictionary

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.chat.agent import ChatAgent
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_repo import PostgresChatRepository
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.deps.providers import OpenAIDep, SettingsDep


async def get_repository(request: Request, settings: SettingsDep) -> AsyncIterator[ChatRepository]:
    if settings.chat_repository == "json":
        yield JsonChatRepository(settings.chat_storage_dir)
        return

    if settings.chat_repository == "postgres":
        engine = getattr(request.app.state, "chat_engine", None)
        if engine is None:
            engine = create_async_engine(settings.chat_database_url, pool_pre_ping=True)
            request.app.state.chat_engine = engine
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            yield PostgresChatRepository(session)
        return

    raise ValueError(f"Unsupported CHAT_REPOSITORY={settings.chat_repository!r}")


ChatRepositoryDep = Annotated[ChatRepository, Depends(get_repository)]


def get_chat_service(request: Request, repo: ChatRepositoryDep, llm: OpenAIDep, settings: SettingsDep) -> ChatService:
    # Turn locks live on app.state: the service is rebuilt per request, but
    # concurrent sends for the same chat must share one lock (F22).
    turn_locks = getattr(request.app.state, "chat_turn_locks", None)
    if turn_locks is None:
        turn_locks = WeakValueDictionary()
        request.app.state.chat_turn_locks = turn_locks
    def track_rag_worker(task: asyncio.Task[None]) -> None:
        # The principal dependency may run after this factory; resolve its
        # lease when generation starts, after all route dependencies finish.
        lease = getattr(request.state, "request_budget_lease", None)
        if lease is not None:
            lease.track_worker(task)

    return ChatService(
        repository=repo, llm_client=llm, settings=settings,
        turn_locks=turn_locks, track_rag_worker=track_rag_worker,
        agent=ChatAgent(request.app.state.agent_graph, request.app.state.agent_checkpointer, track_rag_worker, settings)
        if getattr(request.app.state, "agent_graph", None) is not None else None,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]

