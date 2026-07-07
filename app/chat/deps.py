from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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


def get_chat_service(repo: ChatRepositoryDep, llm: OpenAIDep, settings: SettingsDep) -> ChatService:
    return ChatService(repository=repo, llm_client=llm, settings=settings)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]

