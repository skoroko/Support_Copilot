from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.llm import LLMService
from app.services.vector_store import VectorStore


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_openai(request: Request) -> Any:
    return request.app.state.openai


OpenAIDep = Annotated[Any, Depends(get_openai)]


def get_cache(request: Request) -> Any:
    return request.app.state.cache


CacheDep = Annotated[Any, Depends(get_cache)]


def get_vector_store(request: Request) -> VectorStore:
    store = request.app.state.vector_store
    if store is None:
        raise RuntimeError("Qdrant is disabled: configure QDRANT_URL")
    return store


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


def get_llm_service(openai: OpenAIDep, cache: CacheDep, settings: SettingsDep) -> LLMService:
    return LLMService(openai, cache, settings)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
