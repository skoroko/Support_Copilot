"""Embedding requests inherit the turn deadline without mutating shared clients."""
from __future__ import annotations

from threading import Lock
from typing import Any

import httpx
from llama_index.embeddings.openai import OpenAIEmbedding as BaseOpenAIEmbedding
from llama_index.embeddings.openai.base import aget_embedding, get_embedding
from pydantic import PrivateAttr

from app.services.turn_budget import remaining_seconds

QUERY_TIMEOUT_SECONDS = 15.0


class OpenAIEmbedding(BaseOpenAIEmbedding):
    _client_lock: Any = PrivateAttr(default_factory=Lock)
    _owns_http_client: bool = PrivateAttr(default=False)

    def close(self) -> None:
        # Caller-supplied transports retain their caller's lifecycle. The async
        # adapter keeps its existing SDK lifecycle and is not used by retrieval.
        with self._client_lock:
            if self._owns_http_client and self._http_client is not None:
                self._http_client.close()

    def _request_timeout(self, limit: float) -> httpx.Timeout:
        residual = remaining_seconds(limit)
        return httpx.Timeout(residual, connect=min(10.0, residual), pool=min(5.0, residual))

    # Interactive recall should not inherit the larger batch-ingest timeout.
    def _get_query_embedding(self, query: str):
        return get_embedding(self._get_client(), query, engine=self._query_engine,
            **{**self.additional_kwargs, "timeout": self._request_timeout(min(self.timeout, QUERY_TIMEOUT_SECONDS))})

    async def _aget_query_embedding(self, query: str):
        return await aget_embedding(self._get_aclient(), query, engine=self._query_engine,
            **{**self.additional_kwargs, "timeout": remaining_seconds(min(self.timeout, QUERY_TIMEOUT_SECONDS))})

    def _get_client(self):
        timeout = self._request_timeout(self.timeout)
        with self._client_lock:
            if self._http_client is None:
                self._http_client = httpx.Client(follow_redirects=True, limits=httpx.Limits(
                    max_connections=1000, max_keepalive_connections=100, keepalive_expiry=60.0,
                ))
                self._owns_http_client = True
            client = super()._get_client()
        return client.with_options(timeout=timeout, max_retries=0)

    def _get_aclient(self):
        timeout = remaining_seconds(self.timeout)
        return super()._get_aclient().with_options(timeout=timeout, max_retries=0)
