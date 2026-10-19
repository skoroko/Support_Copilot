"""Embedding requests inherit the turn deadline without mutating shared clients."""
from __future__ import annotations

from llama_index.embeddings.openai import OpenAIEmbedding as BaseOpenAIEmbedding
from llama_index.embeddings.openai.base import aget_embedding, get_embedding

from app.services.turn_budget import remaining_seconds

QUERY_TIMEOUT_SECONDS = 15.0


class OpenAIEmbedding(BaseOpenAIEmbedding):
    # Interactive recall should not inherit the larger batch-ingest timeout.
    def _get_query_embedding(self, query: str):
        return get_embedding(self._get_client(), query, engine=self._query_engine,
            **{**self.additional_kwargs, "timeout": remaining_seconds(min(self.timeout, QUERY_TIMEOUT_SECONDS))})

    async def _aget_query_embedding(self, query: str):
        return await aget_embedding(self._get_aclient(), query, engine=self._query_engine,
            **{**self.additional_kwargs, "timeout": remaining_seconds(min(self.timeout, QUERY_TIMEOUT_SECONDS))})

    def _get_client(self):
        timeout = remaining_seconds(self.timeout)
        return super()._get_client().with_options(timeout=timeout, max_retries=0)

    def _get_aclient(self):
        timeout = remaining_seconds(self.timeout)
        return super()._get_aclient().with_options(timeout=timeout, max_retries=0)
