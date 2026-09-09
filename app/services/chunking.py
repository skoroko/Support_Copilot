from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode
from pydantic import PrivateAttr


ChunkingStrategy = Literal["fixed", "recursive", "semantic"]
_RUSSIAN_SENTENCE_RE = re.compile(r".+?(?:[.!?…]+(?:\s+|$)|$)", re.DOTALL)


def _nonempty_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """Discard parser artefacts that contain no searchable text."""

    searchable: list[BaseNode] = []
    for node in nodes:
        text = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if text and any(character.isalnum() for character in text):
            searchable.append(node)
    return searchable


def russian_sentence_tokenizer(text: str) -> list[str]:
    """Split Russian or English prose while preserving sentence punctuation and spacing."""

    return [match.group(0) for match in _RUSSIAN_SENTENCE_RE.finditer(text) if match.group(0).strip()]


class EmbeddingServiceAdapter(BaseEmbedding):
    """Expose the project's cached EmbeddingService through the LlamaIndex contract."""

    _service: Any = PrivateAttr()

    def __init__(self, service: Any, *, model_name: str, embed_batch_size: int = 100) -> None:
        super().__init__(model_name=model_name, embed_batch_size=embed_batch_size)
        self._service = service

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._service.embed_texts([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._service.embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._service.embed_texts(texts)


def fixed_size(
    documents: Sequence[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[BaseNode]:
    """Token-count baseline that intentionally ignores sentence boundaries."""

    splitter = TokenTextSplitter.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return _nonempty_nodes(splitter.get_nodes_from_documents(list(documents), show_progress=False))


def recursive(
    documents: Sequence[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[BaseNode]:
    """Paragraph-first splitter with sentence-aware fallback for Russian prose."""

    splitter = SentenceSplitter.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=russian_sentence_tokenizer,
    )
    return _nonempty_nodes(splitter.get_nodes_from_documents(list(documents), show_progress=False))


def semantic(
    documents: Sequence[Document],
    *,
    embed_model: BaseEmbedding,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> list[BaseNode]:
    """Create boundaries where adjacent sentence groups become semantically dissimilar."""

    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        sentence_splitter=russian_sentence_tokenizer,
    )
    return _nonempty_nodes(splitter.get_nodes_from_documents(list(documents), show_progress=False))


def chunk_documents(
    strategy: ChunkingStrategy,
    documents: Sequence[Document],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_model: BaseEmbedding | None = None,
) -> list[BaseNode]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if strategy == "fixed":
        return fixed_size(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "recursive":
        return recursive(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "semantic":
        if embed_model is None:
            raise ValueError("semantic chunking requires embed_model")
        return semantic(documents, embed_model=embed_model)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
