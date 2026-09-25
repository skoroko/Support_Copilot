from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document as WordDocument
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.chm_reader import CHMReader
from app.services.embeddings import EmbeddingConfig, EmbeddingService
from app.services.rag import NO_ANSWER, RAG_SYSTEM_PROMPT, RAGConfig


_POINT_NAMESPACE = uuid.UUID("7bf6de45-7de4-49af-9fc1-0647c5e64cb3")


@dataclass(frozen=True)
class TextRecord:
    text: str
    source: str


def naive_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split normalized text by character windows for the framework-free baseline."""

    normalized = " ".join(text.split())
    if not normalized:
        return []
    step = chunk_size - overlap
    return [normalized[start : start + chunk_size] for start in range(0, len(normalized), step)]


class BareMetalRAGService:
    """Explicit RAG flow: loaders -> chunks -> embeddings -> Qdrant -> Chat Completions."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        client: Any | None = None,
        embedding_service: Any | None = None,
        openai_client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self._embedding_service = embedding_service
        self._openai = openai_client
        self._built = False

    def build(self) -> "BareMetalRAGService":
        if self._built:
            return self
        if self.client is None:
            self.client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.request_timeout,
            )
        name = self.config.baremetal_collection_name
        collection_exists = self.client.collection_exists(name)
        if self.config.rebuild_on_start and collection_exists:
            self.client.delete_collection(name)
            collection_exists = False
        if not collection_exists:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.config.embedding_dimensions, distance=Distance.COSINE),
            )
        else:
            vectors = self.client.get_collection(name).config.params.vectors
            if isinstance(vectors, dict):
                raise ValueError(f"RAG collection {name!r} must use one unnamed dense vector")
            if int(vectors.size) != self.config.embedding_dimensions or vectors.distance != Distance.COSINE:
                raise ValueError(
                    f"RAG collection {name!r} must use Cosine/{self.config.embedding_dimensions} dimensions"
                )

        if not collection_exists:
            records = list(self._load_records())
            chunks: list[TextRecord] = []
            for record in records:
                chunks.extend(
                    TextRecord(text=chunk, source=record.source)
                    for chunk in naive_chunks(record.text, self.config.chunk_size, self.config.chunk_overlap)
                )
            if len(chunks) < self.config.top_k:
                raise ValueError("RAG corpus produced fewer chunks than RAG_TOP_K")

            embeddings = self._embeddings().embed_texts([chunk.text for chunk in chunks])
            points = [
                PointStruct(
                    id=str(uuid.uuid5(_POINT_NAMESPACE, f"{chunk.source}\n{index}\n{hashlib.sha256(chunk.text.encode()).hexdigest()}")),
                    vector=embedding,
                    payload={"text": chunk.text, "source": chunk.source, "chunk_index": index},
                )
                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            ]
            self.client.upload_points(collection_name=name, points=points, batch_size=64, wait=True)
        self._built = True
        return self

    def answer(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be blank")
        if not self._built:
            raise RuntimeError("RAG index is not built")
        query_vector = self._embeddings().embed_texts([question.strip()])[0]
        response = self.client.query_points(
            collection_name=self.config.baremetal_collection_name,
            query=query_vector,
            limit=self.config.top_k,
            with_payload=True,
        )
        sources = [
            {
                "text": str((point.payload or {}).get("text", "")),
                "source": str((point.payload or {}).get("source", "unknown")),
                "score": float(point.score),
            }
            for point in response.points
        ]
        top_score = sources[0]["score"] if sources else 0.0
        if top_score < self.config.min_score:
            answer = NO_ANSWER
        else:
            context = "\n\n".join(
                f"Источник: {item['source']}\nФрагмент: {item['text']}" for item in sources
            )
            completion = self._llm_client().chat.completions.create(
                model=self.config.llm_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question.strip()}"},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip() or NO_ANSWER
        return {"answer": answer, "top_score": top_score, "sources": sources}

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        if self._openai is not None:
            self._openai.close()

    def _embeddings(self) -> Any:
        if self._embedding_service is None:
            cache_path = Path(__file__).resolve().parents[2] / "var" / "rag_embeddings.sqlite3"
            self._embedding_service = EmbeddingService(
                EmbeddingConfig(
                    model=self.config.embedding_model,
                    dimensions=self.config.embedding_dimensions,
                    cache_path=cache_path,
                    request_timeout_seconds=self.config.request_timeout,
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                )
            )
        return self._embedding_service

    def _llm_client(self) -> Any:
        if self._openai is None:
            kwargs: dict[str, Any] = {
                "api_key": self.config.openai_api_key,
                "timeout": self.config.generation_timeout,
            }
            if self.config.openai_base_url:
                kwargs["base_url"] = self.config.openai_base_url
            self._openai = OpenAI(**kwargs)
        return self._openai

    def _load_records(self):
        supported = {".pdf", ".docx", ".md", ".txt", ".chm"}
        paths = [path for path in self.config.data_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported]
        if len(paths) < 10:
            raise ValueError(f"RAG corpus must contain at least 10 supported files; found {len(paths)}")
        for path in sorted(paths):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                for page_number, page in enumerate(PdfReader(path).pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        yield TextRecord(text=text, source=f"{path.name}#page={page_number}")
            elif suffix == ".docx":
                text = "\n".join(paragraph.text for paragraph in WordDocument(path).paragraphs)
                if text.strip():
                    yield TextRecord(text=text, source=path.name)
            elif suffix in {".md", ".txt"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    yield TextRecord(text=text, source=path.name)
            elif suffix == ".chm":
                for document in CHMReader().load_data(path):
                    yield TextRecord(text=document.text, source=str(document.metadata["source"]))
