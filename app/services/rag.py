from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.services.chm_reader import CHMReader


RAG_SYSTEM_PROMPT = """Ты — помощник технической поддержки MIDAS GTS NX.
Отвечай только на русском языке и только по предоставленному контексту.
Документы могут быть на русском или английском: переводи найденный ответ на русский.
Если контекст не содержит надёжного ответа, честно сообщи, что сведений в базе нет.
Не используй нерелевантные документы и не придумывай цены, SLA, функции или регламенты.
"""
NO_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."


@dataclass(frozen=True)
class RAGConfig:
    data_dir: Path
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    baremetal_collection_name: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 3
    min_score: float = 0.35
    rebuild_on_start: bool = True
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k < 3:
            raise ValueError("top_k must be at least 3")
        if self.collection_name == self.baremetal_collection_name:
            raise ValueError("LlamaIndex and bare-metal collections must be different")

    @classmethod
    def from_settings(cls, settings: Any) -> "RAGConfig":
        data_dir = Path(settings.rag_data_dir)
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parents[2] / data_dir
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return cls(
            data_dir=data_dir,
            qdrant_url=settings.qdrant_url or "http://localhost:6333",
            qdrant_api_key=api_key,
            collection_name=settings.rag_collection,
            baremetal_collection_name=settings.rag_baremetal_collection,
            llm_model=settings.rag_llm_model,
            embedding_model=settings.rag_embedding_model,
            embedding_dimensions=settings.rag_embedding_dimensions,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            rebuild_on_start=settings.rag_rebuild_on_start,
            openai_api_key=settings.openai_api_key.get_secret_value(),
            openai_base_url=settings.openai_base_url,
            request_timeout=settings.request_timeout,
        )


class RAGService:
    """LlamaIndex implementation of the Support Copilot RAG pipeline."""

    def __init__(self, config: RAGConfig, *, client: Any | None = None, embed_model: Any | None = None, llm: Any | None = None) -> None:
        self.config = config
        self.client = client
        self._embed_model = embed_model
        self._llm = llm
        self._index: Any | None = None
        self._query_engine: Any | None = None

    def build(self) -> "RAGService":
        if self._query_engine is not None:
            return self
        self._validate_corpus()
        if self.client is None:
            self.client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key, timeout=self.config.request_timeout)
        collection_exists = self.client.collection_exists(self.config.collection_name)
        if self.config.rebuild_on_start and collection_exists:
            self.client.delete_collection(self.config.collection_name)
            collection_exists = False

        vector_store = QdrantVectorStore(client=self.client, collection_name=self.config.collection_name)
        if collection_exists:
            self._validate_collection(self.config.collection_name)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embedding_model(),
            )
        else:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=self.config.embedding_dimensions, distance=Distance.COSINE),
            )
            documents = SimpleDirectoryReader(
                input_dir=str(self.config.data_dir),
                recursive=True,
                required_exts=[".pdf", ".docx", ".md", ".txt", ".chm"],
                file_extractor={".chm": CHMReader()},
                raise_on_error=True,
            ).load_data()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[SentenceSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)],
                embed_model=self._embedding_model(),
                show_progress=False,
            )
        self._query_engine = self._index.as_query_engine(
            llm=self._language_model(), similarity_top_k=self.config.top_k, response_mode="compact"
        )
        return self

    def answer(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be blank")
        if self._query_engine is None:
            raise RuntimeError("RAG index is not built")
        response = self._query_engine.query(question.strip())
        sources = [self._source_from_node(item) for item in response.source_nodes[: self.config.top_k]]
        top_score = sources[0]["score"] if sources else 0.0
        answer = str(response).strip() or NO_ANSWER
        if top_score < self.config.min_score:
            answer = NO_ANSWER
        return {"answer": answer, "top_score": top_score, "sources": sources}

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _validate_corpus(self) -> None:
        if not self.config.data_dir.is_dir():
            raise FileNotFoundError(f"RAG_DATA_DIR does not exist: {self.config.data_dir}")
        supported = {".pdf", ".docx", ".md", ".txt", ".chm"}
        files = [path for path in self.config.data_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported]
        if len(files) < 10:
            raise ValueError(f"RAG corpus must contain at least 10 supported files; found {len(files)}")

    def _validate_collection(self, name: str) -> None:
        vectors = self.client.get_collection(name).config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError(f"RAG collection {name!r} must use one unnamed dense vector")
        if int(vectors.size) != self.config.embedding_dimensions or vectors.distance != Distance.COSINE:
            raise ValueError(
                f"RAG collection {name!r} must use Cosine/{self.config.embedding_dimensions} dimensions"
            )

    def _embedding_model(self) -> Any:
        if self._embed_model is None:
            self._embed_model = OpenAIEmbedding(
                model=self.config.embedding_model, dimensions=self.config.embedding_dimensions,
                api_key=self.config.openai_api_key, api_base=self.config.openai_base_url,
                timeout=self.config.request_timeout,
            )
        return self._embed_model

    def _language_model(self) -> Any:
        if self._llm is None:
            self._llm = LlamaOpenAI(
                model=self.config.llm_model, api_key=self.config.openai_api_key,
                api_base=self.config.openai_base_url, timeout=self.config.request_timeout,
                temperature=0, system_prompt=RAG_SYSTEM_PROMPT,
            )
        return self._llm

    @staticmethod
    def _source_from_node(source_node: Any) -> dict[str, Any]:
        node = source_node.node
        metadata = node.metadata or {}
        source = metadata.get("source") or metadata.get("file_name") or metadata.get("file_path") or "unknown"
        return {
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "source": str(source),
            "score": float(source_node.score or 0.0),
        }


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Support Copilot RAG query")
    parser.add_argument("question", help="Question about the configured RAG corpus")
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Confirm that manual fragments may be sent to the configured embedding and LLM provider",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if not args.allow_external_data_transfer:
        raise SystemExit(
            "Refusing to send manual fragments externally. Re-run with "
            "--allow-external-data-transfer only after explicit authorization."
        )

    from app.core.config import get_settings

    service = RAGService(RAGConfig.from_settings(get_settings())).build()
    try:
        print(json.dumps(service.answer(args.question), ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
