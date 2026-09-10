from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from app.routers.rag import router as rag_router
from app.schemas.rag import RAGHistoryMessage
from app.services.chm_reader import CHMReader
from app.services.rag import NO_ANSWER, RAGConfig, RAGService
from app.services.rag_baremetal import BareMetalRAGService, TextRecord, naive_chunks


ROOT = Path(__file__).resolve().parents[2]


def rag_config(tmp_path: Path, **overrides: Any) -> RAGConfig:
    values = {
        "data_dir": tmp_path,
        "qdrant_url": "http://qdrant:6333",
        "qdrant_api_key": "test-key",
        "collection_name": "rag_block_03",
        "baremetal_collection_name": "rag_block_03_baremetal",
        "llm_model": "test-chat",
        "embedding_model": "test-embedding",
        "embedding_dimensions": 3,
        "chunk_size": 32,
        "chunk_overlap": 4,
        "top_k": 3,
        "min_score": 0.35,
        "docstore_path": tmp_path / "ingestion_docstore.json",
        "openai_api_key": "test",
    }
    values.update(overrides)
    return RAGConfig(**values)


def test_corpus_contains_fifty_multiformat_documents_and_chm() -> None:
    corpus = ROOT / "data"
    supported = {".pdf", ".docx", ".html", ".htm", ".md", ".txt", ".chm"}
    files = [
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    ]

    assert len(files) >= 50
    assert sum(path.suffix.lower() == ".pdf" for path in files) >= 1
    assert sum(path.suffix.lower() in {".html", ".htm"} for path in files) >= 1
    assert sum(path.suffix.lower() == ".chm" for path in files) >= 1
    assert any("User Manual" in path.name for path in files)
    assert any("Пособие" in path.name for path in files)
    assert any(path.name == "Руководство по расчетам midas Civil 2020.pdf" for path in files)


def test_chm_reader_decodes_cp1251_and_uses_ascii_workaround(tmp_path: Path) -> None:
    chm_path = tmp_path / "Справка.chm"
    chm_path.write_bytes(b"fixture")
    opened_paths: list[str] = []

    class FakeArchive:
        def files(self, *, include_system: bool):
            assert include_system is False
            return [{"name": "/topic.htm"}, {"name": "/image.png"}]

        def read(self, name: str, *, max_size: int):
            assert name == "/topic.htm"
            assert max_size > 0
            return (
                '<html><head><meta charset="windows-1251"></head>'
                '<body><h1>Степени свободы</h1><script>ignore()</script></body></html>'
            ).encode("windows-1251")

    def factory(path: str):
        opened_paths.append(path)
        return FakeArchive()

    documents = CHMReader(archive_factory=factory).load_data(chm_path, {"language": "ru"})

    assert len(documents) == 1
    assert "Степени свободы" in documents[0].text
    assert "ignore" not in documents[0].text
    assert documents[0].metadata["source"] == "Справка.chm::topic.htm"
    assert documents[0].metadata["language"] == "ru"
    assert opened_paths and opened_paths[0].isascii()


def test_rag_config_requires_overlap_and_separate_collections(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        rag_config(tmp_path, chunk_size=8, chunk_overlap=8)
    with pytest.raises(ValueError, match="collections"):
        rag_config(tmp_path, baremetal_collection_name="rag_block_03")


class FakeRetriever:
    def __init__(self, scores: list[float]) -> None:
        self.questions: list[str] = []
        self.source_nodes = [
            NodeWithScore(
                node=TextNode(
                    text=f"fragment-{index}",
                    metadata={"source": f"manual-{index}.pdf", "page_number": index},
                ),
                score=score,
            )
            for index, score in enumerate(scores, start=1)
        ]

    def retrieve(self, question: str):
        self.questions.append(question)
        return self.source_nodes


class FakeLLM:
    def __init__(self, text: str = "Ответ по базе") -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.text)


def test_llamaindex_answer_contract_and_threshold(tmp_path: Path) -> None:
    llm = FakeLLM()
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = FakeRetriever([0.91, 0.82, 0.73])

    result = service.answer("Что такое степень свободы?")

    assert result["answer"].startswith("Ответ по базе")
    assert result["top_score"] == 0.91
    assert len(result["sources"]) == 3
    assert set(result["sources"][0]) == {"id", "file_name", "page", "score", "snippet", "confident"}
    assert result["sources"][0]["file_name"] == "manual-1.pdf"
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["confident"] is True
    assert "[1]" in result["answer"] and "[2]" in result["answer"]

    service._retriever = FakeRetriever([0.2, 0.1, 0.05])
    assert service.answer("Вопрос вне базы")["answer"] == NO_ANSWER
    assert len(llm.prompts) == 1


def test_llamaindex_reranks_after_top_k_and_guards_before_llm(tmp_path: Path) -> None:
    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def rerank(self, question: str, candidates, *, top_n: int):
            self.calls.append((question, len(candidates), top_n))
            return [candidates[2], candidates[1]][:top_n]

    llm = FakeLLM()
    reranker = FakeReranker()
    service = RAGService(
        rag_config(tmp_path, rerank_enabled=True, rerank_top_n=2, allow_external_data_transfer=True),
        llm=llm,
        reranker=reranker,
    )
    service._retriever = FakeRetriever([0.91, 0.82, 0.73])

    result = service.query("Каков срок возврата?")

    assert reranker.calls == [("Каков срок возврата?", 3, 2)]
    assert [source["file_name"] for source in result["sources"]] == ["manual-3.pdf", "manual-2.pdf"]
    assert result["top_score"] == 0.82
    assert len(llm.prompts) == 1


def test_follow_up_question_uses_history_for_deterministic_condense(tmp_path: Path) -> None:
    retriever = FakeRetriever([0.91, 0.82, 0.73])
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=FakeLLM())
    service._retriever = retriever

    service.answer(
        "А для электроники?",
        history=[
            {"role": "user", "content": "Каков срок возврата?"},
            RAGHistoryMessage(role="assistant", content="Срок зависит от категории товара."),
        ],
    )

    assert retriever.questions == [
        "Контекст предыдущего вопроса: Каков срок возврата?\nУточняющий вопрос: А для электроники?"
    ]


def test_standalone_question_and_condense_hook_override_default_rewrite(tmp_path: Path) -> None:
    retriever = FakeRetriever([0.91, 0.82, 0.73])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM(),
        condense_hook=lambda question, _history: f"rewritten: {question}",
    )
    service._retriever = retriever

    service.answer("А для электроники?", history=[{"role": "user", "content": "Каков срок возврата?"}])
    service.answer(
        "А для электроники?",
        history=[{"role": "user", "content": "Каков срок возврата?"}],
        standalone_question="срок возврата электроники",
    )

    assert retriever.questions == ["rewritten: А для электроники?", "срок возврата электроники"]


def test_rag_requires_explicit_permission_before_embedding_question(tmp_path: Path) -> None:
    llm = FakeLLM()
    service = RAGService(rag_config(tmp_path), llm=llm)
    service._retriever = FakeRetriever([0.91, 0.82, 0.73])

    with pytest.raises(RuntimeError, match="RAG_ALLOW_EXTERNAL_DATA_TRANSFER"):
        service.answer("Каков срок возврата?")

    assert llm.prompts == []


def test_llamaindex_build_runs_end_to_end_without_network(tmp_path: Path) -> None:
    for index in range(30):
        (tmp_path / f"document-{index}.txt").write_text(
            f"GTS NX document {index}: mesh and degrees of freedom.",
            encoding="utf-8",
        )
    config = rag_config(
        tmp_path,
        collection_name="llamaindex_test",
        baremetal_collection_name="baremetal_test",
        embedding_dimensions=8,
        chunk_size=128,
        chunk_overlap=16,
        min_score=0,
    )
    service = RAGService(
        config,
        client=QdrantClient(":memory:"),
        embed_model=MockEmbedding(embed_dim=8),
        llm=MockLLM(max_tokens=32),
    ).build()

    result = service.answer("mesh")

    assert len(result["sources"]) == config.top_k
    assert result["answer"]
    assert service.build() is service
    service.close()


def test_llamaindex_build_reuses_existing_collection_with_incremental_ingestion(tmp_path: Path) -> None:
    for index in range(30):
        (tmp_path / f"document-{index}.txt").write_text(
            f"GTS NX document {index}: mesh and degrees of freedom.",
            encoding="utf-8",
        )
    client = QdrantClient(":memory:")
    base_config = rag_config(
        tmp_path,
        collection_name="llamaindex_reuse_test",
        baremetal_collection_name="baremetal_reuse_test",
        embedding_dimensions=8,
        chunk_size=128,
        chunk_overlap=16,
        min_score=0,
    )
    RAGService(
        base_config,
        client=client,
        embed_model=MockEmbedding(embed_dim=8),
        llm=MockLLM(max_tokens=32),
    ).build()

    reused = RAGService(
        rag_config(
            tmp_path,
            collection_name="llamaindex_reuse_test",
            baremetal_collection_name="baremetal_reuse_test",
            embedding_dimensions=8,
            chunk_size=128,
            chunk_overlap=16,
            min_score=0,
            rebuild_on_start=False,
        ),
        client=client,
        embed_model=MockEmbedding(embed_dim=8),
        llm=MockLLM(max_tokens=32),
    ).build()

    assert len(reused.answer("mesh")["sources"]) == base_config.top_k
    reused.close()


def test_naive_chunks_has_overlap() -> None:
    assert naive_chunks("abcdefghij", chunk_size=6, overlap=2) == ["abcdef", "efghij", "ij"]


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeQdrant:
    def __init__(self, scores: list[float] | None = None, *, existing: bool = False) -> None:
        self.scores = scores or [0.9, 0.8, 0.7]
        self.existing = existing
        self.created: list[str] = []
        self.uploaded: list[Any] = []

    def collection_exists(self, _name: str) -> bool:
        return self.existing

    def create_collection(self, *, collection_name: str, vectors_config: Any) -> None:
        assert vectors_config.size == 3
        self.created.append(collection_name)
        self.existing = True

    def get_collection(self, _name: str) -> Any:
        vectors = SimpleNamespace(size=3, distance=Distance.COSINE)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def upload_points(self, *, collection_name: str, points: list[Any], batch_size: int, wait: bool) -> None:
        assert collection_name == "rag_block_03_baremetal"
        assert batch_size == 64 and wait is True
        self.uploaded = points

    def query_points(self, **_kwargs: Any):
        return SimpleNamespace(
            points=[
                SimpleNamespace(score=score, payload={"text": f"context-{index}", "source": f"source-{index}"})
                for index, score in enumerate(self.scores, start=1)
            ]
        )

    def close(self) -> None:
        pass


class TinyBareMetal(BareMetalRAGService):
    def _load_records(self):
        yield TextRecord("first document text", "first.txt")
        yield TextRecord("second document text", "second.txt")
        yield TextRecord("third document text", "third.txt")


def test_baremetal_build_uses_separate_flat_payload_collection(tmp_path: Path) -> None:
    client = FakeQdrant()
    embeddings = FakeEmbeddingService()
    service = TinyBareMetal(
        rag_config(tmp_path, chunk_size=64, chunk_overlap=8),
        client=client,
        embedding_service=embeddings,
    ).build()

    assert service._built is True
    assert client.created == ["rag_block_03_baremetal"]
    assert len(client.uploaded) == 3
    assert set(client.uploaded[0].payload) == {"text", "source", "chunk_index"}


def test_baremetal_build_reuses_existing_collection_without_ingestion(tmp_path: Path) -> None:
    client = FakeQdrant(existing=True)
    embeddings = FakeEmbeddingService()
    service = TinyBareMetal(
        rag_config(tmp_path, chunk_size=64, chunk_overlap=8, rebuild_on_start=False),
        client=client,
        embedding_service=embeddings,
    ).build()

    assert service._built is True
    assert client.created == []
    assert client.uploaded == []
    assert embeddings.calls == []


def test_baremetal_answer_calls_chat_completions_in_russian(tmp_path: Path) -> None:
    create_calls: list[dict[str, Any]] = []

    def create(**kwargs: Any):
        create_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Русский ответ"))])

    openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: None,
    )
    service = BareMetalRAGService(
        rag_config(tmp_path),
        client=FakeQdrant(),
        embedding_service=FakeEmbeddingService(),
        openai_client=openai_client,
    )
    service._built = True

    result = service.answer("Вопрос на русском")

    assert result["answer"] == "Русский ответ"
    assert len(result["sources"]) == 3
    assert create_calls[0]["model"] == "test-chat"
    assert "Отвечай только на русском" in create_calls[0]["messages"][0]["content"]


def test_baremetal_low_score_does_not_call_llm(tmp_path: Path) -> None:
    service = BareMetalRAGService(
        rag_config(tmp_path),
        client=FakeQdrant(scores=[0.2, 0.1, 0.05]),
        embedding_service=FakeEmbeddingService(),
        openai_client=None,
    )
    service._built = True

    assert service.answer("Вопрос вне базы")["answer"] == NO_ANSWER
    assert service._openai is None


def test_rag_endpoint_is_separate_and_returns_contract() -> None:
    class FakeRAG:
        def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"history": [RAGHistoryMessage(role="user", content="Каков срок возврата?")]}
            return {
                "answer": f"Ответ: {question}",
                "top_score": 0.9,
                "sources": [
                    {
                        "id": f"node-{index}",
                        "file_name": f"source-{index}.pdf",
                        "page": index + 1,
                        "score": 0.9 - index / 10,
                        "snippet": f"text-{index}",
                        "confident": True,
                    }
                    for index in range(3)
                ],
            }

    api = FastAPI()
    api.include_router(rag_router)
    api.state.rag_service = FakeRAG()

    response = TestClient(api).post(
        "/rag/query",
        json={
            "question": "Проверочный вопрос",
            "chat_id": "chat-1",
            "history": [{"role": "user", "content": "Каков срок возврата?"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ: Проверочный вопрос"
    assert len(response.json()["sources"]) == 3
    assert set(response.json()["sources"][0]) == {"id", "file_name", "page", "score", "snippet", "confident"}
    assert not any(route.path == "/chat" for route in api.routes)
