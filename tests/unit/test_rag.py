from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from app.routers.rag import router as rag_router
from app.schemas.rag import RAGHistoryMessage
from app.services.chm_reader import CHMReader
from app.services.rag import NO_ANSWER, RAGConfig, RAGService, RetrievalPlan, build_llm_condense_hook
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


def test_rag_config_reads_ingest_on_start_setting() -> None:
    config = RAGConfig.from_settings(SimpleNamespace(rag_ingest_on_start=False))

    assert config.ingest_on_start is False


def test_rag_config_reads_lexical_retrieval_setting() -> None:
    config = RAGConfig.from_settings(SimpleNamespace(rag_lexical_retrieval_enabled=False))

    assert config.lexical_retrieval_enabled is False


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
    llm = FakeLLM("Ответ по базе [1] и [2].")
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = FakeRetriever([0.91, 0.82, 0.73])

    result = service.answer("Что такое степень свободы?")

    assert result["answer"].startswith("Ответ по базе")
    assert result["top_score"] == 0.91
    assert len(result["sources"]) == 2
    assert set(result["sources"][0]) == {
        "id",
        "file_name",
        "page",
        "printed_page",
        "score",
        "snippet",
        "confident",
    }
    assert result["sources"][0]["file_name"] == "manual-1.pdf"
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["confident"] is True
    assert "[1]" in result["answer"] and "[2]" in result["answer"]

    service._retriever = FakeRetriever([0.2, 0.1, 0.05])
    assert service.answer("Вопрос вне базы")["answer"] == NO_ANSWER
    assert len(llm.prompts) == 1


def test_evaluate_inputs_returns_complete_contexts_without_second_retrieval(
    tmp_path: Path,
) -> None:
    full_text = "Полный проверяемый контекст. " * 80
    source = NodeWithScore(
        node=TextNode(
            id_="full-context",
            text=full_text,
            metadata={"source": "manual.pdf", "page_number": 7},
        ),
        score=0.91,
    )

    class SingleCallRetriever:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, _question: str) -> list[NodeWithScore]:
            self.calls += 1
            return [source]

    retriever = SingleCallRetriever()
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    result = service.evaluate_inputs("Что сказано в руководстве?")

    assert retriever.calls == 1
    assert result["retrieved_contexts"] == [" ".join(full_text.split())]
    assert len(result["retrieved_contexts"][0]) > len(result["sources"][0]["snippet"])
    assert "retrieved_contexts" not in service.answer("Что сказано в руководстве?")


def test_rag_does_not_append_all_retrieval_candidates_as_citations() -> None:
    answer = "Ответ подтверждён источником [7]."

    assert RAGService._ensure_citations(answer, [{"id": "one"}] * 10) == answer


def test_rag_removes_citations_outside_final_source_list() -> None:
    answer = "Подтверждено [1, 7], дополнение [0] и [2]."

    assert RAGService._ensure_citations(answer, [{"id": "one"}, {"id": "two"}]) == (
        "Подтверждено [1], дополнение  и [2]."
    )


def test_rag_returns_only_cited_sources_and_renumbers_them() -> None:
    sources = [{"id": str(index)} for index in range(1, 5)]

    answer, cited_sources = RAGService._ground_answer("Сначала [2], затем [4, 2], ошибка [9].", sources)

    assert answer == "Сначала [1], затем [2, 1], ошибка ."
    assert cited_sources == [{"id": "2"}, {"id": "4"}]


def test_rag_delivers_uncited_answer_without_sources() -> None:
    """A missing citation contract no longer masks a usable answer as a KB miss."""

    answer, cited_sources = RAGService._ground_answer("Неподтверждённый ответ.", [{"id": "1"}])

    assert answer == "Неподтверждённый ответ."
    assert cited_sources == []


def test_rag_normalizes_cjk_citation_brackets() -> None:
    sources = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

    answer, cited_sources = RAGService._ground_answer("Модуль 【3】 и нагрузки 【1】.", sources)

    assert answer == "Модуль [2] и нагрузки [1]."
    assert cited_sources == [{"id": "1"}, {"id": "3"}]


def test_rag_normalizes_annotated_cjk_citation() -> None:
    answer, cited_sources = RAGService._ground_answer(
        "Метод описан в мануале 【1†source】.", [{"id": "1"}]
    )

    assert answer == "Метод описан в мануале [1]."
    assert cited_sources == [{"id": "1"}]


def test_rag_normalizes_footnote_citation() -> None:
    answer, cited_sources = RAGService._ground_answer("Метод описан в мануале[^1].", [{"id": "1"}])

    assert answer == "Метод описан в мануале[1]."
    assert cited_sources == [{"id": "1"}]


def test_rag_normalizes_source_word_citations() -> None:
    sources = [{"id": "1"}, {"id": "2"}]

    answer, cited_sources = RAGService._ground_answer("Согласно источнику 2.", sources)

    assert answer == "Согласно источнику [1]."
    assert cited_sources == [{"id": "2"}]


def test_rag_normalizes_multi_number_source_word_citations() -> None:
    sources = [{"id": "1"}, {"id": "2"}]

    answer, cited_sources = RAGService._ground_answer("См. источники 1 и 2.", sources)

    assert answer == "См. источники [1, 2]."
    assert cited_sources == sources


def test_rag_merges_mixed_citation_formats_into_one_numbering() -> None:
    sources = [{"id": "1"}, {"id": "2"}, {"id": "3"}]

    answer, cited_sources = RAGService._ground_answer("Шаг [1], детали 【3】.", sources)

    assert answer == "Шаг [1], детали [2]."
    assert cited_sources == [{"id": "1"}, {"id": "3"}]


def test_rag_converts_midline_paren_citation_without_canonical_markers() -> None:
    sources = [{"id": "1"}, {"id": "2"}]

    answer, cited_sources = RAGService._ground_answer("Настройка описана в разделе (2).", sources)

    assert answer == "Настройка описана в разделе [1]."
    assert cited_sources == [{"id": "2"}]


def test_rag_keeps_paren_numbers_when_canonical_citations_exist() -> None:
    sources = [{"id": "1"}, {"id": "2"}]

    answer, cited_sources = RAGService._ground_answer(
        "Ответ [1]. Шаги: (1) откройте файл, (2) сохраните.", sources
    )

    assert answer == "Ответ [1]. Шаги: (1) откройте файл, (2) сохраните."
    assert cited_sources == [{"id": "1"}]


def test_rag_keeps_line_start_paren_enumeration_as_prose() -> None:
    answer, cited_sources = RAGService._ground_answer(
        "Шаги:\n(1) Откройте файл\n(2) Сохраните.",
        [{"id": "1"}],
    )

    assert answer == "Шаги:\n(1) Откройте файл\n(2) Сохраните."
    assert cited_sources == []


def test_rag_strips_dangling_citations_but_delivers_answer() -> None:
    answer, cited_sources = RAGService._ground_answer("Подтверждено [9].", [{"id": "1"}])

    assert answer == "Подтверждено ."
    assert cited_sources == []


def test_rag_refuses_answer_that_is_only_dangling_citations() -> None:
    answer, cited_sources = RAGService._ground_answer("[9] [8]", [{"id": "1"}])

    assert answer == NO_ANSWER
    assert cited_sources == []


def test_rag_uncited_llm_answer_is_delivered_without_sources(tmp_path: Path) -> None:
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Шесть степеней свободы."),
    )
    service._retriever = FakeRetriever([0.91, 0.82])

    result = service.answer("Сколько степеней свободы?")

    assert result["answer"] == "Шесть степеней свободы."
    assert result["sources"] == []


@pytest.mark.parametrize(
    "answer",
    [
        "В предоставленном контексте сведений о стоимости нет. [1][2]",
        "Информация о ценах в документах не указана. [1]",
        "В базе знаний отсутствуют данные о стоимости лицензий. [2]",
    ],
)
def test_rag_normalizes_paraphrased_abstention_and_removes_sources(answer: str) -> None:
    grounded, cited_sources = RAGService._ground_answer(
        answer,
        [{"id": "one"}, {"id": "two"}],
    )

    assert grounded == NO_ANSWER
    assert cited_sources == []


def test_rag_keeps_factual_negative_statement_with_citation() -> None:
    answer = "Твердотельные элементы не содержат вращательных степеней свободы. [1]"

    grounded, cited_sources = RAGService._ground_answer(answer, [{"id": "solid"}])

    assert grounded == answer
    assert cited_sources == [{"id": "solid"}]


def test_overview_question_keeps_full_context_and_prefers_diverse_files(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_=f"boundary-{index}",
                text=text,
                metadata={"source": file_name},
            ),
            score=score,
        )
        for index, (file_name, text, score) in enumerate(
            [
                ("manual-shared.pdf", "Фрагмент о граничных условиях в опорах.", 0.95),
                ("manual-shared.pdf", "Фрагмент о граничных условиях в опорах", 0.94),
                ("manual-3.pdf", "Связи в узлах и закрепления конечной сетки.", 0.93),
                ("manual-4.pdf", "Типы опорных реакций и перемещений.", 0.92),
                ("manual-5.pdf", "Схемы приложения внешних нагрузок.", 0.91),
                ("manual-6.pdf", "Особенности симметрии расчётной области.", 0.90),
            ],
            start=1,
        )
    ]
    llm = FakeLLM("Связный обзор [1][2][3][4][5].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Расскажи про граничные условия")

    assert len(result["sources"]) == 5
    assert len({source["file_name"] for source in result["sources"]}) == 5
    assert "Это широкий обзорный вопрос" in llm.prompts[0]
    assert "Особенности симметрии расчётной области." in llm.prompts[0]
    assert llm.prompts[0].count("Фрагмент о граничных условиях в опорах") == 1


def test_survey_opener_with_focused_action_keeps_full_context(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                text=text,
                metadata={"source": "manual-seepage.pdf"},
            ),
            score=score,
        )
        for text, score in [
            ("Фильтрационный расчёт задаётся отдельным типом анализа.", 0.95),
            ("Границы сетки фильтрации назначаются через набор узлов.", 0.94),
            ("Коэффициенты проницаемости вводятся в свойствах грунта.", 0.93),
            ("Параметры сетки фильтрации влияют на сходимость решения.", 0.92),
            ("Результаты напоров выводятся после завершения расчёта.", 0.91),
        ]
    ]
    llm = FakeLLM("Порядок настройки [1][2][3][4][5].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Расскажи, как настроить фильтрацию сетки")

    assert len(result["sources"]) == 5
    assert all(source["file_name"] == "manual-seepage.pdf" for source in result["sources"])
    assert "Это широкий обзорный вопрос" not in llm.prompts[0]


def test_overview_question_backfills_context_from_single_file(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                text=text,
                metadata={"source": "manual-boundary.pdf"},
            ),
            score=score,
        )
        for text, score in [
            ("Граничные условия задают поведение модели на краях области.", 0.95),
            ("Опорные закрепления ограничивают перемещения выбранных узлов.", 0.94),
            ("Симметричные границы позволяют сократить расчётную область.", 0.93),
            ("Кинематические условия назначаются через мастер границ.", 0.92),
            ("Значения реакций проверяются после окончания расчёта.", 0.91),
        ]
    ]
    llm = FakeLLM("Обзор граничных условий [1][2][3][4][5].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Что такое граничные условия?")

    assert len(result["sources"]) == 5
    assert all(source["file_name"] == "manual-boundary.pdf" for source in result["sources"])
    assert "Это широкий обзорный вопрос" in llm.prompts[0]


def test_overview_question_prefers_diverse_files_then_backfills(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(text=text, metadata={"source": file_name}),
            score=score,
        )
        for file_name, text, score in [
            ("manual-a.pdf", "Граничные условия в опорах.", 0.95),
            ("manual-a.pdf", "Граничные условия в закреплениях.", 0.94),
            ("manual-b.pdf", "Граничные условия при нагрузках.", 0.93),
            ("manual-a.pdf", "Граничные условия для свай.", 0.92),
        ]
    ]
    llm = FakeLLM("Обзор [1][2][3][4].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Расскажи про граничные условия")

    assert [source["file_name"] for source in result["sources"]] == [
        "manual-a.pdf",
        "manual-b.pdf",
        "manual-a.pdf",
        "manual-a.pdf",
    ]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Расскажи про граничные условия", True),
        ("Расскажи про типы расчётов", True),
        ("Расскажи, что такое фильтрационный расчёт", True),
        ("Что такое модель Мора-Кулона?", True),
        ("Что ты знаешь о граничных условиях?", True),
        ("What is Mohr Coulomb?", True),
        ("Расскажи, как настроить фильтрацию сетки", False),
        ("Расскажи как настроить фильтрацию сетки", False),
        ("Расскажи про установку программы", False),
        ("Расскажи про ошибку лицензирования", False),
        ("Объясни, почему расчёт не сходится", False),
        ("Опиши шаги создания модели", False),
        ("Explain how to fix mesh errors", False),
        ("Tell me about the installation steps", False),
        ("Какие степени свободы используются?", False),
        ("Чем отличается модифицированная модель?", False),
    ],
)
def test_overview_question_detection(question: str, expected: bool) -> None:
    assert RAGService._is_overview_question(question) is expected


def test_rag_filters_legacy_civil_payloads_and_keeps_generic_gts_manual(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(text="GTS NX degrees of freedom", metadata={"source": "03_midas GTS NX_elements.pdf"}),
            score=0.91,
        ),
        NodeWithScore(
            node=TextNode(text="Civil bridge calculation", metadata={"source": "midas Civil 2020.pdf"}),
            score=0.89,
        ),
        NodeWithScore(
            node=TextNode(text="Generic GTS chapter", metadata={"source": "05_User Manual_Chapter5.pdf"}),
            score=0.87,
        ),
    ]
    retriever = SimpleNamespace(retrieve=lambda _question: nodes)
    llm = FakeLLM("Ответ [1] и [2].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = retriever

    result = service.answer("Какие степени свободы используются?")

    assert [source["file_name"] for source in result["sources"]] == [
        "03_midas GTS NX_elements.pdf",
        "05_User Manual_Chapter5.pdf",
    ]
    assert "Civil bridge calculation" not in llm.prompts[0]


def test_rag_limits_context_to_five_and_removes_duplicates_without_file_cap(tmp_path: Path) -> None:
    sources = [
        ("manual-a.pdf", "identical fragment", 0.99),
        ("manual-a.pdf", "second fragment", 0.98),
        ("manual-a.pdf", "third valid fragment", 0.97),
        ("manual-b.pdf", "identical fragment", 0.96),
        ("manual-b.pdf", "fourth unique fragment", 0.95),
        ("manual-c.pdf", "fifth unique fragment", 0.94),
        ("manual-d.pdf", "sixth unique fragment", 0.93),
        ("manual-e.pdf", "seventh unique fragment", 0.92),
    ]
    nodes = [
        NodeWithScore(
            node=TextNode(text=text, metadata={"source": file_name}),
            score=score,
        )
        for file_name, text, score in sources
    ]
    service = RAGService(
        rag_config(tmp_path, top_k=10, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1], [2], [3], [4], [5]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Вопрос")

    assert len(result["sources"]) == 5
    assert [source["file_name"] for source in result["sources"]].count("manual-a.pdf") == 3
    assert [source["snippet"] for source in result["sources"]].count("identical fragment") == 1


def test_expanded_raw_pool_recovers_five_contexts_after_product_filter(tmp_path: Path) -> None:
    nodes = [
        *[
            NodeWithScore(
                node=TextNode(text=f"Civil fragment {index}", metadata={"source": "midas Civil manual.pdf"}),
                score=0.99 - index / 100,
            )
            for index in range(8)
        ],
        *[
            NodeWithScore(
                node=TextNode(text=f"Dominant GTS fragment {index}", metadata={"source": "GTS_NX dominant.pdf"}),
                score=0.90 - index / 100,
            )
            for index in range(6)
        ],
        *[
            NodeWithScore(
                node=TextNode(text=f"Diverse GTS fragment {index}", metadata={"source": f"GTS_NX valid-{index}.pdf"}),
                score=0.80 - index / 100,
            )
            for index in range(5)
        ],
    ]
    service = RAGService(
        rag_config(tmp_path, top_k=10, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1], [2], [3], [4], [5]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Вопрос")

    assert service._raw_candidate_limit() == 30
    assert len(result["sources"]) == 5
    assert all("Civil" not in source["file_name"] for source in result["sources"])
    assert [source["file_name"] for source in result["sources"]].count("GTS_NX dominant.pdf") == 5


def _civil_nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="Пружинные опоры задаются в свойствах граничных условий.",
                metadata={"source": "midas Civil 2020.pdf"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="Граничные условия Civil назначаются по узлам.",
                metadata={"source": "midas Civil guide.pdf"},
            ),
            score=0.85,
        ),
    ]


def test_rag_product_scope_fallback_reaches_other_product_when_primary_empty(tmp_path: Path) -> None:
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True, product_scope="fallback"),
        llm=FakeLLM("Пружинные опоры задаются так [1] и [2]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: _civil_nodes())

    result = service.answer("Как задать пружинные опоры?")

    assert result["answer"] != NO_ANSWER
    assert [source["file_name"] for source in result["sources"]] == [
        "midas Civil 2020.pdf",
        "midas Civil guide.pdf",
    ]


def test_rag_product_scope_strict_keeps_other_product_unreachable(tmp_path: Path) -> None:
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True, product_scope="strict"),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: _civil_nodes())

    result = service.answer("Как задать пружинные опоры?")

    assert result["answer"] == NO_ANSWER
    assert result["sources"] == []


def test_rag_product_scope_fallback_prefers_primary_evidence(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="Пружинные опоры GTS NX задаются в граничных условиях.",
                metadata={"source": "GTS_NX boundary.pdf"},
            ),
            score=0.9,
        ),
        *_civil_nodes(),
    ]
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True, product_scope="fallback"),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Как задать пружинные опоры?")

    assert [source["file_name"] for source in result["sources"]] == ["GTS_NX boundary.pdf"]


def test_rag_product_scope_fallback_engages_when_primary_below_threshold(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="Слабый фрагмент GTS NX о границах.",
                metadata={"source": "GTS_NX weak.pdf"},
            ),
            score=0.2,
        ),
        *_civil_nodes(),
    ]
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True, product_scope="fallback"),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Как задать пружинные опоры?")

    assert result["answer"] != NO_ANSWER
    assert result["sources"][0]["file_name"] == "midas Civil 2020.pdf"


def test_rag_product_scope_all_disables_product_filter(tmp_path: Path) -> None:
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True, product_scope="all"),
        llm=FakeLLM("Ответ [1] и [2]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: _civil_nodes())

    result = service.answer("Как задать пружинные опоры?")

    assert result["answer"] != NO_ANSWER
    assert {source["file_name"] for source in result["sources"]} == {
        "midas Civil 2020.pdf",
        "midas Civil guide.pdf",
    }


@pytest.mark.parametrize("scope", ["all", "any", "none"])
def test_rag_config_rejects_scope_tokens_as_target_product(tmp_path: Path, scope: str) -> None:
    with pytest.raises(ValueError, match="target_product"):
        rag_config(tmp_path, target_product=scope)


def test_rag_config_rejects_unknown_product_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="product_scope"):
        rag_config(tmp_path, product_scope="everything")


def test_rag_config_reads_product_scope_setting() -> None:
    assert RAGConfig.from_settings(SimpleNamespace()).product_scope == "fallback"
    assert RAGConfig.from_settings(SimpleNamespace(rag_product_scope="strict")).product_scope == "strict"
    assert RAGConfig.from_settings(SimpleNamespace(rag_product_scope="all")).product_scope == "all"


def test_lexical_evidence_recovers_exact_dof_chunk_before_dense_distractors(tmp_path: Path) -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="Для оболочечных элементов условия освобождения степеней свободы задаются в NCS.",
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page_number": 41},
            ),
            score=0.96,
        ),
        NodeWithScore(
            node=TextNode(
                text="Степени свободы узла элемента пяты сваи задаются в осевом направлении.",
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page_number": 73},
            ),
            score=0.93,
        ),
        NodeWithScore(
            node=TextNode(
                text=(
                    "Конструкционный расчет характеризуется тремя поступательными и тремя "
                    "вращательными степенями свободы. Всего узел имеет 6 степеней свободы."
                ),
                metadata={
                    "source": "02_Пособие_GTS_NX_Узел_Степень свободы.pdf",
                    "page_number": 8,
                },
            ),
            score=0.68,
        ),
        NodeWithScore(
            node=TextNode(
                text="Конструкционный расчет Civil использует степени свободы мостовых узлов.",
                metadata={"source": "Руководство midas Civil 2020.pdf", "page_number": 20},
            ),
            score=0.99,
        ),
    ]
    llm = FakeLLM("Три поступательные и три вращательные, всего шесть [1].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _question: nodes)

    result = service.answer("Какие степени свободы используются в конструкционном расчёте GTS NX?")

    assert result["sources"][0]["file_name"] == "02_Пособие_GTS_NX_Узел_Степень свободы.pdf"
    assert result["sources"][0]["page"] == 8
    assert "тремя поступательными" in result["sources"][0]["snippet"]
    assert all("Civil" not in source["file_name"] for source in result["sources"])
    assert "Всего узел имеет 6 степеней свободы" in llm.prompts[0]


def test_generation_uses_full_selected_chunk_not_truncated_source_preview(tmp_path: Path) -> None:
    """An answer after a long PDF page header must remain available to the LLM.

    This reflects the GTS NX page where the phrase "тремя поступательными"
    occurs before the phrase "тремя вращательными".  The public source preview
    is intentionally short, but it must not become the generation context.
    """

    header = "ПОСОБИЕ ПО РАСЧЕТАМ. Midas GTS NX. " * 24
    selected_text = (
        f"{header}Задача конструкционного расчета характеризуется тремя поступательными "
        "и тремя вращательными степенями свободы. Всего используется 6 степеней свободы."
    )
    assert len(" ".join(selected_text.split())) > 700
    class GroundedFakeLLM(FakeLLM):
        def complete(self, prompt: str):
            self.prompts.append(prompt)
            if "тремя вращательными степенями свободы" not in prompt:
                return SimpleNamespace(text="В контексте перечисление обрывается [1].")
            return SimpleNamespace(text="Три поступательные и три вращательные, всего шесть [1].")

    llm = GroundedFakeLLM()
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(
        retrieve=lambda _question: [
            NodeWithScore(
                node=TextNode(
                    text=selected_text,
                    metadata={
                        "source": "02_Пособие_GTS_NX_Узел_Степень свободы.pdf",
                        "page_number": 8,
                    },
                ),
                score=0.91,
            )
        ]
    )

    result = service.answer("Какие степени свободы используются в конструкционном расчёте GTS NX?")

    assert len(result["sources"]) == 1
    assert len(result["sources"][0]["snippet"]) == 700
    assert "тремя вращательными" not in result["sources"][0]["snippet"]
    assert "тремя вращательными степенями свободы" in llm.prompts[0]
    assert "Всего используется 6 степеней свободы" in llm.prompts[0]
    assert result["answer"] == "Три поступательные и три вращательные, всего шесть [1]."


def test_llamaindex_reranks_after_top_k_and_guards_before_llm(tmp_path: Path) -> None:
    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, int]] = []

        def rerank(self, question: str, candidates, *, top_n: int):
            self.calls.append((question, len(candidates), top_n))
            return [candidates[2], candidates[1], candidates[0]][:top_n]

    llm = FakeLLM("Ответ [1] и [2].")
    reranker = FakeReranker()
    service = RAGService(
        rag_config(tmp_path, rerank_enabled=True, rerank_top_n=2, allow_external_data_transfer=True),
        llm=llm,
        reranker=reranker,
    )
    service._retriever = FakeRetriever([0.91, 0.82, 0.73])

    result = service.query("Каков срок возврата?")

    assert reranker.calls == [("Каков срок возврата?", 3, 3)]
    assert [source["file_name"] for source in result["sources"]] == ["manual-3.pdf", "manual-2.pdf"]
    # Confidence is measured on the ranked pool before the context limit
    # drops the reranked tail: the 0.91 candidate still proves retrieval
    # found a strong match even though it did not reach the prompt.
    assert result["top_score"] == 0.91
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
        "А для электроники?",
        "Каков срок возврата? Срок зависит от категории товара. А для электроники?",
    ]


def test_independent_question_does_not_use_unrelated_history_variant(tmp_path: Path) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        "Какие модели материала доступны?",
        history=[{"role": "user", "content": "Как настроить сетку?"}],
    )

    assert retriever.questions == ["Какие модели материала доступны?"]


def test_self_contained_question_with_pronoun_does_not_use_unrelated_history(tmp_path: Path) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    question = "Какие модели материала доступны и как их параметры влияют на расчёт?"
    service.answer(question, history=[{"role": "user", "content": "Как настроить сетку?"}])

    assert retriever.questions == [question]


def test_explicit_noun_follow_up_is_already_standalone(tmp_path: Path) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    question = "Какие степени свободы имеет элемент плоской деформации?"
    service.answer(question, history=[{"role": "user", "content": "Расскажи о типах элементов."}])

    assert retriever.questions == [question]


def test_history_follow_up_merges_raw_recall_with_standalone_topic(tmp_path: Path) -> None:
    """A pronoun follow-up must recover the plane-strain DOF subsection.

    The first node has a higher dense score and mentions both concepts, but in
    separate sentences about a different element.  The selected node contains
    the topic and new constraint in one answer sentence.
    """

    orientation_distractor = NodeWithScore(
        node=TextNode(
            text=(
                "Ориентация материала для элемента плоского напряженного состояния задается углом. "
                "Степени свободы элемента плоского напряженного состояния определяются по осям ECS."
            ),
            metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page_number": 79},
        ),
        score=0.98,
    )
    plane_strain_dof = NodeWithScore(
        node=TextNode(
            text=(
                "Элемент плоской деформации характеризуется степенями свободы "
                "по направлениям осей x и y элементной системы координат."
            ),
            metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page_number": 45},
        ),
        score=0.9,
    )
    unrelated = NodeWithScore(
        node=TextNode(
            text="Плоская деформация применяется при расчете толщины основания.",
            metadata={"source": "GTS_NX_guide.pdf", "page_number": 12},
        ),
        score=0.91,
    )

    class GroundedFakeLLM(FakeLLM):
        def complete(self, prompt: str):
            self.prompts.append(prompt)
            if "по направлениям осей x и y элементной системы координат" not in prompt:
                return SimpleNamespace(text="Надежного ответа нет.")
            return SimpleNamespace(text="Используются степени свободы по осям x и y ECS [1].")

    class FollowUpRetriever:
        def __init__(self) -> None:
            self.questions: list[str] = []

        def retrieve(self, query: str):
            self.questions.append(query)
            # The short wording can recall the exact subsection by its heading,
            # while the focused topic query recalls a dense but wrong neighbour.
            if query == "А какие из них используются для плоской деформации?":
                return [plane_strain_dof]
            return [orientation_distractor, unrelated]

    llm = GroundedFakeLLM()
    retriever = FollowUpRetriever()
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = retriever

    result = service.answer(
        "А какие из них используются для плоской деформации?",
        history=[
            {
                "role": "user",
                "content": "Какие степени свободы используются в конструкционном расчёте GTS NX?",
            }
        ],
    )

    assert result["sources"][0]["page"] == 45
    assert "по направлениям осей x и y" in result["sources"][0]["snippet"]
    assert retriever.questions == [
        "А какие из них используются для плоской деформации?",
        (
            "Какие степени свободы используются в конструкционном расчёте GTS NX? "
            "А какие из них используются для плоской деформации?"
        ),
    ]
    assert llm.prompts and "по направлениям осей x и y элементной системы координат" in llm.prompts[0]
    assert result["answer"] == "Используются степени свободы по осям x и y ECS [1]."


def test_how_question_with_early_reference_is_follow_up(tmp_path: Path) -> None:
    """«как» belongs to the wh-set: a mid-length how-question anchored by an
    anaphora resolves its subject from the previous exchange."""

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        "Как мне настроить этот параметр?",
        history=[{"role": "user", "content": "Какие параметры сетки бывают?"}],
    )

    assert retriever.questions == [
        "Как мне настроить этот параметр?",
        "Какие параметры сетки бывают? Как мне настроить этот параметр?",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Подробнее",
        "А подробнее?",
        "Почему?",
        "Например, для грунтов?",
        "Для 2D?",
        "Для него?",
        "И что будет, если его не задать?",
        "Ну а для плоской деформации?",
        "Этот метод применяется для плоской деформации?",
    ],
)
def test_widened_follow_up_markers_use_history_variant(tmp_path: Path, question: str) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        question,
        history=[{"role": "user", "content": "Какие параметры сетки бывают?"}],
    )

    assert retriever.questions == [question, f"Какие параметры сетки бывают? {question}"]


def test_follow_up_merges_last_assistant_answer_for_assistant_owned_referent(
    tmp_path: Path,
) -> None:
    """The referent may live in the assistant's answer, not the user question.

    "из них" points at the list of element types produced by the assistant, so
    the expanded query must carry the previous exchange, not only the last
    user wording.  Stale citation markers are stripped from the merge.
    """

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        "А какие из них подходят для плоской деформации?",
        history=[
            {"role": "user", "content": "Какие типы элементов есть в GTS NX?"},
            {"role": "assistant", "content": "В GTS NX есть 1D, 2D и 3D элементы [1]."},
        ],
    )

    assert retriever.questions == [
        "А какие из них подходят для плоской деформации?",
        (
            "Какие типы элементов есть в GTS NX? "
            "В GTS NX есть 1D, 2D и 3D элементы. "
            "А какие из них подходят для плоской деформации?"
        ),
    ]


def test_chained_follow_up_uses_latest_exchange_not_first_question(tmp_path: Path) -> None:
    """A follow-up to a follow-up must not resurrect the first user question.

    The previous user turn is itself elliptical, so the assistant's answer is
    the part that still carries the topical terms for retrieval.
    """

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        "А для осесимметричной задачи?",
        history=[
            {"role": "user", "content": "Какие типы элементов есть в GTS NX?"},
            {"role": "assistant", "content": "Есть 1D, 2D и 3D элементы."},
            {"role": "user", "content": "А какие из них для плоской деформации?"},
            {
                "role": "assistant",
                "content": "Для плоской деформации применяются элементы с двумя степенями свободы.",
            },
        ],
    )

    merged = retriever.questions[1]
    assert "А какие из них для плоской деформации?" in merged
    assert "Для плоской деформации применяются элементы с двумя степенями свободы." in merged
    assert "Какие типы элементов есть в GTS NX?" not in merged


def test_follow_up_merge_skips_abstention_assistant_answer(tmp_path: Path) -> None:
    """A stored NO_ANSWER assistant turn adds no topical terms to the merge."""

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    service.answer(
        "А для электроники?",
        history=[
            {"role": "user", "content": "Каков срок возврата?"},
            {"role": "assistant", "content": NO_ANSWER},
        ],
    )

    assert retriever.questions == [
        "А для электроники?",
        "Каков срок возврата? А для электроники?",
    ]


def test_self_contained_why_question_without_reference_stays_independent(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    question = "Почему метод сил не используется для динамического расчёта?"
    service.answer(question, history=[{"role": "user", "content": "Как настроить сетку?"}])

    assert retriever.questions == [question]


def test_self_contained_question_with_late_possessive_stays_independent(
    tmp_path: Path,
) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
    )
    service._retriever = retriever

    question = "Какие параметры элемента влияют на его жёсткость в расчёте?"
    service.answer(question, history=[{"role": "user", "content": "Как настроить сетку?"}])

    assert retriever.questions == [question]


def test_generation_prompt_includes_recent_dialog_without_stale_citations(tmp_path: Path) -> None:
    """The generator must see the conversation, not only the elliptical wording.

    The follow-up "А для электроники?" is unanswerable without the prior turn,
    and citation markers inside previous assistant messages belong to a
    different fragment numbering: they are stripped from the dialog block so
    they cannot be mistaken for current sources.
    """

    llm = FakeLLM("Срок возврата электроники — 14 дней [1].")
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = FakeRetriever([0.91])

    service.answer(
        "А для электроники?",
        history=[
            {"role": "user", "content": "Каков срок возврата товара?"},
            {"role": "assistant", "content": "Для большинства товаров 30 дней [1]."},
        ],
    )

    prompt = llm.prompts[0]
    assert "Диалог:\n" in prompt
    dialog = prompt.split("Диалог:", 1)[1].split("Контекст:", 1)[0]
    assert "Пользователь: Каков срок возврата товара?" in dialog
    assert "Ассистент: Для большинства товаров 30 дней" in dialog
    assert "[1]" not in dialog and "[2]" not in dialog
    assert "Вопрос пользователя: А для электроники?" in prompt


def test_generation_prompt_omits_dialog_section_without_history(tmp_path: Path) -> None:
    llm = FakeLLM("Ответ [1].")
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = FakeRetriever([0.91])

    service.answer("Самостоятельный вопрос")

    assert "Диалог:" not in llm.prompts[0]


def test_generation_dialog_is_bounded_to_recent_turns(tmp_path: Path) -> None:
    llm = FakeLLM("Ответ [1].")
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True), llm=llm)
    service._retriever = FakeRetriever([0.91])

    history = [
        {"role": "user", "content": f"сообщение-{index}"}
        for index in range(10)
    ]
    service.answer("Итоговый вопрос", history=history)

    dialog = llm.prompts[0].split("Диалог:", 1)[1].split("Контекст:", 1)[0]
    assert "сообщение-9" in dialog
    assert "сообщение-4" in dialog
    assert "сообщение-3" not in dialog


def test_follow_up_builds_strict_lexical_query_from_current_constraint_and_prior_subject(
    tmp_path: Path,
) -> None:
    service = RAGService(rag_config(tmp_path))
    question = "А какие из них используются для плоской деформации?"
    history = [
        {
            "role": "user",
            "content": "Какие степени свободы используются в конструкционном расчёте GTS NX?",
        }
    ]
    plan = service._retrieval_plan(question, history, None)

    assert service._lexical_query(question, history, plan) == (
        "плоской деформации степени свободы"
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Расскажи мне про модель Мора-Кулона", "модель мора кулона"),
        ("Что такое Мора-Кулон?", "мора кулон"),
        ("Что ты знаешь о граничных условиях?", "граничных условиях"),
        ("Please tell me about Mohr Coulomb", "mohr coulomb"),
        ("What is Mohr Coulomb?", "mohr coulomb"),
    ],
)
def test_independent_lexical_query_removes_conversational_lead_ins(
    tmp_path: Path,
    question: str,
    expected: str,
) -> None:
    service = RAGService(rag_config(tmp_path))
    plan = service._retrieval_plan(question, None, None)

    assert service._lexical_query(question, None, plan) == expected


def test_lexical_queries_add_entity_tail_fallbacks_for_unknown_lead_in(tmp_path: Path) -> None:
    service = RAGService(rag_config(tmp_path))
    question = "Меня интересует модель Мора-Кулона"
    plan = service._retrieval_plan(question, None, None)

    queries = service._lexical_queries(question, None, plan)

    assert queries[0] == "меня интересует модель мора"
    assert "модель мора кулона" in queries
    assert len(queries) <= 8


def test_lexical_queries_find_entity_before_trailing_conversation(tmp_path: Path) -> None:
    service = RAGService(rag_config(tmp_path))
    question = "Модель Мора-Кулона, хочу узнать про неё"
    plan = service._retrieval_plan(question, None, None)

    queries = service._lexical_queries(question, None, plan)

    assert queries[0] == "модель мора кулона хочу"
    assert queries[1] == "модель мора кулона"


def test_fulltext_ranking_uses_dense_score_to_break_exact_entity_ties(tmp_path: Path) -> None:
    modified = NodeWithScore(
        node=TextNode(
            id_="modified",
            text="Модифицированная модель Мора-Кулона применяется для упрочняющихся грунтов.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf"},
        ),
        score=0.10,
    )
    classic = NodeWithScore(
        node=TextNode(
            id_="classic",
            text="Модель Мора-Кулона применяется для моделирования поведения грунтов.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf"},
        ),
        score=0.38,
    )

    ranked = RAGService(rag_config(tmp_path))._rank_fulltext_evidence(
        "Расскажи про модель Мора-Кулона",
        [modified, classic],
        retrieval_query="модель мора кулона",
    )

    assert ranked[0].node.node_id == "classic"


def test_conversational_entity_query_accepts_strict_lexical_evidence_below_dense_threshold(
    tmp_path: Path,
) -> None:
    dense = NodeWithScore(
        node=TextNode(
            text="Общие сведения о материалах.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf"},
        ),
        score=0.38,
    )
    classic = NodeWithScore(
        node=TextNode(
            id_="mohr-coulomb-classic",
            text=(
                "Модель Мора-Кулона используется для моделирования поведения "
                "грунтовых оснований в нелинейных расчётах."
            ),
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf", "page": 39},
        ),
        score=0.0,
    )
    service = RAGService(
        rag_config(tmp_path, min_score=0.45, allow_external_data_transfer=True),
        llm=FakeLLM("Модель описывает нелинейное поведение грунтов [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _query: [dense])
    service._lexical_available = True
    service._retrieve_lexical_candidates = lambda _query: [classic]  # type: ignore[method-assign]

    result = service.answer("Расскажи мне про модель Мора-Кулона")

    assert result["answer"] != NO_ANSWER
    assert result["sources"][0]["id"] == "mohr-coulomb-classic"
    assert result["sources"][0]["confident"] is True


def test_entity_tail_fallback_recovers_named_concept_after_unknown_lead_in(
    tmp_path: Path,
) -> None:
    dense = NodeWithScore(
        node=TextNode(
            text="Общие сведения о материалах.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf"},
        ),
        score=0.38,
    )
    classic = NodeWithScore(
        node=TextNode(
            id_="mohr-coulomb-classic",
            text="Модель Мора-Кулона описывает поведение грунтовых оснований.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf", "page": 39},
        ),
        score=0.0,
    )
    calls: list[str] = []

    def lexical_candidates(query: str) -> list[NodeWithScore]:
        calls.append(query)
        return [classic] if query == "модель мора кулона" else []

    service = RAGService(
        rag_config(tmp_path, min_score=0.45, allow_external_data_transfer=True),
        llm=FakeLLM("Модель описывает поведение грунта [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _query: [dense])
    service._lexical_available = True
    service._retrieve_lexical_candidates = lexical_candidates  # type: ignore[method-assign]

    result = service.answer("Меня интересует модель Мора-Кулона")

    assert calls == [
        "меня интересует модель мора",
        "меня интересует модель",
        "модель мора кулона",
    ]
    assert result["answer"] != NO_ANSWER
    assert result["sources"][0]["id"] == "mohr-coulomb-classic"


def test_dense_entity_window_recovers_misspelled_subject_before_trailing_phrase(
    tmp_path: Path,
) -> None:
    question = "Модель Мора-Кулана, хочу узнать про неё"
    low_dense = NodeWithScore(
        node=TextNode(
            id_="generic",
            text="Общие сведения о материалах.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf"},
        ),
        score=0.2,
    )
    classic = NodeWithScore(
        node=TextNode(
            id_="mohr-coulomb-classic",
            text="Модель Мора-Кулона описывает поведение грунтовых оснований.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf", "page": 39},
        ),
        score=0.78,
    )

    class WindowRetriever:
        def __init__(self) -> None:
            self.questions: list[str] = []

        def retrieve(self, query: str) -> list[NodeWithScore]:
            self.questions.append(query)
            if query == "модель мора кулана":
                return [classic]
            return [low_dense] if query == question else []

    retriever = WindowRetriever()
    service = RAGService(
        rag_config(tmp_path, min_score=0.45, allow_external_data_transfer=True),
        llm=FakeLLM("Модель описывает поведение грунта [1]."),
    )
    service._retriever = retriever
    service._lexical_available = True
    service._retrieve_lexical_candidates = lambda _query: []  # type: ignore[method-assign]

    result = service.answer(question)

    assert retriever.questions == [question, "модель мора кулана"]
    assert result["answer"] != NO_ANSWER
    assert result["sources"][0]["id"] == "mohr-coulomb-classic"


def test_lexical_context_focus_excludes_distant_neighbouring_section(tmp_path: Path) -> None:
    node = NodeWithScore(
        node=TextNode(
            text=(
                ("Ориентация материала задаётся проекцией и углом. " * 30)
                + "Степени свободы (DOF). Элемент плоской деформации характеризуется "
                "степенями свободы по направлениям осей x и y элементной системы координат."
            ),
            metadata={
                "source": "03_Пособие_GTS_NX_Элементы.pdf",
                "_lexical_focus_query": "плоской деформации степени свободы",
            },
        ),
        score=0.0,
    )

    focused = RAGService(rag_config(tmp_path))._context_text(node)

    assert "осей x и y элементной системы координат" in focused
    assert focused.count("Ориентация материала") < 10


def test_hybrid_recall_reserves_exact_fulltext_evidence_missing_from_dense_results(
    tmp_path: Path,
) -> None:
    dense = [
        NodeWithScore(
            node=TextNode(
                id_=f"dense-{index}",
                text=f"Геометрическая жесткость и ориентация элемента {index}.",
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf"},
            ),
            score=0.90 - index / 1000,
        )
        for index in range(30)
    ]
    lexical = [
        NodeWithScore(
            node=TextNode(
                id_="civil-fulltext-neighbour",
                text="Civil: плоская деформация и степени свободы элементов.",
                metadata={"source": "Руководство по расчетам midas Civil 2020.pdf", "page": 9},
            ),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_="gts-chm-non-direct",
                text=(
                    "Для двумерной модели задана ширина плоской деформации. "
                    + ("Описание параметров демпфирования. " * 12)
                    + "Отдельно пользователь выбирает степени свободы демпфирования."
                ),
                metadata={"source": "GTS_NX 2021.chm"},
            ),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_="axisymmetric-neighbour",
                text=(
                    "Осесимметричный элемент и ориентация материала. "
                    + ("Соседнее описание. " * 30)
                    + "Для плоской деформации и степеней свободы приведена ссылка."
                ),
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page": 37},
            ),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_="orientation-neighbour",
                text=(
                    "Плоское напряжённое состояние и ориентация материала. "
                    + ("Соседнее описание. " * 30)
                    + "Упомянут элемент плоской деформации и его степени свободы."
                ),
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page": 56},
            ),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_="plane-strain-exact",
                text=(
                    "ПОСОБИЕ ПО РАСЧЕТАМ. Midas GTS NX\n45\n"
                    "Элемент плоской деформации характеризуется степенями свободы "
                    "по направлениям осей x и y элементной системы координат."
                ),
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page": 32},
            ),
            score=0.0,
        ),
    ]

    class CitationByEvidenceLLM(FakeLLM):
        def complete(self, prompt: str):
            self.prompts.append(prompt)
            labels = re.findall(r"\[(\d+)\] Файл: ([^\n]+)", prompt)
            exact_position = next(
                (
                    number
                    for number, block in re.findall(
                        r"\[(\d+)\] Файл:.*?\n(.*?)(?=\n\n\[\d+\] Файл:|\n\nВопрос пользователя:)",
                        prompt,
                        flags=re.DOTALL,
                    )
                    if "по направлениям осей x и y элементной системы координат" in block
                ),
                None,
            )
            assert exact_position is not None, labels
            return SimpleNamespace(
                text=f"Используются степени свободы по осям x и y [{exact_position}]."
            )

    service = RAGService(
        rag_config(tmp_path, top_k=10, allow_external_data_transfer=True),
        llm=CitationByEvidenceLLM(),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _query: dense)
    service._lexical_available = True
    service._retrieve_lexical_candidates = lambda _query: lexical  # type: ignore[method-assign]

    result = service.answer(
        "А какие из них используются для плоской деформации?",
        history=[
            {
                "role": "user",
                "content": "Какие степени свободы используются в конструкционном расчёте GTS NX?",
            }
        ],
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["id"] == "plane-strain-exact"
    assert result["sources"][0]["page"] == 32
    assert result["sources"][0]["printed_page"] == 45
    assert result["sources"][0]["confident"] is True
    assert all("Civil" not in source["file_name"] for source in result["sources"])


def test_follow_up_lexical_hit_keeps_dense_alternatives_in_context(tmp_path: Path) -> None:
    """A wrong lexical hit must not starve the follow-up context.

    The reserved all-token fragment still leads the prompt, but the strongest
    dense candidates follow it: when the lexical match lands on an off-topic
    glossary mention, the generator still sees the fragment carrying the
    answer instead of being locked to a single wrong source.
    """

    off_topic_lexical = NodeWithScore(
        node=TextNode(
            id_="glossary-mention",
            text=(
                "В глоссарии перечислены термины: плоская деформация, "
                "степени свободы, осесимметричная задача."
            ),
            metadata={"source": "GTS_NX_glossary.chm"},
        ),
        score=0.0,
    )
    dof_answer = NodeWithScore(
        node=TextNode(
            id_="plane-strain-dof",
            text=(
                "Элемент плоской деформации характеризуется степенями свободы "
                "по направлениям осей x и y элементной системы координат."
            ),
            metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page": 45},
        ),
        score=0.9,
    )
    llm = FakeLLM("Степени свободы по осям x и y [2].")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=llm,
    )
    service._retriever = SimpleNamespace(retrieve=lambda _query: [dof_answer])
    service._lexical_available = True
    service._retrieve_lexical_candidates = lambda _query: [off_topic_lexical]  # type: ignore[method-assign]

    result = service.answer(
        "А какие из них используются для плоской деформации?",
        history=[
            {
                "role": "user",
                "content": "Какие степени свободы используются в конструкционном расчёте GTS NX?",
            }
        ],
    )

    prompt = llm.prompts[0]
    assert "[1] Файл: GTS_NX_glossary.chm" in prompt
    assert "[2] Файл: 03_Пособие_GTS_NX_Элементы.pdf" in prompt
    assert "по направлениям осей x и y элементной системы координат" in prompt
    assert result["answer"] == "Степени свободы по осям x и y [1]."
    assert [source["id"] for source in result["sources"]] == ["plane-strain-dof"]


def test_history_follow_up_deduplicates_candidates_from_both_queries(tmp_path: Path) -> None:
    shared = NodeWithScore(
        node=TextNode(text="Общий фрагмент с темой возврата.", metadata={"source": "faq.pdf"}),
        score=0.6,
    )
    focused_only = NodeWithScore(
        node=TextNode(text="Срок возврата электроники составляет 14 дней.", metadata={"source": "faq.pdf"}),
        score=0.8,
    )

    class VariantRetriever:
        def retrieve(self, query: str):
            if query == "А для электроники?":
                return [shared]
            return [NodeWithScore(node=shared.node, score=0.9), focused_only]

    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1] и [2]."),
    )
    service._retriever = VariantRetriever()

    result = service.answer(
        "А для электроники?",
        history=[{"role": "user", "content": "Каков срок возврата?"}],
    )

    assert len(result["sources"]) == 2
    assert sorted(source["score"] for source in result["sources"]) == [0.8, 0.9]
    assert len({source["id"] for source in result["sources"]}) == 2


def test_top_score_uses_candidate_pool_before_deduplication(tmp_path: Path) -> None:
    """A top-scoring candidate dropped as a near-duplicate must not abstain.

    The kept twin carries the same passage, so the retriever genuinely found
    confident evidence.  Scoring only the post-dedup selection used to fall
    below ``min_score`` and return ``NO_ANSWER`` even though the candidate
    pool contained a strong match.
    """

    shared_passage = (
        "Раздел пособия описывает материалы грунтов и их параметры. "
        "Значения cohesion, friction angle и dilatancy задаются в диалоге "
        "свойств материала и применяются в нелинейных расчётах основания."
    )
    low_twin = NodeWithScore(
        node=TextNode(
            id_="low-score-twin",
            text=f"{shared_passage} Модель Мора-Кулона описывает поведение грунта.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf", "page": 39},
        ),
        score=0.34,
    )
    high_twin = NodeWithScore(
        node=TextNode(
            id_="high-score-twin",
            text=f"{shared_passage} Модель Мор-Кулоновой описывает поведение грунта.",
            metadata={"source": "04_Пособие_GTS_NX_Материалы.pdf", "page": 40},
        ),
        score=0.9,
    )
    service = RAGService(
        rag_config(tmp_path, min_score=0.35, allow_external_data_transfer=True),
        llm=FakeLLM("Модель описывает поведение грунта [1]."),
    )
    service._retriever = SimpleNamespace(retrieve=lambda _query: [high_twin, low_twin])

    result = service.answer("модель мора кулона")

    assert result["answer"] != NO_ANSWER
    assert result["top_score"] == 0.9
    assert [source["id"] for source in result["sources"]] == ["low-score-twin"]


def test_follow_up_keeps_single_variant_exact_evidence_until_lexical_ranking(tmp_path: Path) -> None:
    shared = [
        NodeWithScore(
            node=TextNode(
                id_=f"shared-{index}",
                text=f"Общий раздел о степенях свободы, материал и ориентация {index}.",
                metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf"},
            ),
            # The thirtieth shared result is dense-strong enough to beat the
            # exact single-variant node at the old pre-lexical fusion cutoff.
            score=0.90 if index == 29 else 0.80 - index / 1000,
        )
        for index in range(30)
    ]
    exact = NodeWithScore(
        node=TextNode(
            id_="plane-strain-exact",
            text=(
                "Элемент плоской деформации характеризуется степенями свободы "
                "по направлениям осей x и y элементной системы координат."
            ),
            metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf", "page": 32},
        ),
        score=0.70,
    )

    class VariantRetriever:
        def retrieve(self, query: str):
            if query.startswith("А какие"):
                return [*shared[:29], exact]
            return shared

    service = RAGService(
        rag_config(tmp_path, top_k=10, allow_external_data_transfer=True),
        llm=FakeLLM("Используются оси x и y [1]."),
    )
    service._retriever = VariantRetriever()

    result = service.answer(
        "А какие из них используются для плоской деформации?",
        history=[
            {
                "role": "user",
                "content": "Какие степени свободы используются в конструкционном расчёте GTS NX?",
            }
        ],
    )

    assert result["sources"][0]["id"] == "plane-strain-exact"
    assert result["sources"][0]["page"] == 32


def test_legacy_pdf_page_and_printed_page_are_recovered_without_reindex(tmp_path: Path) -> None:
    node = TextNode(
        text=(
            "ПОСОБИЕ ПО РАСЧЕТАМ. Midas GTS NX\n"
            "Элемент плоской деформации характеризуется степенями свободы по направлениям "
            "осей x и y элементной системы координат.\n\n45\n"
        ),
        metadata={"source": "03_Пособие_GTS_NX_Элементы.pdf"},
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="legacy-document_part_31")
        },
    )
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True))

    source = service._source_from_node(NodeWithScore(node=node, score=0.91))

    assert source["page"] == 32
    assert source["printed_page"] == 45
    assert "PDF-страница: 32" in service._context_source_label(source)
    assert "страница документа: 45" in service._context_source_label(source)


def test_printed_page_does_not_use_isolated_table_value_late_in_chunk(tmp_path: Path) -> None:
    text = "Заголовок без номера страницы. " + ("Описание раздела. " * 30) + "\n45\n"
    service = RAGService(rag_config(tmp_path, allow_external_data_transfer=True))

    source = service._source_from_node(
        NodeWithScore(
            node=TextNode(text=text, metadata={"source": "GTS_NX_manual.pdf", "page": 12}),
            score=0.91,
        )
    )

    assert source["page"] == 12
    assert source["printed_page"] is None


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


def test_llm_condense_hook_rewrites_follow_up_into_single_standalone_query(tmp_path: Path) -> None:
    """The injected LLM hook replaces the regex merge with one resolved query."""

    condense_llm = FakeLLM("Какие степени свободы у элемента плоской деформации?")
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
        condense_hook=build_llm_condense_hook(rag_config(tmp_path), llm=condense_llm),
    )
    service._retriever = retriever

    service.answer(
        "А какие из них для плоской деформации?",
        history=[{"role": "user", "content": "Какие типы элементов есть в GTS NX?"}],
    )

    assert retriever.questions == ["Какие степени свободы у элемента плоской деформации?"]
    assert len(condense_llm.prompts) == 1
    assert "Последнее сообщение пользователя: А какие из них для плоской деформации?" in (
        condense_llm.prompts[0]
    )
    assert "Пользователь: Какие типы элементов есть в GTS NX?" in condense_llm.prompts[0]


def test_llm_condense_hook_echo_marks_question_independent(tmp_path: Path) -> None:
    """An unchanged rewrite is the model's verdict that the turn is standalone."""

    condense_llm = FakeLLM("Какие модели материала доступны?")
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
        condense_hook=build_llm_condense_hook(rag_config(tmp_path), llm=condense_llm),
    )

    plan = service._retrieval_plan(
        "Какие модели материала доступны?",
        [{"role": "user", "content": "Как настроить сетку?"}],
        None,
    )

    assert plan.mode == "independent"
    assert plan.queries == ("Какие модели материала доступны?",)


def test_llm_condense_hook_failure_falls_back_to_deterministic_merge(tmp_path: Path) -> None:
    """A condense outage must not disable the proven heuristic follow-up path."""

    class FailingLLM:
        def complete(self, _prompt: str):
            raise TimeoutError("provider timeout")

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
        condense_hook=build_llm_condense_hook(rag_config(tmp_path), llm=FailingLLM()),
    )
    service._retriever = retriever

    service.answer(
        "А для электроники?",
        history=[{"role": "user", "content": "Каков срок возврата?"}],
    )

    assert retriever.questions == [
        "А для электроники?",
        "Каков срок возврата? А для электроники?",
    ]


def test_raising_condense_hook_falls_back_to_deterministic_merge(tmp_path: Path) -> None:
    """Even a non-cooperative hook must not crash the answer path."""

    def broken_hook(_question: str, _history: object) -> str:
        raise RuntimeError("custom hook exploded")

    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
        condense_hook=broken_hook,
    )
    service._retriever = retriever

    service.answer(
        "А для электроники?",
        history=[{"role": "user", "content": "Каков срок возврата?"}],
    )

    assert retriever.questions == [
        "А для электроники?",
        "Каков срок возврата? А для электроники?",
    ]


@pytest.mark.parametrize("bad_output", ["", "   \n  ", "x" * 600])
def test_llm_condense_hook_rejects_contract_violating_output(
    tmp_path: Path, bad_output: str
) -> None:
    retriever = FakeRetriever([0.91])
    service = RAGService(
        rag_config(tmp_path, allow_external_data_transfer=True),
        llm=FakeLLM("Ответ [1]."),
        condense_hook=build_llm_condense_hook(rag_config(tmp_path), llm=FakeLLM(bad_output)),
    )
    service._retriever = retriever

    service.answer(
        "А для электроники?",
        history=[{"role": "user", "content": "Каков срок возврата?"}],
    )

    assert retriever.questions == [
        "А для электроники?",
        "Каков срок возврата? А для электроники?",
    ]


def test_llm_condense_hook_prompt_renders_bounded_dialog_without_stale_citations(
    tmp_path: Path,
) -> None:
    condense_llm = FakeLLM("Каков срок возврата электроники?")
    hook = build_llm_condense_hook(rag_config(tmp_path), llm=condense_llm)

    result = hook(
        "А для электроники?",
        [
            {"role": "user", "content": "Каков срок возврата товара?"},
            {"role": "assistant", "content": "Для большинства товаров 30 дней [1]."},
        ],
    )

    assert result == "Каков срок возврата электроники?"
    prompt = condense_llm.prompts[0]
    assert "Пользователь: Каков срок возврата товара?" in prompt
    assert "Ассистент: Для большинства товаров 30 дней" in prompt
    assert "[1]" not in prompt
    assert "Последнее сообщение пользователя: А для электроники?" in prompt


def test_llm_condense_hook_skips_llm_without_usable_history(tmp_path: Path) -> None:
    condense_llm = FakeLLM("не должен вызываться")
    hook = build_llm_condense_hook(rag_config(tmp_path), llm=condense_llm)

    assert hook("Вопрос", []) is None
    assert hook("Вопрос", [{"role": "system", "content": "правила"}]) is None
    assert condense_llm.prompts == []


def test_llm_condense_hook_strips_output_label_and_quotes(tmp_path: Path) -> None:
    condense_llm = FakeLLM("Самостоятельный вопрос: «Какие типы элементов есть?»")
    hook = build_llm_condense_hook(rag_config(tmp_path), llm=condense_llm)

    result = hook("А какие?", [{"role": "user", "content": "Расскажи про элементы."}])

    assert result == "Какие типы элементов есть?"


def test_lexical_query_uses_condensed_question_terms(tmp_path: Path) -> None:
    """A hook-rewritten plan feeds lexical recall from the resolved subject."""

    service = RAGService(rag_config(tmp_path))
    plan = RetrievalPlan(
        "Какие степени свободы у элемента плоской деформации?",
        ("Какие степени свободы у элемента плоской деформации?",),
        "condense_hook",
    )

    assert service._lexical_query(
        "А для них?", [{"role": "user", "content": "Расскажи про элементы."}], plan
    ) == "степени свободы элемента плоской"


def test_rag_config_reads_condense_llm_setting() -> None:
    config = RAGConfig.from_settings(SimpleNamespace(rag_condense_llm_enabled=False))

    assert config.condense_llm_enabled is False


def test_rag_config_reads_lexical_rank_weights() -> None:
    config = RAGConfig.from_settings(
        SimpleNamespace(rag_lexical_term_weight=0.5, rag_lexical_pair_weight=0.5)
    )

    assert config.lexical_term_weight == 0.5
    assert config.lexical_pair_weight == 0.5


def test_rag_config_defaults_keep_eval_tuned_lexical_weights(tmp_path: Path) -> None:
    config = rag_config(tmp_path)

    assert config.lexical_term_weight == 0.7
    assert config.lexical_pair_weight == 0.3


@pytest.mark.parametrize("field", ["lexical_term_weight", "lexical_pair_weight"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rag_config_rejects_out_of_range_lexical_weights(
    tmp_path: Path, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        rag_config(tmp_path, **{field: value})


def test_lexical_rank_weights_zero_disables_lexical_boost(tmp_path: Path) -> None:
    question = "степени свободы конструкционного расчёта"
    lexical_hit = NodeWithScore(
        node=TextNode(
            text="Узел имеет степени свободы конструкционного расчёта по всем осям.",
            metadata={"source": "manual.pdf"},
        ),
        score=0.50,
    )
    dense_only = NodeWithScore(
        node=TextNode(
            text="Несвязанное описание интерфейса программы.",
            metadata={"source": "other.pdf"},
        ),
        score=0.90,
    )

    default_service = RAGService(rag_config(tmp_path))
    ranked = default_service._rank_by_lexical_evidence(question, [dense_only, lexical_hit])
    assert ranked[0] is lexical_hit

    dense_service = RAGService(
        rag_config(tmp_path, lexical_term_weight=0.0, lexical_pair_weight=0.0)
    )
    ranked = dense_service._rank_by_lexical_evidence(question, [dense_only, lexical_hit])
    assert ranked[0] is dense_only


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
        llm=FakeLLM("Ответ [1], [2], [3], [4], [5]."),
    ).build()

    result = service.answer("mesh")

    assert len(result["sources"]) == 5
    assert result["answer"]
    assert service.build() is service
    service.close()


def test_llamaindex_build_reuses_existing_collection_without_startup_ingestion(tmp_path: Path) -> None:
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

    class IngestionMustNotRun:
        def ingest(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("existing collection must not be ingested at startup")

    existing_ingestion = IngestionMustNotRun()
    reused = RAGService(
        rag_config(
            tmp_path / "corpus-is-not-needed",
            collection_name="llamaindex_reuse_test",
            baremetal_collection_name="baremetal_reuse_test",
            embedding_dimensions=8,
            chunk_size=128,
            chunk_overlap=16,
            min_score=0,
            rebuild_on_start=False,
            ingest_on_start=False,
        ),
        client=client,
        embed_model=MockEmbedding(embed_dim=8),
        llm=FakeLLM("Ответ [1], [2], [3], [4], [5]."),
        ingestion_service=existing_ingestion,
    ).build()

    assert len(reused.answer("mesh")["sources"]) == 5
    assert reused.ingestion_service is existing_ingestion
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
        headers={
            "X-Internal-Token": "test-internal-token-0000000000000001",
            "X-User-ID": "support-engineer-1",
            "X-User-Interface": "web",
        },
        json={
            "question": "Проверочный вопрос",
            "chat_id": "chat-1",
            "history": [{"role": "user", "content": "Каков срок возврата?"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ: Проверочный вопрос"
    assert len(response.json()["sources"]) == 3
    assert set(response.json()["sources"][0]) == {
        "id",
        "file_name",
        "page",
        "printed_page",
        "score",
        "snippet",
        "confident",
    }
    assert not any(getattr(route, "path", None) == "/chat" for route in api.routes)
