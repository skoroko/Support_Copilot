from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import Document

from app.services.chunking import chunk_documents, russian_sentence_tokenizer
from app.services.reranker import BGEReranker, RerankCandidate
from app.services.retrieval_eval import (
    EvidencePredicate,
    RetrievalCase,
    answer_evidence_metrics,
    answer_focus_metrics,
    citation_provenance_metrics,
    evidence_metrics,
    load_retrieval_dataset,
    retrieval_metrics,
)
from scripts.load_to_qdrant import build_embedding_service
from scripts.evaluate_chunking import ExperimentConfig, evaluate_collection


ROOT = Path(__file__).resolve().parents[2]


def test_golden_retrieval_dataset_has_required_coverage() -> None:
    cases = load_retrieval_dataset(ROOT / "tests" / "eval" / "retrieval_dataset.json")

    assert len(cases) == 29
    assert all(1 <= len(case.relevant_doc_ids) <= 3 for case in cases)
    assert any(len(case.relevant_doc_ids) == 3 for case in cases)
    assert any("AUTO-SOLID" in case.question for case in cases)
    assert any(case.history for case in cases)
    assert any(predicate.printed_page == 45 for case in cases for predicate in case.relevant_evidence)


def test_russian_sentence_tokenizer_preserves_boundaries() -> None:
    assert russian_sentence_tokenizer("Первое предложение. Второе? Third!") == [
        "Первое предложение. ",
        "Второе? ",
        "Third!",
    ]


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_all_chunking_strategies_return_nodes(strategy: str) -> None:
    documents = [
        Document(
            text=("Первое предложение о сетке. Второе предложение о материале. " * 20),
            metadata={"file_name": "manual.pdf"},
        )
    ]
    nodes = chunk_documents(
        strategy,
        documents,
        chunk_size=64,
        chunk_overlap=8,
        embed_model=MockEmbedding(embed_dim=8),
    )

    assert nodes
    assert all(node.get_content().strip() for node in nodes)
    assert all(node.metadata["file_name"] == "manual.pdf" for node in nodes)


def test_semantic_chunking_requires_embedding_model() -> None:
    with pytest.raises(ValueError, match="embed_model"):
        chunk_documents("semantic", [Document(text="Текст. Ещё текст.")])


def test_offline_embedding_handles_punctuation_only_text() -> None:
    service = build_embedding_service(8, offline=True)

    vector = service.embed_texts(["---"])[0]

    assert len(vector) == 8
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_retrieval_metrics_use_document_level_golden_labels() -> None:
    cases = [
        RetrievalCase("q1", ("a.pdf",)),
        RetrievalCase("q2", ("b.pdf", "c.pdf")),
    ]
    rankings = [
        ["x.pdf", "a.pdf"],
        ["b.pdf", "x.pdf", "c.pdf"],
    ]

    metrics = retrieval_metrics(cases, rankings)

    assert metrics == {"hit_rate@5": 1.0, "mrr@10": 0.75, "recall@10": 1.0}


def test_answer_focus_metrics_reject_adjacent_topic_leakage() -> None:
    case = RetrievalCase(
        "Какие степени свободы используются для плоской деформации?",
        ("manual.pdf",),
        expected_answer_excludes_all=("угол поворота", "ориентация материала"),
    )

    assert answer_focus_metrics([case], ["Используются перемещения по осям x и y."]) == {
        "answer_focus_rate": 1.0
    }
    assert answer_focus_metrics(
        [case],
        ["Используются x и y; ориентация материала задаётся через угол поворота."],
    ) == {"answer_focus_rate": 0.0}


def test_evidence_metrics_reject_wrong_chunk_from_the_right_document_and_page() -> None:
    doc_id = "03_Пособие_GTS_NX_Элементы.pdf"
    case = RetrievalCase(
        "Какие степени свободы используются?",
        (doc_id,),
        relevant_evidence=(
            EvidencePredicate(
                doc_id,
                ("элемент плоской деформации", "осей x и y", "элементной системы координат"),
                page=32,
                printed_page=45,
            ),
        ),
        expected_answer_contains_all=("оси x", "оси y"),
    )
    wrong_same_document = {
        "doc_id": doc_id,
        "text": "Ориентация материала элемента плоской деформации задаётся углом.",
        "page": 60,
        "printed_page": 79,
    }
    correct = {
        "doc_id": doc_id,
        "text": (
            "Элемент плоской деформации характеризуется степенями свободы "
            "по направлениям осей x и y элементной системы координат."
        ),
        "page": 32,
        "printed_page": 45,
    }

    assert evidence_metrics([case], [[wrong_same_document]]) == {
        "evidence_hit_rate@5": 0.0,
        "evidence_mrr@10": 0.0,
        "evidence_recall@10": 0.0,
    }
    assert evidence_metrics([case], [[wrong_same_document, correct]]) == {
        "evidence_hit_rate@5": 1.0,
        "evidence_mrr@10": 0.5,
        "evidence_recall@10": 1.0,
    }
    assert answer_evidence_metrics([case], ["Используются оси x и оси y [1]."]) == {
        "answer_evidence_rate": 1.0
    }
    assert citation_provenance_metrics([case], [[wrong_same_document]]) == {
        "citation_provenance_rate": 0.0
    }
    assert citation_provenance_metrics([case], [[correct]]) == {
        "citation_provenance_rate": 1.0
    }


def test_chunking_experiment_reports_page_aware_evidence_metrics() -> None:
    doc_id = "03_Пособие_GTS_NX_Элементы.pdf"
    case = RetrievalCase(
        "Какие степени свободы используются для плоской деформации?",
        (doc_id,),
        relevant_evidence=(
            EvidencePredicate(
                doc_id,
                ("элемент плоской деформации", "осей x и y"),
                page=32,
                printed_page=45,
            ),
        ),
    )

    class FakeEvaluationClient:
        def query_points(self, **_kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=0.95,
                        payload={
                            "text": "Ориентация материала элемента плоской деформации.",
                            "source_doc_id": doc_id,
                            "source": doc_id,
                            "page": 60,
                            "printed_page": 79,
                        },
                    ),
                    SimpleNamespace(
                        score=0.90,
                        payload={
                            "text": (
                                "Элемент плоской деформации характеризуется степенями свободы "
                                "по направлениям осей x и y."
                            ),
                            "source_doc_id": doc_id,
                            "source": doc_id,
                            "page": 32,
                            "printed_page": 45,
                        },
                    ),
                ]
            )

    metrics = evaluate_collection(
        FakeEvaluationClient(),
        [case],
        [[1.0, 0.0]],
        ExperimentConfig("test", "recursive", "test", 512, 64, 10),
    )

    assert metrics["hit_rate@5"] == 1.0
    assert metrics["evidence_hit_rate@5"] == 1.0
    assert metrics["evidence_mrr@10"] == 0.5
    assert metrics["evidence_recall@10"] == 1.0


def test_bge_reranker_resorts_candidates_and_limits_top_n() -> None:
    model = SimpleNamespace(predict=lambda pairs, **_kwargs: [0.1, 0.9, 0.4])
    reranker = BGEReranker(model=model)
    candidates = [
        RerankCandidate(text=f"text-{index}", doc_id=f"doc-{index}", source="source", score=1 - index / 10)
        for index in range(3)
    ]

    result = reranker.rerank("query", candidates, top_n=2)

    assert [item.doc_id for item in result] == ["doc-1", "doc-2"]
    assert [item.rerank_score for item in result] == [0.9, 0.4]
