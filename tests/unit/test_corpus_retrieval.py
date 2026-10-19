from __future__ import annotations

import hashlib
import json
from threading import Barrier
from types import SimpleNamespace

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.prepared_evidence import prepare_evidence
from app.services.rag import RAGConfig, RAGService, RetrievalPlan


def service(tmp_path):
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"original")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    location = {"location_id": "page1", "source_id": "s", "source_hash": digest,
                "page": 1, "product": "gts_nx", "version": "unknown"}
    catalog = {"schema_version": 1,
               "sources": [{"source_id": "s", "source_hash": digest, "relative_path": source.name}],
               "locations": [location, {**location, "location_id": "page2", "page": 2}],
               "blocks": [
                   {"block_id": "row", "source_id": "s", "location_id": "page1", "kind": "table_row",
                    "text": "Pressure | 25", "relations": [{"kind": "table_header", "to_id": "header"}]},
                   {"block_id": "header", "source_id": "s", "location_id": "page2", "kind": "table_header",
                    "text": "Parameter | Value (kPa)", "relations": []},
               ]}
    catalog_path = tmp_path / "store.corpus.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    config = RAGConfig(data_dir=tmp_path, qdrant_url="http://unused", qdrant_api_key=None,
                       collection_name="test", baremetal_collection_name="other", llm_model="test",
                       embedding_model="test", embedding_dimensions=3, evidence_catalog_path=catalog_path,
                       condense_llm_enabled=False, lexical_retrieval_enabled=False)
    rag = RAGService(config, embed_model=MockEmbedding(embed_dim=3))
    rag._index = object()
    node = NodeWithScore(node=TextNode(id_="row", text="Pressure | 25", metadata={
        **location, "block_id": "row", "file_name": source.name, "file_path": str(source),
        "extraction_version": "corpus-v1", "atomic_evidence": True}), score=0.9)
    rag._retriever = SimpleNamespace(retrieve=lambda question: [node])
    return rag, source, node


def test_structural_evidence_has_own_page_citation_and_never_inherits_score(tmp_path):
    rag, _, _ = service(tmp_path)
    result = rag.retrieve("Pressure value?")
    assert result["contexts"] == ["Pressure | 25", "Parameter | Value (kPa)"]
    assert [item["page"] for item in result["sources"]] == [1, 2]
    assert len({item["id"] for item in result["sources"]}) == 2
    assert result["sources"][1]["score"] == 0
    assert result["sources"][1]["origin"] == "structural_relation"
    assert result["sources"][1]["source_locator"] == "manual.pdf#page=2"


def test_stale_catalog_does_not_add_companions_and_weak_seed_stays_a_miss(tmp_path):
    rag, source, node = service(tmp_path)
    source.write_bytes(b"modified")
    assert len(rag.retrieve("Pressure value?")["sources"]) == 1
    node.score = 0.01
    assert rag.retrieve("Pressure value?")["status"] == "no_evidence"


def test_independent_variants_overlap_and_success_survives_sibling_failure(tmp_path):
    rag, _, node = service(tmp_path)
    barrier = Barrier(2)

    def retrieve(query):
        barrier.wait(timeout=2)
        if query == "bad":
            raise ConnectionError("sensitive provider error body")
        return [node]

    rag._retriever = SimpleNamespace(retrieve=retrieve)
    rag._retrieval_plan = lambda *args: RetrievalPlan(
        queries=("good", "bad"), ranking_question="Pressure value?", mode="standalone")
    result = rag.retrieve("Pressure value?")
    assert result["status"] == "ok"
    assert result["telemetry"]["dense_query_count"] == 2
    assert result["telemetry"]["dense_retrieval_error_types"] == ["ConnectionError"]
    packet = prepare_evidence(result, query="Pressure value?")["packet"]
    assert "retrieval_partial_failure" in packet["limitations"]
    assert "sensitive" not in json.dumps(packet)


def test_outage_cannot_be_reported_as_absence_from_corpus(tmp_path):
    rag, _, _ = service(tmp_path)

    def fail(question):
        raise ConnectionError("provider unavailable")

    rag._retriever = SimpleNamespace(retrieve=fail)
    with pytest.raises(RuntimeError, match="unavailable or incomplete"):
        rag.retrieve("Unknown parameter?")


def test_visual_extraction_limitations_survive_preparation():
    packet = prepare_evidence({"status": "ok", "sources": [{
        "id": "1", "block_attributes": {"visual_review_required": True},
        "figure_regions": [{"bbox": [1, 2, 3, 4]}]}], "contexts": ["x = 2"]}, query="Equation?")["packet"]
    assert "unverified_formula_or_layout" in packet["limitations"]
    assert "visual_regions_not_transcribed" in packet["limitations"]
