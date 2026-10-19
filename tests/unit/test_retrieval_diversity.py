from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import RAGConfig, RAGService


def service(tmp_path: Path) -> RAGService:
    return RAGService(RAGConfig(data_dir=tmp_path, qdrant_url="http://unused:6333", qdrant_api_key=None,
        collection_name="test", baremetal_collection_name="test_bare", llm_model="unused",
        embedding_model="unused", embedding_dimensions=3, allow_external_data_transfer=True,
        product_scope="all", condense_enabled=False, docstore_path=tmp_path / "missing.json"))


def node(identity: str, text: str, score: float, *, location: str | None = None,
         suffix: str = ".pdf", version: str = "unknown", source_hash: str = "a" * 64,
         atomic: bool = False) -> NodeWithScore:
    location = location or identity
    return NodeWithScore(node=TextNode(id_=identity, text=text, metadata={
        "file_name": "manual" + suffix, "source_id": "manual", "source_hash": source_hash,
        "location_id": location, "page": None if suffix != ".pdf" else 1,
        "chm_member": "/member.htm" if suffix == ".chm" else None,
        "product": "test", "version": version, "extraction_version": "corpus-v1",
        "block_id": identity, "block_ids": [identity], "atomic_evidence": atomic,
    }), score=score)


def test_lexical_union_searches_later_object_and_records_query_lineage(tmp_path: Path) -> None:
    rag = service(tmp_path)
    first = node("alpha", "ALPHA MATRIX uses a triangular arrangement.", 0.)
    second = node("beta", "BETA VECTOR uses a circular arrangement.", 0.)
    rag._retriever = SimpleNamespace(retrieve=lambda _: [node("dense", "Other documented settings.", .8)])
    rag._lexical_available = True
    queries = ("alpha matrix beta vector", "alpha matrix", "beta vector")
    rag._lexical_queries = lambda *_: queries
    calls = []

    def lexical(query):
        calls.append(query)
        return {queries[0]: [], queries[1]: [first], queries[2]: [second]}[query]

    rag._retrieve_lexical_candidates = lexical
    result = rag.retrieve("Compare ALPHA MATRIX and BETA VECTOR")
    assert calls == list(queries)
    assert {"alpha", "beta"} <= {source["chunk_id"] for source in result["sources"]}
    reports = result["telemetry"]["lexical_queries"]
    assert [row["candidate_count"] for row in reports] == [0, 1, 1]
    assert reports[1]["selected_candidate_ids"] == ["alpha"]
    assert reports[2]["selected_candidate_ids"] == ["beta"]
    assert len({row["query_hash"] for row in reports}) == 3
    assert all("candidate_ids" not in row for row in reports)


@pytest.mark.parametrize("suffix", [".pdf", ".html", ".chm"])
def test_complementary_chunks_keep_exact_locator_and_independent_citations(tmp_path: Path, suffix: str) -> None:
    rag = service(tmp_path)
    title = node("title", "Overview of the command.", .95, location="same", suffix=suffix)
    body = node("body", "Procedure: choose the element, apply the command, inspect the result.", .4,
                location="same", suffix=suffix)
    others = [node(f"other-{i}", f"Different context {i}.", .9-i*.01, suffix=suffix) for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, *others, body])
    result = rag.retrieve("explain")
    pairs = {source["chunk_id"]: (source, text) for source, text in zip(result["sources"], result["contexts"], strict=True)}
    assert pairs["body"][1] == body.node.text
    assert pairs["title"][1] == title.node.text
    assert pairs["body"][0]["origin"] == "retrieved_same_location"
    assert pairs["body"][0]["related_to"] == ["title"]
    assert pairs["body"][0]["location_id"] == "same"
    assert pairs["body"][0]["source_hash"] == "a" * 64
    assert pairs["body"][0]["source_locator"] == pairs["title"][0]["source_locator"]
    assert len(result["sources"]) == 8


@pytest.mark.parametrize("changed", [{"version": "2027"}, {"source_hash": "b" * 64}, {"location": "other"}])
def test_location_completion_preserves_version_hash_and_location_boundaries(tmp_path: Path, changed: dict) -> None:
    rag = service(tmp_path)
    title = node("title", "Overview.", .95, location="same")
    values = {"location": "same", **changed}
    body = node("body", "Different applicability.", .4, **values)
    others = [node(f"other-{i}", f"Context {i}.", .9-i*.01) for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, *others, body])
    result = rag.retrieve("explain")
    assert "body" not in {source["chunk_id"] for source in result["sources"]}


def test_location_diversity_defers_chunks_and_completion_is_bounded(tmp_path: Path) -> None:
    rag = service(tmp_path)
    repeated = [node(f"same-{i}", f"Distinct evidence {i}.", .95-i*.01, location="same") for i in range(9)]
    other = node("other", "Independent evidence.", .5)
    rag._retriever = SimpleNamespace(retrieve=lambda _: [*repeated, other])
    result = rag.retrieve("explain")
    ids = [source["chunk_id"] for source in result["sources"]]
    assert "other" in ids[:5]
    assert len(ids) <= 13
    assert len([source for source in result["sources"] if source["origin"] == "retrieved_same_location"]) <= 2
    reports = result["telemetry"]["related_evidence_diagnostics"]
    assert any(row.get("origin") == "retrieved_same_location" and row["bounded"] for row in reports)


def test_oversized_atomic_companion_is_omitted_with_diagnostic(tmp_path: Path) -> None:
    rag = service(tmp_path)
    title = node("title", "Overview.", .95, location="same")
    body = node("table", "Visible cell | exact value\n" * 600, .4, location="same", atomic=True)
    others = [node(f"other-{i}", f"Context {i}.", .9-i*.01) for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, *others, body])
    result = rag.retrieve("explain")
    assert "table" not in {source["chunk_id"] for source in result["sources"]}
    assert result["telemetry"]["related_evidence_truncated"]
    assert all("Visible cell" not in text for text in result["contexts"])


def test_missing_source_hash_never_joins_chunks_by_filename_alone(tmp_path: Path) -> None:
    rag = service(tmp_path)
    title = node("title", "Overview.", .95, location="same", source_hash="")
    body = node("body", "Unbound body.", .4, location="same", source_hash="")
    others = [node(f"other-{i}", f"Context {i}.", .9-i*.01) for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, *others, body])
    result = rag.retrieve("explain")
    assert "body" not in {source["chunk_id"] for source in result["sources"]}


def test_repeated_lexical_hits_merge_all_query_hashes_without_duplicate_node() -> None:
    first = node("same", "One exact passage.", 0.)
    second = node("same", "One exact passage.", 0.)
    first.node.metadata["_lexical_query_hashes"] = ["hash-one"]
    second.node.metadata["_lexical_query_hashes"] = ["hash-two"]
    merged = RAGService._merge_lexical_pools([first, second])
    assert len(merged) == 1
    assert merged[0].node.metadata["_lexical_query_hashes"] == ["hash-one", "hash-two"]
