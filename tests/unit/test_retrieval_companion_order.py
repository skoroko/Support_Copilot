from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import app.services.rag as rag_module
from app.services.rag import RAGConfig, RAGService


@pytest.fixture(autouse=True)
def companion_budget() -> None:
    """These ordering scenarios were crafted for the original 12k budget."""
    previous = rag_module.COMPANION_CONTEXT_BUDGET
    rag_module.COMPANION_CONTEXT_BUDGET = 12_000
    yield
    rag_module.COMPANION_CONTEXT_BUDGET = previous


def node(identity: str, text: str, score: float, *, location: str, suffix: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=identity, text=text, metadata={
        "file_name": "manual" + suffix, "source_id": "manual", "source_hash": "a" * 64,
        "location_id": location, "page": 1 if suffix == ".pdf" else None,
        "chm_member": "/member.htm" if suffix == ".chm" else None,
        "product": "test", "version": "2021", "extraction_version": "corpus-v1",
        "block_id": identity, "block_ids": [identity],
    }), score=score)


def service(tmp_path: Path) -> RAGService:
    return RAGService(RAGConfig(data_dir=tmp_path, qdrant_url="http://unused:6333", qdrant_api_key=None,
        collection_name="test", baremetal_collection_name="test_bare", llm_model="unused",
        embedding_model="unused", embedding_dimensions=3, allow_external_data_transfer=True,
        product_scope="all", condense_enabled=False, docstore_path=tmp_path / "missing.json"))


@pytest.mark.parametrize("suffix", [".pdf", ".html", ".chm"])
@pytest.mark.parametrize("repeat_direct_block", [False, True])
def test_direct_retrieved_complements_precede_saturating_structural_expansion(
    tmp_path: Path, suffix: str, repeat_direct_block: bool,
) -> None:
    rag = service(tmp_path)
    title = node("title", "Command overview.", .95, location="same", suffix=suffix)
    body = node("body", "Documented cause, including its applicability.", .7, location="same", suffix=suffix)
    remedy = node("remedy", "Documented remedy and conditions.", .69, location="same", suffix=suffix)
    others = [node(f"other-{i}", f"Different context {i}.", .9-i*.01, location=f"other-{i}", suffix=suffix)
              for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, body, remedy, *others])
    calls = []

    def resolve(metadata, text, *, question, max_blocks, max_chars, diagnostics):
        calls.append((max_blocks, max_chars))
        diagnostics.update(selected_blocks=max_blocks, bounded=True)
        related = [{"block_id": f"related-{len(calls)}-{i}", "text": "Structural relation text.",
                    "metadata": {**title.node.metadata, "block_id": f"related-{len(calls)}-{i}",
                                 "block_ids": [f"related-{len(calls)}-{i}"], "origin": "structural_relation"}}
                   for i in range(max_blocks)]
        if repeat_direct_block and len(calls) == 1:
            related[0] = {"block_id": "body", "text": body.node.text, "metadata": body.node.metadata}
        return related

    rag._related_evidence = SimpleNamespace(resolve=resolve, source_relative_path=lambda _: None)
    result = rag.retrieve("explain")
    pairs = {source["chunk_id"]: (source, text) for source, text in
             zip(result["sources"], result["contexts"], strict=True)}
    assert {"body", "remedy"} <= pairs.keys()
    for identity, original in (("body", body), ("remedy", remedy)):
        source, text = pairs[identity]
        assert text == original.node.text
        assert source["origin"] == "retrieved_same_location"
        assert source["related_to"] == ["title"]
        assert source["source_hash"] == "a" * 64
        assert source["location_id"] == "same"
    assert calls[0] == (6, 12_000 - len(body.node.text) - len(remedy.node.text))
    assert len(result["sources"]) == 15
    assert len([source for source in result["sources"] if source["origin"] == "structural_relation"]) == 6
    assert sum(len(text) for source, text in pairs.values() if source["origin"] in
               {"retrieved_same_location", "structural_relation"}) <= 12_000
    assert len(pairs) == len(result["sources"])


def test_direct_complement_over_shared_char_budget_is_omitted_whole(tmp_path: Path) -> None:
    rag = service(tmp_path)
    title = node("title", "Overview.", .95, location="same", suffix=".chm")
    body = node("body", "Exact table cell | value\n" * 600, .4, location="same", suffix=".chm")
    body.node.metadata["atomic_evidence"] = True
    others = [node(f"other-{i}", f"Context {i}.", .9-i*.01, location=f"other-{i}", suffix=".chm")
              for i in range(6)]
    rag._retriever = SimpleNamespace(retrieve=lambda _: [title, *others, body])
    calls = []

    def resolve(metadata, text, *, question, max_blocks, max_chars, diagnostics):
        calls.append((max_blocks, max_chars))
        return []

    rag._related_evidence = SimpleNamespace(resolve=resolve, source_relative_path=lambda _: None)
    result = rag.retrieve("explain")
    assert "body" not in {source["chunk_id"] for source in result["sources"]}
    assert calls[0] == (8, 12_000)
    reports = result["telemetry"]["related_evidence_diagnostics"]
    report = next(row for row in reports if row.get("origin") == "retrieved_same_location")
    assert report["bounded"]
    assert {"chunk_id": "body", "reason": "location_context_budget"} in report["omitted"]
