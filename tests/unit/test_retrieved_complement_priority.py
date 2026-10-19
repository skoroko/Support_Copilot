"""Selection boundaries for independently cited, already retrieved complements."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import RAGConfig, RAGService, RetrievalPlan


def node(identity, text=None, *, location=None, score=.8, **metadata):
    return NodeWithScore(node=TextNode(id_=identity, text=text or f"Evidence {identity}.", metadata={
        "file_name": "manual.txt", "source_id": "manual", "source_hash": "a" * 64,
        "location_id": location or identity, "product": "gts_nx", "version": "2021",
        "block_id": identity, "block_ids": [identity], "atomic_evidence": True, **metadata,
    }), score=score)


def select(tmp_path: Path, pool, *, structural=False, structural_limit=8, structural_text="Linked evidence."):
    rag = RAGService(RAGConfig(
        data_dir=tmp_path, qdrant_url="http://unused:6333", qdrant_api_key=None,
        collection_name="test", baremetal_collection_name="test_bare", llm_model="unused",
        embedding_model="unused", embedding_dimensions=3, allow_external_data_transfer=True,
        product_scope="all", condense_enabled=False, docstore_path=tmp_path / "missing.json",
    ))
    # Isolate selection from ranking: this is the original complete-ranked order.
    rag._fuse_retrieval_candidates = lambda rankings: list(rankings[0])
    rag._rank_by_lexical_evidence = lambda question, items: list(items)
    calls = []
    structural_selected = 0

    def resolve(metadata, text, *, question, max_blocks, max_chars, diagnostics):
        nonlocal structural_selected
        calls.append((max_blocks, max_chars))
        if not structural:
            return []
        result = []
        for index in range(min(max_blocks, structural_limit-structural_selected)):
            identity = f"structural-{len(calls)}-{index}"
            item = node(identity, structural_text, origin="structural_relation")
            if sum(len(row["text"]) for row in result) + len(item.node.text) > max_chars:
                break
            result.append({"block_id": identity, "text": item.node.text, "metadata": item.node.metadata})
        structural_selected += len(result)
        return result

    rag._related_evidence = SimpleNamespace(resolve=resolve, source_relative_path=lambda _: None)
    result = rag._select_context(
        question="explain", plan=RetrievalPlan("explain", ("explain",), "independent"),
        rankings=[pool], lexical_candidates=[], lexical_query=None, overview_question=False,
        preserve_full_context=True,
    )
    return result, calls


def seeds():
    return [node(f"seed-{index}", score=.95-index*.01) for index in range(5)]


def ids(result):
    return [item.node.node_id for item in result.ranked]


def test_existing_same_location_and_structural_prefix_precedes_idle_fill(tmp_path):
    initial = seeds()
    other = node("other", related_to=["unverified-old-link"])
    same = node("same", location="seed-0")
    result, calls = select(tmp_path, [*initial, other, same], structural=True, structural_limit=1)
    assert ids(result) == [*[item.node.node_id for item in initial], "same", "structural-1-0", "other"]
    assert calls[0] == (7, 12_000-len(same.node.text))
    source = result.sources[7]
    assert source["origin"] == "retrieved_other_location"
    assert source["related_to"] == []
    assert source["location_id"] == "other"
    assert source["source_hash"] == other.node.metadata["source_hash"]
    assert result.retrieved_contexts[7] == other.node.text
    assert result.ranked[7].score == other.score
    assert result.sources[5]["related_to"] == ["seed-0"]
    reports = result.related_evidence_diagnostics
    assert reports[0]["origin"] == "retrieved_same_location"
    assert reports[-1]["origin"] == "retrieved_other_location"
    assert result.top_score == result.selected_top_score == .95


def test_saturating_existing_structural_additions_cannot_be_displaced(tmp_path):
    initial = seeds()
    same = node("same", location="seed-0")
    result, calls = select(tmp_path, [*initial, node("other"), same], structural=True)
    assert ids(result) == [*[item.node.node_id for item in initial], "same",
                           *[f"structural-1-{index}" for index in range(7)]]
    assert len(result.ranked) == 13
    assert calls[0] == (7, 12_000-len(same.node.text))
    assert all(source["origin"] != "retrieved_other_location" for source in result.sources)


def test_shared_count_and_other_location_limit_leave_final_thirteen(tmp_path):
    same = node("same", location="seed-0")
    extras = [node(f"same-other-{index}", location="other-page") for index in range(3)]
    extras.extend(node(f"other-{index}") for index in range(6))
    result, calls = select(tmp_path, [*seeds(), same, *extras])
    assert ids(result)[5:] == ["same", "same-other-0", "same-other-1", *[f"other-{i}" for i in range(5)]]
    assert len(result.ranked) == 13
    assert calls[0][0] == 7
    report = next(row for row in result.related_evidence_diagnostics if row.get("origin") == "retrieved_other_location")
    assert report["bounded"]
    assert {row["chunk_id"] for row in report["omitted"]} == {"same-other-2", "other-5"}


def test_shared_char_budget_omits_whole_other_location_block(tmp_path):
    same = node("same", "S" * 5_000, location="seed-0")
    other = node("other", "O" * 7_000)
    result, calls = select(tmp_path, [*seeds(), same, other, node("overflow")])
    assert ids(result)[5:] == ["same", "other"]
    assert result.retrieved_contexts[5:] == [same.node.text, other.node.text]
    assert calls[0] == (7, 7_000)


def test_existing_structural_chars_are_reserved_before_other_location(tmp_path):
    other = node("other", "O" * 7_001)
    result, _ = select(tmp_path, [*seeds(), other], structural=True,
                       structural_limit=1, structural_text="S" * 5_000)
    assert ids(result)[5:] == ["structural-1-0"]
    assert result.retrieved_contexts[5:] == ["S" * 5_000]


@pytest.mark.parametrize("blocks,retained", [(["structural-1-0"], False),
                                           (["structural-1-0", "new-fragment"], True)])
def test_structural_seen_blocks_skip_only_fully_represented_candidates(tmp_path, blocks, retained):
    other = node("other", block_ids=blocks)
    result, _ = select(tmp_path, [*seeds(), other], structural=True, structural_limit=1)
    assert ("other" in ids(result)) == retained


def test_complements_cannot_promote_weak_evidence(tmp_path):
    pool = [node(f"weak-{index}", score=.1) for index in range(8)]
    result, calls = select(tmp_path, pool, structural=True)
    assert not result.has_evidence
    assert result.top_score == result.selected_top_score == .1
    assert ids(result) == [f"weak-{index}" for index in range(5)]
    assert not calls


def test_other_location_requires_positive_source_binding(tmp_path):
    result, _ = select(tmp_path, [*seeds(), node("unbound", source_hash="")])
    assert "unbound" not in ids(result)


@pytest.mark.parametrize("difference", [{}, {"version": "2022"}, {"product": "civil"},
                                         {"source_hash": "b" * 64}])
def test_exact_text_dedupe_preserves_distinct_provenance(tmp_path, difference):
    initial = seeds()
    duplicate = node("equal-text", initial[0].node.text, **difference)
    result, _ = select(tmp_path, [*initial, duplicate])
    assert ("equal-text" in ids(result)) == bool(difference)


def test_explicit_shared_origin_deduplicates_equal_text(tmp_path):
    initial = seeds()
    initial[0].node.metadata["canonical_origin"] = "common-origin"
    duplicate = node("equal-text", initial[0].node.text, source_hash="b" * 64, canonical_origin="common-origin")
    result, _ = select(tmp_path, [*initial, duplicate])
    assert "equal-text" not in ids(result)
