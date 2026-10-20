"""Saved full evidence must survive the raw/API to compact/model boundary."""
import hashlib
import json
from copy import deepcopy

import pytest
from langchain_core.messages import ToolMessage

from app.services.agent_persistent import _citation_record, _saved_citation_evidence
from app.services.prepared_evidence import evidence_identity, source_for_model


def source():
    return {"id": "chunk-A", "chunk_id": "chunk-A", "document_id": "C:/private/data/manual.pdf",
            "file_name": "manual.pdf", "source_locator": "manual.pdf#page=3", "page": 3,
            "product": "product-A", "version": "unknown", "source_id": "source-A",
            "source_hash": "hash-A", "location_id": "location-A", "block_id": "block-A",
            "block_bbox": [1., 2., 3., 4.], "snippet": "Display only.",
            "figure_regions": [{"region_id": "untranscribed"}],
            "block_attributes": {"glyphs": [{"text": "x"}], "raw_char_text": "bulky",
                                 "verification_scope": "literal_source_text", "semantic_verified": False}}


def test_full_saved_evidence_survives_consistent_raw_compact_identity():
    raw = source()
    full = "Full exact evidence beyond display.\nComplete table row | condition | units."
    record = _citation_record(raw, full, 1)
    restored = _saved_citation_evidence({"answer_evidence": [raw],
        "citation_evidence": [record], "messages": []})
    assert restored[0]["text"] == full
    assert "text_sha256" not in restored[0]
    assert evidence_identity(raw) == evidence_identity(record["source"])


@pytest.mark.parametrize(("field", "value"), [
    ("chunk_id", "chunk-B"), ("source_id", "source-B"), ("source_hash", "hash-B"),
    ("file_name", "other.pdf"), ("source_locator", "manual.pdf#page=9"),
    ("page", 4), ("printed_page", "4-1"), ("product", "product-B"), ("version", "2024"),
    ("section", "other-section"), ("chm_member", "/different/member.htm"),
])
def test_distinct_evidence_binding_does_not_borrow_saved_text(field, value):
    first, second = source(), source()
    second[field] = value
    records = [_citation_record(first, "Complete first evidence.", 1),
               _citation_record(second, "Complete second evidence.", 2)]
    restored = _saved_citation_evidence({"answer_evidence": [first, second],
        "citation_evidence": records, "messages": []})
    assert [record["text"] for record in restored] == [
        "Complete first evidence.", "Complete second evidence."]
    assert evidence_identity(first) != evidence_identity(second)


def test_model_source_projection_stays_compact_idempotent_and_does_not_mutate_raw():
    raw = source()
    original = deepcopy(raw)
    compact = source_for_model(raw)
    assert source_for_model(compact) == compact
    assert raw == original
    binding = json.dumps(["corpus-v1", raw["source_id"], raw["source_hash"]], ensure_ascii=False)
    assert compact["document_key"] == hashlib.sha256(binding.encode()).hexdigest()[:16]
    for dropped in ("document_id", "source_id", "source_hash", "location_id", "block_id",
                    "block_bbox", "text_sha256", "snippet", "figure_regions"):
        assert dropped not in compact
    assert "glyphs" not in compact["block_attributes"]
    assert "raw_char_text" not in compact["block_attributes"]
    assert compact["block_attributes"]["semantic_verified"] is False
    assert compact["untranscribed_visual_regions_present"] is True


@pytest.mark.parametrize("compact", [False, True])
def test_legacy_tool_evidence_recovers_full_text_with_raw_or_compact_source(compact):
    raw = source()
    saved_source = source_for_model(raw) if compact else raw
    message = ToolMessage(name="search_knowledge_base", tool_call_id="legacy-search",
        content=json.dumps({"evidence": [{"source": saved_source, "text": "Full legacy evidence."}]}))
    restored = _saved_citation_evidence({"answer_evidence": [raw],
        "citation_evidence": [], "messages": [message]})
    assert restored[0]["text"] == "Full legacy evidence."


def test_document_id_only_legacy_distinction_is_preserved_without_portable_provenance():
    first = {"id": "legacy-reused", "chunk_id": "legacy-reused", "file_name": "manual.pdf",
             "page": 3, "product": "product-A", "version": "unknown",
             "document_id": "C:/legacy-A/manual.pdf", "snippet": "First display."}
    second = {**first, "document_id": "C:/legacy-B/manual.pdf", "snippet": "Second display."}
    records = [_citation_record(first, "Complete first legacy document.", 1),
               _citation_record(second, "Complete second legacy document.", 2)]
    assert evidence_identity(first) != evidence_identity(second)
    assert evidence_identity(source_for_model(first)) != evidence_identity(source_for_model(second))
    restored = _saved_citation_evidence({"answer_evidence": [first, second],
        "citation_evidence": records, "messages": []})
    assert [record["text"] for record in restored] == [
        "Complete first legacy document.", "Complete second legacy document."]


def test_direct_and_related_routes_share_identity_and_keep_unique_text():
    from app.services.prepared_evidence import prepare_evidence
    direct = source()
    related = {**direct, "document_id": direct["chunk_id"], "origin": "related"}
    assert evidence_identity(direct) == evidence_identity(related)
    assert evidence_identity(source_for_model(direct)) == evidence_identity(related)
    packet = prepare_evidence({"status": "ok", "sources": [direct, related],
        "contexts": ["Exact definition.", "Exact definition."]}, query="definition")
    assert len(packet["citation_evidence"]) == 1
    assert packet["citation_evidence"][0]["text"] == "Exact definition."
    restored = _saved_citation_evidence({"answer_evidence": [direct],
        "citation_evidence": [_citation_record(related, "All exact evidence.", 1)], "messages": []})
    assert restored[0]["text"] == "All exact evidence."


@pytest.mark.parametrize("missing", ["source_id", "source_hash"])
def test_partial_corpus_binding_keeps_legacy_document_distinction(missing):
    first = source()
    first.pop(missing)
    second = {**first, "document_id": "other-document"}
    assert evidence_identity(first) != evidence_identity(second)


def old_compact(raw):
    # Explicit pre-deploy representation: do not construct its key with the
    # new implementation under test.
    compact = source_for_model(raw)
    compact["document_key"] = hashlib.sha256(raw["document_id"].encode("utf8")).hexdigest()[:16]
    return compact


@pytest.mark.parametrize("tool_only", [False, True])
def test_predeploy_compact_checkpoint_recovers_full_text(tool_only):
    raw = source()
    record = {"source": old_compact(raw), "text": "FULL EXACT EVIDENCE WITH CONDITIONS"}
    state = {"answer_evidence": [raw], "citation_evidence": [] if tool_only else [record],
             "messages": [ToolMessage(name="search_knowledge_base", tool_call_id="old",
                                       content=json.dumps({"evidence": [record]}))] if tool_only else []}
    assert _saved_citation_evidence(state)[0]["text"] == record["text"]


@pytest.mark.parametrize("field", ["chunk_id", "file_name", "page", "product", "version", "source_locator"])
def test_predeploy_alias_requires_every_other_identity_field(field):
    raw = source()
    compact = old_compact(raw)
    compact[field] = "different"
    state = {"answer_evidence": [raw], "citation_evidence": [{"source": compact, "text": "WRONG"}]}
    assert _saved_citation_evidence(state)[0]["text"] == raw["snippet"]


def test_predeploy_alias_does_not_choose_between_ambiguous_revisions_or_override_current():
    raw = source()
    old = {"source": old_compact(raw), "text": "Old full text"}
    other_revision = {**raw, "source_hash": "other-revision"}
    assert [r["text"] for r in _saved_citation_evidence({
        "answer_evidence": [raw, other_revision], "citation_evidence": [old]})] == [raw["snippet"]] * 2
    current = _citation_record(raw, "Current full text", 1)
    for records in ([old, current], [current, old]):
        assert _saved_citation_evidence({"answer_evidence": [raw], "citation_evidence": records})[0]["text"] == "Current full text"
