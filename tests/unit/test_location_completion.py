"""Physical completion is literal, scoped, atomic, and bounded."""
import hashlib
import json

import pytest

from app.services.related_evidence import RelatedEvidenceIndex


def fixture(tmp_path):
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"original source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    location = {"source_id": "s", "source_hash": digest, "location_id": "p1", "page": 1,
                "product": "civil", "version": "2020"}
    blocks = [{"source_id": "s", "location_id": "p1", "block_id": "seed", "kind": "paragraph",
               "ordinal": 1, "text": "Exact introduction.", "relations": []}]
    blocks += [{"source_id": "s", "location_id": "p1", "block_id": f"fact{i}",
                "kind": "formula_candidate", "ordinal": i + 2, "text": f"DOF {i} = literal θ direction.",
                "relations": []} for i in range(1, 7)]
    blocks += [{"source_id": "s", "location_id": "p2", "block_id": "other", "kind": "paragraph",
                "ordinal": 1, "text": "Never infer page continuation."}]
    catalog = {"sources": [{"source_id": "s", "source_hash": digest, "relative_path": "guide.pdf", "format": "pdf"}],
               "locations": [location, {**location, "location_id": "p2", "page": 2}], "blocks": blocks}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return RelatedEvidenceIndex(tmp_path, path), {**location, "block_id": "seed"}, source, path, catalog


def test_full_location_discovers_six_unretrieved_atomic_facts_with_stable_identity(tmp_path):
    resolver, seed, _, _, _ = fixture(tmp_path)
    report = {}
    group = resolver.location_context(seed, "Exact introduction.", diagnostics=report)
    assert group == resolver.location_context(seed, "Exact introduction.")
    assert group["text"].count("DOF") == 6
    assert "Never infer" not in group["text"]
    meta = group["metadata"]
    assert meta["block_ids"] == ["seed", *[f"fact{i}" for i in range(1, 7)]]
    assert meta["source_locator"] == "guide.pdf#page=1"
    assert (meta["product"], meta["version"], meta["origin"]) == ("civil", "2020", "physical_location")
    assert meta["atomic_evidence"] and meta["block_attributes"]["complete_text_location"]
    assert not ({"relation", "related_to"} & meta.keys())
    assert meta["text_sha256"] == hashlib.sha256(group["text"].encode()).hexdigest()
    assert report["selected_blocks"] == 7


def test_whole_group_budget_no_prefix_and_no_oversized_chm_fallback(tmp_path):
    resolver, seed, _, path, catalog = fixture(tmp_path)
    length = len(resolver.location_context(seed, "Exact introduction.")["text"])
    assert resolver.location_context(seed, "Exact introduction.", max_chars=length)
    report = {}
    assert resolver.location_context(seed, "Exact introduction.", max_chars=length-1, diagnostics=report) is None
    assert report["reason"] == "character_budget" and report["required_chars"] == length
    catalog["blocks"][0]["text"] += " " * 7000
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.location_context(seed, "Exact introduction.") is None


@pytest.mark.parametrize("change", ["source", "hash", "location", "text", "scope", "catalog"])
def test_invalid_or_stale_binding_fails_closed(tmp_path, change):
    resolver, seed, source, path, _ = fixture(tmp_path)
    assert resolver.location_context(seed, "Exact introduction.")
    text = "Exact introduction."
    if change == "source":
        source.write_bytes(b"changed bytes")
    elif change == "catalog":
        path.write_text("invalid", encoding="utf-8")
    elif change == "hash":
        seed["source_hash"] = "wrong"
    elif change == "location":
        seed["location_id"] = "p2"
    elif change == "scope":
        seed["product"] = "other"
    else:
        text = "Invented text"
    assert resolver.location_context(seed, text) is None


def test_visuals_unsafe_grids_pending_and_parent_containers_are_not_literal_evidence(tmp_path):
    resolver, seed, _, path, catalog = fixture(tmp_path)
    extras = [("image", {}), ("visual_unextracted", {}), ("figure", {}),
              ("table", {}), ("table_row", {}), ("section", {}),
              ("paragraph", {"review_status": "pending_review"}),
              ("formula_candidate", {"review_status": "rejected"})]
    for i, (kind, attrs) in enumerate(extras):
        catalog["blocks"].append({"source_id": "s", "location_id": "p1", "block_id": f"omit{i}",
                                  "kind": kind, "ordinal": 20+i, "text": "Unsafe duplicate parent text.", "attributes": attrs})
    path.write_text(json.dumps(catalog), encoding="utf-8")
    report = {}
    group = resolver.location_context(seed, "Exact introduction.", diagnostics=report)
    assert "Unsafe" not in group["text"]
    assert len(report["omitted"]) == len(extras)
    assert group["metadata"]["block_attributes"]["untranscribed_visual_regions_present"]


def test_catalog_change_invalidates_identity_and_excess_member_count_is_atomic(tmp_path):
    resolver, seed, _, path, catalog = fixture(tmp_path)
    before = resolver.location_context(seed, "Exact introduction.")
    catalog["blocks"][1]["text"] = "Changed literal fact."
    path.write_text(json.dumps(catalog), encoding="utf-8")
    after = resolver.location_context(seed, "Exact introduction.")
    assert before["block_id"] != after["block_id"]
    for i in range(65):
        catalog["blocks"].append({"source_id": "s", "location_id": "p1", "block_id": f"extra{i}",
                                  "kind": "paragraph", "ordinal": 30+i, "text": "x"})
    path.write_text(json.dumps(catalog), encoding="utf-8")
    report = {}
    assert resolver.location_context(seed, "Exact introduction.", diagnostics=report) is None
    assert report["reason"] == "block_budget"


def test_literal_formula_candidates_keep_symbols_and_review_limitations(tmp_path):
    resolver, seed, _, path, catalog = fixture(tmp_path)
    catalog["blocks"][1]["attributes"] = {"visual_review_required": True, "verified": False,
                                         "transcription": "source_text_only"}
    catalog["locations"][0]["figure_regions"] = [{"bbox": [0, 0, 1, 1]}]
    path.write_text(json.dumps(catalog), encoding="utf-8")
    group = resolver.location_context(seed, "Exact introduction.")
    assert "DOF 1 = literal θ direction." in group["text"]
    attrs = group["metadata"]["block_attributes"]
    assert attrs["extraction_only"] and attrs["visual_review_required"]
    assert attrs["untranscribed_visual_regions_present"] and attrs["visual_region_count"] == 1
    assert "verified" not in attrs
    group["metadata"]["figure_regions"].clear()
    assert resolver.location_context(seed, "Exact introduction.")["metadata"]["figure_regions"]


def test_dom_rows_are_allowed_only_after_span_and_parent_validation(tmp_path):
    from app.services.corpus_html import parse_html

    source = tmp_path / "grid.html"
    source.write_text('<html><title>Literal</title><table><tr><td>A</td><td>B</td></tr>'
                      '<tr><td rowspan="2">α</td><td>O</td></tr><tr><td>X</td></tr></table></html>', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    parsed = parse_html(source.read_bytes(), "loc")
    location = {"location_id": "loc", "source_id": "s", "source_hash": digest,
                "chm_member": "/grid.html", "product": "civil", "version": "unknown"}
    catalog = {"sources": [{"source_id": "s", "source_hash": digest, "relative_path": source.name, "format": "html"}],
               "locations": [location], "blocks": [{**b, "source_id": "s", "location_id": "loc"} for b in parsed["blocks"]]}
    path = tmp_path / "dom.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    resolver = RelatedEvidenceIndex(tmp_path, path)
    row = next(b for b in catalog["blocks"] if b["kind"] == "table_row")
    seed = {**location, "block_id": row["block_id"]}
    group = resolver.location_context(seed, row["text"])
    assert group["text"].count("Column 1=α") == 2
    assert "Column 1=A | Column 2=B" in group["text"]
    row["attributes"]["resolved_cells"][0]["text"] = "invented"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    group = resolver.location_context(seed, row["text"])
    assert group is None or "Column 1=" not in group["text"]
