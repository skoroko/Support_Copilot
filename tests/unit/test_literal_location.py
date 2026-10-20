"""Optional literal completion preserves old evidence and never infers links."""
import hashlib
import json
from copy import deepcopy

import pytest

from app.services.literal_location import complete_selected_atoms
from app.services.related_evidence import RelatedEvidenceIndex


def fixture(tmp_path):
    path = tmp_path / "arbitrary.pdf"
    path.write_bytes(b"native PDF fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    location = {"source_id":"s", "source_hash":digest, "location_id":"p1", "page":1}
    blocks = [{"source_id":"s", "location_id":"p1", "block_id":f"a{i}", "ordinal":i,
        "kind":"formula_candidate", "text":f"Renamed variable {i} = native value {i}", "relations":[],
        "section_path":[], "bbox":[1,2,3,4],
        "attributes":{"verified":False, "visual_review_required":True, "method":"native"}}
        for i in range(6)]
    catalog = {"sources":[{"source_id":"s", "source_hash":digest, "relative_path":path.name}],
        "locations":[location, {**location,"location_id":"p2","page":2}],
        "blocks":[*blocks, {**blocks[0],"block_id":"different-page","location_id":"p2"}]}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog),encoding="utf8")
    source = {**location,"file_name":path.name,"source_locator":"arbitrary.pdf#page=1",
        "score":.9,"confident":True,"snippet":blocks[2]["text"],
        "id":"a2","chunk_id":"a2","document_id":"legacy-parent","block_id":"a2",
        "block_type":"formula_candidate","atomic_evidence":True,"block_attributes":blocks[2]["attributes"]}
    return RelatedEvidenceIndex(tmp_path,catalog_path), source, blocks, path, catalog_path


def test_group_preserves_every_old_text_slots_provenance_and_unverified_flags(tmp_path):
    resolver, source, atoms, _, _ = fixture(tmp_path)
    other = {"id":"other", "block_type":"paragraph"}
    old_sources, old_contexts = [source,other], [atoms[2]["text"],"Unchanged independent evidence."]
    original = deepcopy((old_sources,old_contexts))
    sources, contexts, count = complete_selected_atoms(old_sources,old_contexts,resolver,budget=1000)
    assert count == 1 and len(sources) == len(old_sources)
    assert all(before in after for before,after in zip(old_contexts,contexts))
    assert (old_sources,old_contexts) == original
    assert sources[0]["block_ids"] == [a["block_id"] for a in atoms]
    from app.schemas.rag import RAGSource
    roundtrip = RAGSource.model_validate(sources[0]).model_dump()
    assert roundtrip["block_attributes"]["literal_blocks"][2]["attributes"] == atoms[2]["attributes"]
    assert roundtrip["block_attributes"]["literal_blocks"][2]["text"] == atoms[2]["text"]
    assert sources[0]["block_attributes"]["verified"] is False
    assert sources[0]["block_attributes"]["semantic_verified"] is False
    assert sources[0]["block_attributes"]["visual_review_required"] is True
    assert sources[0]["relation"] is None and sources[0]["related_to"] == []
    assert sources[0]["source_locator"] == source["source_locator"]
    assert "different-page" not in sources[0]["block_ids"]


def test_no_headroom_or_truncated_seed_returns_original_evidence(tmp_path):
    resolver, source, atoms, _, _ = fixture(tmp_path)
    text = atoms[2]["text"]
    assert complete_selected_atoms([source],[text],resolver,budget=len(text)) == ([source],[text],0)
    assert complete_selected_atoms([source],[text[:15]],resolver,budget=1000) == ([source],[text[:15]],0)


@pytest.mark.parametrize("tamper",["hash","page","bytes","pending","large","linked","foreign_source"])
def test_invalid_binding_or_out_of_scope_completion_fails_closed(tmp_path,tamper):
    resolver, source, atoms, path, catalog_path = fixture(tmp_path)
    if tamper == "hash":source["source_hash"]="other"
    elif tamper == "page":source["location_id"]="p2"
    elif tamper == "bytes":path.write_bytes(b"changed")
    else:
        catalog=json.loads(catalog_path.read_text())
        if tamper == "pending":catalog["blocks"][0]["attributes"]["review_status"]="pending_review"
        elif tamper == "large":catalog["blocks"][0]["text"]="x"*2401
        elif tamper == "foreign_source":catalog["blocks"][0]["source_id"]="foreign"
        else:catalog["blocks"][2]["relations"]=[{"kind":"defines","to_id":"a1"}]
        catalog_path.write_text(json.dumps(catalog),encoding="utf8")
    text=atoms[2]["text"]
    assert complete_selected_atoms([source],[text],resolver,budget=10000) == ([source],[text],0)


def test_group_identity_is_stable_across_selected_constituents_and_budget_never_grows(tmp_path):
    resolver, source, atoms, _, _ = fixture(tmp_path)
    first, contexts, _=complete_selected_atoms([source],[atoms[2]["text"]],resolver,budget=1000)
    other={**source,"id":"a4","chunk_id":"a4","block_id":"a4"}
    second,_,_=complete_selected_atoms([other],[atoms[4]["text"]],resolver,budget=1000)
    assert first[0]["chunk_id"] == second[0]["chunk_id"]
    assert sum(map(len,contexts)) <= 1000


def test_group_recalculates_digest_and_only_shares_identical_aggregate_attributes(tmp_path):
    _, source, atoms, _, catalog_path = fixture(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    for block in catalog["blocks"]:
        block["attributes"]["reading_order"] = "layout_candidate_unverified"
        block["attributes"]["quality_status"] = "needs_review"
    catalog["blocks"][2]["attributes"].update(reading_order="confirmed", quality_status="good",
                                              header_status="confirmed")
    catalog_path.write_text(json.dumps(catalog), encoding="utf8")
    source.update(text_sha256="old-digest", evidence_scope={"chunk_id": "a2"})
    resolver = RelatedEvidenceIndex(tmp_path, catalog_path)
    sources, contexts, count = complete_selected_atoms([source], [atoms[2]["text"]], resolver, budget=1000)
    flags = sources[0]["block_attributes"]
    assert count == 1 and sources[0]["text_sha256"] == hashlib.sha256(contexts[0].encode()).hexdigest()
    assert "evidence_scope" not in sources[0]
    assert not {"reading_order", "quality_status", "header_status"} & flags.keys()
    assert flags["literal_blocks"][2]["attributes"]["reading_order"] == "confirmed"
    assert flags["literal_blocks"][0]["attributes"]["reading_order"] == "layout_candidate_unverified"


@pytest.mark.parametrize("span_count", [1, 2])
def test_selected_native_prose_span_can_include_all_atoms_using_only_unused_space(tmp_path, span_count):
    _, source, atoms, _, catalog_path = fixture(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    prose = [{**atoms[0], "block_id": f"p{i}", "ordinal": -2+i, "kind": "paragraph",
              "text": f"Original native prose {i}.", "attributes": {"verified": True}}
             for i in range(span_count)]
    catalog["blocks"].extend(prose)
    catalog_path.write_text(json.dumps(catalog), encoding="utf8")
    resolver = RelatedEvidenceIndex(tmp_path, catalog_path)
    text = "\n".join(b["text"] for b in prose)
    selected = {**source, "block_id": prose[0]["block_id"] if span_count == 1 else None,
                "block_ids": [b["block_id"] for b in prose], "block_type": "paragraph"}
    other_text = "Preserve independent evidence in full."
    budget = len(text) + len(other_text) + sum(len(b["text"]) + 1 for b in atoms)
    originals = ([selected, {"id": "other"}], [text, other_text])
    sources, contexts, count = complete_selected_atoms(*originals, resolver, budget=budget)
    assert count == 1 and contexts[0] == text + "\n" + "\n".join(b["text"] for b in atoms)
    assert contexts[1] == other_text and sources[1] == originals[0][1]
    assert sum(map(len, contexts)) == budget
    assert sources[0]["block_ids"] == [b["block_id"] for b in [*prose, *atoms]]
    assert sources[0]["block_attributes"]["verified"] is False
    assert complete_selected_atoms(*originals, resolver, budget=budget-1) == (*originals, 0)


@pytest.mark.parametrize("tamper", ["truncated_span", "later_caption", "gap"])
def test_completion_never_reorders_or_skips_unseen_native_text(tmp_path, tamper):
    _, source, atoms, _, catalog_path = fixture(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    prose = [{**atoms[0], "block_id": f"p{i}", "ordinal": -2+i, "kind": "paragraph",
              "text": f"Original prose {i}."} for i in range(2)]
    if tamper == "later_caption":
        for block in prose:
            block["ordinal"] += 10
    elif tamper == "gap":
        catalog["blocks"].append({**prose[0], "block_id": "gap", "ordinal": -.5,
                                   "text": "A heading which was not selected."})
    catalog["blocks"].extend(prose)
    catalog_path.write_text(json.dumps(catalog), encoding="utf8")
    resolver = RelatedEvidenceIndex(tmp_path, catalog_path)
    text = prose[0]["text"] if tamper == "truncated_span" else "\n".join(b["text"] for b in prose)
    selected = {**source, "block_id": None, "block_ids": [b["block_id"] for b in prose],
                "block_type": "paragraph"}
    assert complete_selected_atoms([selected], [text], resolver, budget=1000) == ([selected], [text], 0)


def member_fixture(tmp_path):
    path = tmp_path / "renamed.chm"
    path.write_bytes(b"native help fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    location = {"source_id": "s", "source_hash": digest, "location_id": "article", "chm_member": "arbitrary.htm"}
    contents = [("paragraph", "Home > Arbitrary parameter"), ("link", "Home"),
                ("paragraph", "Overview"), ("paragraph", "The documented native value is zero."),
                ("image", "Untranscribed figure"), ("paragraph", "A later explanation is not appended.")]
    blocks = [{"source_id": "s", "location_id": "article", "block_id": f"b{i}", "ordinal": i,
               "kind": kind, "text": text, "relations": [], "section_path": [],
               "attributes": {"method": "native_html", "verified": False}}
              for i, (kind, text) in enumerate(contents)]
    catalog = {"sources": [{"source_id": "s", "source_hash": digest, "relative_path": path.name}],
               "locations": [location], "blocks": blocks}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf8")
    source = {**location, "block_type": "paragraph", "block_id": "b0", "block_ids": ["b0"],
              "id": "b0", "source_locator": "renamed.chm::arbitrary.htm"}
    return source, blocks, path, catalog_path, catalog


def test_short_help_member_opening_reads_contiguous_text_before_image_without_displacement(tmp_path):
    source, blocks, _, catalog_path, _ = member_fixture(tmp_path)
    resolver = RelatedEvidenceIndex(tmp_path, catalog_path)
    original = blocks[0]["text"]
    expected = "\n".join(b["text"] for b in blocks[:4])
    other = "Other evidence is preserved in full."
    sources, contexts, count = complete_selected_atoms([source, {"id": "other"}], [original, other],
                                                      resolver, budget=len(expected) + len(other))
    assert count == 1 and contexts == [expected, other]
    assert sources[0]["block_ids"] == ["b0", "b1", "b2", "b3"]
    assert sources[0]["block_attributes"]["verification_scope"] == "literal_member_prefix"
    assert sources[0]["block_attributes"]["verified"] is False
    assert sources[0]["related_to"] == [] and sources[0]["source_locator"] == source["source_locator"]
    assert complete_selected_atoms([source], [original], resolver, budget=len(expected)-1) == ([source], [original], 0)


@pytest.mark.parametrize("tamper", ["bytes", "hash", "foreign", "middle", "truncated", "large", "pending"])
def test_help_member_completion_requires_exact_short_current_opening(tmp_path, tamper):
    source, blocks, path, catalog_path, catalog = member_fixture(tmp_path)
    text = blocks[0]["text"]
    if tamper == "bytes":
        path.write_bytes(b"changed")
    elif tamper == "hash":
        source["source_hash"] = "changed"
    elif tamper == "foreign":
        catalog["blocks"][3]["source_id"] = "other"
    elif tamper == "middle":
        source.update(block_id="b2", block_ids=["b2"])
        text = blocks[2]["text"]
    elif tamper == "truncated":
        text = text[:10]
    elif tamper == "large":
        catalog["blocks"][5]["text"] = "x" * 2401
    else:
        catalog["blocks"][0]["attributes"]["review_status"] = "pending_review"
    catalog_path.write_text(json.dumps(catalog), encoding="utf8")
    resolver = RelatedEvidenceIndex(tmp_path, catalog_path)
    assert complete_selected_atoms([source], [text], resolver, budget=12000) == ([source], [text], 0)
