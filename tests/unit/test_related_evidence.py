import hashlib
import json

from app.services.related_evidence import RelatedEvidenceIndex, needs_section_context


def fixture(tmp_path):
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"original bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    loc = {"source_id": "s", "source_hash": digest, "location_id": "p1", "page": 1,
           "product": "civil", "version": "unknown"}
    blocks = [
        {"source_id": "s", "location_id": "p1", "block_id": "seed", "kind": "paragraph", "ordinal": 1,
         "text": "Exact retrieved fact.", "section_path": ["One"],
         "relations": [{"kind": "table_header", "to_id": "header"}, {"kind": "next_page", "to_id": "adjacent"}]},
        {"source_id": "s", "location_id": "p1", "block_id": "header", "kind": "table_header", "ordinal": 2,
         "text": "Value | unit (kN)", "section_path": ["One"], "relations": [{"kind": "defines", "to_id": "seed"}]},
        {"source_id": "s", "location_id": "p1", "block_id": "steps", "kind": "paragraph", "ordinal": 3,
         "text": "Step one. Step two.", "section_path": ["One"]},
        {"source_id": "s", "location_id": "p1", "block_id": "other", "kind": "paragraph", "ordinal": 4,
         "text": "Unrelated topic.", "section_path": ["Two"]},
        {"source_id": "s", "location_id": "p2", "block_id": "adjacent", "kind": "paragraph", "ordinal": 5,
         "text": "Another physical page.", "section_path": ["One"]},
    ]
    catalog = {"sources": [{"source_id": "s", "relative_path": source.name, "source_hash": digest}],
               "locations": [loc, {**loc, "location_id": "p2", "page": 2}], "blocks": blocks}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return RelatedEvidenceIndex(tmp_path, path), {**loc, "block_id": "seed"}, path, source


def test_links_have_own_locators_and_no_implicit_page_or_topic_continuation(tmp_path):
    resolver, seed, _, _ = fixture(tmp_path)
    result = resolver.resolve(seed, "Exact retrieved fact.", question="What is the value?")
    assert [row["block_id"] for row in result] == ["header"]
    assert result[0]["metadata"]["source_locator"] == "guide.pdf#page=1"
    assert result[0]["metadata"]["origin"] == "structural_relation"
    assert result[0]["metadata"]["version"] == "unknown"
    assert result[0]["text"] == "Value | unit (kN)"


def test_procedure_gets_confirmed_same_section_with_budget_and_cycle_bound(tmp_path):
    resolver, seed, _, _ = fixture(tmp_path)
    result = resolver.resolve(seed, "Exact retrieved fact.", question="Как выполнить процедуру?")
    assert [row["block_id"] for row in result] == ["header", "steps"]
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор", max_chars=2) == []
    assert len(resolver.resolve(seed, "Exact retrieved fact.", question="Обзор", max_blocks=1)) == 1


def test_changed_source_invalidates_catalog_and_legacy_chunk_is_not_extended(tmp_path):
    resolver, seed, _, source = fixture(tmp_path)
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор")
    assert resolver.resolve({}, "Exact retrieved fact.", question="Обзор") == []
    source.write_bytes(b"changed source")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор") == []


def test_corrupt_catalog_cannot_reuse_stale_snapshot(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор")
    path.write_text("invalid", encoding="utf-8")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор") == []


def test_missing_or_ambiguous_span_and_path_traversal_fail_closed(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    assert resolver.resolve(seed, "Invented text", question="Обзор") == []
    catalog = json.loads(path.read_text())
    catalog["sources"][0]["relative_path"] = "../outside.pdf"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор") == []


def test_question_operation_is_independent_of_object():
    assert needs_section_context("How do I configure an unknown feature?")
    assert needs_section_context("Сравни неизвестные A и B")
    assert not needs_section_context("What does x mean?")


def test_splitter_merged_prose_is_checked_against_all_declared_blocks(tmp_path):
    resolver, seed, _, _ = fixture(tmp_path)
    seed["block_ids"] = ["seed", "steps"]
    assert resolver.resolve(seed, "Exact retrieved fact.\nStep one. Step two.", question="What is x?")
    assert resolver.resolve(seed, "Invented step.\nStep one. Step two.", question="What is x?") == []


def test_empty_structural_parent_can_connect_explicit_note(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    catalog = json.loads(path.read_text())
    catalog["blocks"][0]["relations"] = [{"kind": "parent_table", "to_id": "table"}]
    catalog["blocks"] += [
        {"source_id": "s", "location_id": "p1", "block_id": "table", "kind": "table", "text": ""},
        {"source_id": "s", "location_id": "p2", "block_id": "note", "kind": "paragraph", "text": "Only for drained conditions.",
         "relations": [{"kind": "table_note_for", "to_id": "table"}]},
    ]
    path.write_text(json.dumps(catalog), encoding="utf-8")
    result = resolver.resolve(seed, "Exact retrieved fact.", question="Value?")
    assert {row["block_id"] for row in result} == {"header", "note"}
    assert next(row for row in result if row["block_id"] == "note")["metadata"]["page"] == 2


def test_invalid_catalog_shape_and_mismatching_source_hash_fail_closed(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    catalog = json.loads(path.read_text())
    catalog["sources"][0]["source_hash"] = "other"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор") == []
    path.write_text("[]", encoding="utf-8")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Обзор") == []


def test_required_note_precedes_sibling_rows_and_omissions_are_explicit(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    catalog = json.loads(path.read_text())
    catalog["blocks"][0]["relations"] = [{"kind": "parent_table", "to_id": "table"}]
    catalog["blocks"] += [{"source_id": "s", "location_id": "p1", "block_id": "table", "kind": "table", "text": ""}]
    catalog["blocks"] += [{"source_id": "s", "location_id": "p1", "block_id": f"row{i}", "kind": "table_row", "text": f"Value {i}",
                           "relations": [{"kind": "parent_table", "to_id": "table"}]} for i in range(9)]
    catalog["blocks"] += [{"source_id": "s", "location_id": "p2", "block_id": "note", "kind": "paragraph", "text": "Only for drained conditions.",
                           "relations": [{"kind": "table_note_for", "to_id": "table"}]}]
    path.write_text(json.dumps(catalog), encoding="utf-8")
    diagnostics = {}
    result = resolver.resolve(seed, "Exact retrieved fact.", question="Value?", max_blocks=8, diagnostics=diagnostics)
    assert "note" in [row["block_id"] for row in result]
    assert diagnostics["bounded"] and any(row["reason"] == "block_budget" for row in diagnostics["omitted"])
    resolver.resolve(seed, "Exact retrieved fact.", question="Value?", max_chars=5, diagnostics=diagnostics)
    assert any(row.get("block_id") == "note" and row["reason"] == "character_budget" for row in diagnostics["omitted"])


def test_malformed_attributes_disable_optional_catalog(tmp_path):
    resolver, seed, path, _ = fixture(tmp_path)
    catalog = json.loads(path.read_text())
    catalog["blocks"][1]["attributes"] = ["malformed"]
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.resolve(seed, "Exact retrieved fact.", question="Value?") == []
