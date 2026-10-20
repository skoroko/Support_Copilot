"""Navigation hints are not evidence; reads revalidate exact PDF bindings."""
import hashlib
import json
from copy import deepcopy

import pytest

from app.services.related_evidence import RelatedEvidenceIndex


def fixture(tmp_path, *, pages=8, name="fictional-guide.pdf", language="en", sid="source"):
    source = tmp_path / name
    source.write_bytes(b"fictional immutable PDF source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    locations, blocks, seeds = [], [], []
    for page in range(1, pages + 1):
        loc = {"source_id": sid, "source_hash": digest, "location_id": f"{sid}-p{page}", "page": page,
               "product": "fictional", "version": "unknown", "language": language}
        locations.append(loc)
        text = (f"Fictional procedure {page}: literal overview." if language == "en" else
                f"Вымышленная процедура {page}: буквальный обзор.")
        for ordinal, (identity, content, kind) in enumerate([
            (f"header{page}", "Exact repeated running header", "paragraph"),
            (f"body{page}", text, "paragraph"), (f"grid{page}", "Unsafe grid text", "table_row"),
            (f"image{page}", "Untranscribed image label", "image")]):
            blocks.append({"source_id": sid, "location_id": loc["location_id"], "block_id": sid + identity,
                           "text": content, "kind": kind, "ordinal": ordinal})
        if page in [2, 4]:
            seeds.append({"metadata": {**loc, "block_id": sid + f"body{page}"}, "text": text})
    catalog = {"sources": [{"source_id": sid, "source_hash": digest, "relative_path": name, "format": "pdf"}],
               "locations": locations, "blocks": blocks}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return RelatedEvidenceIndex(tmp_path, path), seeds, catalog, path, source


def projection(result):
    return [{key: value for key, value in row.items() if key != "metadata"} for row in result["items"]]


@pytest.mark.parametrize("language", ["en", "ru"])
def test_two_distinct_verified_seed_locations_unlock_literal_non_citable_manifest(tmp_path, language):
    resolver, seeds, _, _, _ = fixture(tmp_path, language=language)
    assert resolver.navigation(seeds[:1]) == {"items": [], "truncated": False}
    assert resolver.navigation([seeds[0], seeds[0]]) == {"items": [], "truncated": False}
    nav = resolver.navigation(seeds)
    assert len(nav["items"]) == 8 and not nav["truncated"]
    assert nav == resolver.navigation(seeds)
    for item in nav["items"]:
        assert item["not_citable"] is True and len(item["hint"]) <= 160
        assert "header" not in item["hint"] and "Unsafe" not in item["hint"] and "image" not in item["hint"]
        assert item["hint"].startswith("Fictional" if language == "en" else "Вымышленная")
        group = resolver.read_navigation(item)
        assert group and group["metadata"]["page"] == item["page"]
        assert "Unsafe" not in group["text"] and "image label" not in group["text"]
        assert group["metadata"]["source_locator"].endswith(f"#page={item['page']}")
        assert group["metadata"]["block_attributes"]["extraction_only"]
        assert "relation" not in group["metadata"]


def test_character_page_and_source_caps_are_explicit_and_metadata_is_server_only(tmp_path):
    resolver, seeds, _, _, _ = fixture(tmp_path, pages=105)
    nav = resolver.navigation(seeds, max_chars=40_000)
    assert len(nav["items"]) == 100 and nav["truncated"]
    bounded = resolver.navigation(seeds, max_chars=700)
    assert bounded["items"] and bounded["truncated"]
    assert len(json.dumps(projection(bounded), ensure_ascii=False, separators=(",", ":"))) <= 700
    assert len(json.dumps(bounded, ensure_ascii=False)) > 700  # Bindings are a separate server-only budget.
    assert resolver.navigation(seeds, max_chars=2) == {"items": [], "truncated": True}
    assert resolver.navigation(seeds, max_sources=0) == {"items": [], "truncated": True}


def test_long_manual_starts_near_verified_seeds_instead_of_page_one(tmp_path):
    resolver, seeds, catalog, _, _ = fixture(tmp_path, pages=130)
    seeds = []
    for page in (117, 120):
        location = catalog['locations'][page - 1]
        block = next(row for row in catalog['blocks'] if row['block_id'] == f'sourcebody{page}')
        seeds.append({'metadata': {**location, 'block_id': block['block_id']}, 'text': block['text']})
    nav = resolver.navigation(seeds, max_chars=1200)
    pages = [row['page'] for row in nav['items']]
    assert pages[:2] == [117, 120]
    assert any(page in pages for page in (116, 118, 119, 121))
    assert nav['truncated'] and nav == resolver.navigation(seeds, max_chars=1200)


@pytest.mark.parametrize("change", ["id", "hash", "location", "product", "version", "page", "filename", "path", "not_citable"])
def test_read_rejects_tampered_binding_instead_of_following_arbitrary_paths(tmp_path, change):
    resolver, seeds, _, _, _ = fixture(tmp_path)
    item = resolver.navigation(seeds)["items"][0]
    if change in {"hash", "location", "product", "version"}:
        item["metadata"][{"hash": "source_hash", "location": "location_id"}.get(change, change)] = "altered"
    elif change == "path":
        item["metadata"]["file_path"] = "../outside.pdf"
    else:
        item[{"filename": "file_name"}.get(change, change)] = False if change == "not_citable" else "altered"
    assert resolver.read_navigation(item) is None


def test_changed_source_and_catalog_disable_stale_manifest_reads(tmp_path):
    resolver, seeds, _, path, source = fixture(tmp_path)
    item = resolver.navigation(seeds)["items"][0]
    source.write_bytes(b"changed source")
    assert resolver.read_navigation(item) is None and resolver.navigation(seeds)["items"] == []
    path.write_text("invalid", encoding="utf-8")
    assert resolver.read_navigation(item) is None


def test_ambiguous_page_is_not_listed_or_read_and_invalid_seeds_do_not_unlock(tmp_path):
    resolver, seeds, catalog, path, _ = fixture(tmp_path)
    item = next(row for row in resolver.navigation(seeds)["items"] if row['page'] == 1)
    catalog["locations"].append({**catalog["locations"][0], "location_id": "duplicate"})
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert 1 not in [row["page"] for row in resolver.navigation(seeds)["items"]]
    assert resolver.read_navigation(item) is None
    altered = deepcopy(seeds)
    altered[1]["text"] = "Invented text"
    assert resolver.navigation(altered)["items"] == []
    altered = deepcopy(seeds)
    altered[1]["metadata"]["version"] = "different"
    assert resolver.navigation(altered)["items"] == []


def test_no_chm_guesses_or_oversized_page_prefix_fallback(tmp_path):
    resolver, seeds, catalog, path, source = fixture(tmp_path)
    item = next(row for row in resolver.navigation(seeds)["items"] if row['page'] == 1)
    catalog["blocks"][1]["text"] += "x" * 6000
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.read_navigation(item) is None
    source.rename(source.with_suffix(".chm"))
    catalog["sources"][0].update(relative_path=source.with_suffix(".chm").name, format="chm")
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.navigation(seeds)["items"] == []
    assert resolver.read_navigation(item) is None


def test_exact_running_header_requires_three_distinct_pages_and_pending_text_is_not_hint(tmp_path):
    resolver, seeds, catalog, path, _ = fixture(tmp_path, pages=4)
    for block in catalog["blocks"]:
        if block["block_id"] in {"sourcebody1", "sourcebody3"}:
            block["attributes"] = {"review_status": "pending_review"}
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert [row["page"] for row in resolver.navigation(seeds)["items"]] == [2, 4]
    # Repeat a paragraph three times on one page; that does not establish a running header.
    for block in catalog["blocks"]:
        if block["block_id"].startswith("sourceheader"):
            block["text"] = "Unique heading " + str(block["block_id"])
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.navigation(seeds)["items"][0]["hint"].startswith("Unique heading")


def test_source_order_and_two_source_cap_do_not_merge_same_hash_documents(tmp_path):
    resolver, seeds, catalog, path, source = fixture(tmp_path)
    source_seeds = [seeds]
    template = deepcopy(catalog)
    for number in (2, 3):
        sid = f"other{number}"
        (tmp_path / f"other{number}.pdf").write_bytes(source.read_bytes())
        catalog["sources"].append({**template["sources"][0], "source_id": sid,
                                   "relative_path": f"other{number}.pdf"})
        catalog["locations"] += [{**loc, "source_id": sid, "location_id": loc["location_id"].replace("source", sid)}
                                 for loc in template["locations"]]
        catalog["blocks"] += [{**block, "source_id": sid, "location_id": block["location_id"].replace("source", sid),
                               "block_id": block["block_id"].replace("source", sid)} for block in template["blocks"]]
        source_seeds.append([{"metadata": {**seed["metadata"], "source_id": sid,
                            "location_id": seed["metadata"]["location_id"].replace("source", sid),
                            "block_id": seed["metadata"]["block_id"].replace("source", sid)}, "text": seed["text"]}
                             for seed in seeds])
    path.write_text(json.dumps(catalog), encoding="utf-8")
    ordered = source_seeds[1] + source_seeds[0] + source_seeds[2]
    nav = resolver.navigation(ordered, max_sources=1)
    assert {item["file_name"] for item in nav["items"]} == {"other2.pdf"} and nav["truncated"]
    nav = resolver.navigation(ordered, max_sources=99)
    assert list(dict.fromkeys(item["file_name"] for item in nav["items"])) == ["other2.pdf", "fictional-guide.pdf"]
    assert nav["truncated"] and len({item["id"] for item in nav["items"]}) == len(nav["items"])


def test_conflicting_verified_scopes_within_one_pdf_do_not_unlock_navigation(tmp_path):
    resolver, seeds, catalog, path, _ = fixture(tmp_path)
    for page in (6, 8):
        loc = catalog["locations"][page-1]
        loc["version"] = "other-version"
        block = next(row for row in catalog["blocks"] if row["block_id"] == f"sourcebody{page}")
        seeds.append({"metadata": {**loc, "block_id": block["block_id"]}, "text": block["text"]})
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert resolver.navigation(seeds)["items"] == []
