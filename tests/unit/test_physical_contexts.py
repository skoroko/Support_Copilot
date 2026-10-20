"""Batch physical retrieval amortizes checks, never stale evidence bindings."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest

from app.services import related_evidence
from app.services.related_evidence import RelatedEvidenceIndex


def fixture(root):
    root.mkdir(exist_ok=True)
    catalog = {"sources": [], "locations": [], "blocks": []}
    seeds = []
    for sid in ("a", "b"):
        source = root / f"{sid}.pdf"
        source.write_bytes(f"source {sid}".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        catalog["sources"].append({"source_id": sid, "source_hash": digest,
                                   "relative_path": source.name, "format": "pdf"})
        for page in (1, 2, 3):
            loc = {"source_id": sid, "source_hash": digest, "location_id": f"{sid}{page}",
                   "page": page, "product": "civil", "version": "unknown",
                   "figure_regions": [{"bbox": [0, 0, 1, 1]}]}
            block = {"source_id": sid, "location_id": loc["location_id"],
                     "block_id": f"block-{sid}{page}", "ordinal": 1, "kind": "paragraph",
                     "text": f"Literal {sid} page {page} θ", "relations": []}
            catalog["locations"].append(loc)
            catalog["blocks"].append(block)
            if page == 2:
                seeds.append({"metadata": {**loc, "block_id": block["block_id"]}, "text": block["text"]})
    path = root / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return RelatedEvidenceIndex(root, path), seeds, path, catalog


def serial(index, seeds, max_chars=6000):
    result = []
    for seed in seeds:
        local_report, neighbor_report = {}, {}
        group = index.location_context(**seed, max_chars=max_chars, diagnostics=local_report)
        groups = [group] if group else []
        groups.extend(index.nearby_contexts(**seed, max_chars=max_chars, diagnostics=neighbor_report))
        result.append({"groups": groups, "diagnostics": [local_report, neighbor_report]})
    return result


@pytest.mark.parametrize("max_chars", [1, 6000])
def test_serial_parity_order_scope_and_detached_groups(tmp_path, max_chars):
    index, seeds, _, _ = fixture(tmp_path)
    seeds += [deepcopy(seeds[0]), {"metadata": {**seeds[0]["metadata"], "product": "other"},
                                 "text": seeds[0]["text"]}]
    expected = serial(index, seeds, max_chars)
    actual = index.physical_contexts(seeds, max_chars=max_chars)
    assert actual == expected
    assert len(actual) == len(seeds)
    if max_chars == 6000:
        assert [g["metadata"]["page"] for g in actual[0]["groups"]] == [2, 1, 3]
        actual[0]["groups"][0]["metadata"]["figure_regions"].clear()
        assert actual[2]["groups"][0]["metadata"]["figure_regions"]
        assert index.physical_contexts(seeds)[0]["groups"][0]["metadata"]["figure_regions"]


def test_file_checks_scale_with_distinct_sources_and_standalone_remains_fresh(tmp_path, monkeypatch):
    index, seeds, _, _ = fixture(tmp_path)
    assert index.warmup()["valid_sources"] == 2
    original = index._source_binding
    checks = []
    stamps = []
    original_stamp = related_evidence._stamp

    def binding(sid):
        checks.append(sid)
        return original(sid)

    def stamp(path):
        stamps.append(path.name)
        return original_stamp(path)

    monkeypatch.setattr(index, "_source_binding", binding)
    monkeypatch.setattr(related_evidence, "_stamp", stamp)
    batch = index.physical_contexts(seeds * 15)
    assert all(len(row["groups"]) == 3 for row in batch)
    assert sorted(checks) == ["a", "a", "b", "b"]
    assert sorted(stamps) == ["a.pdf", "a.pdf", "b.pdf", "b.pdf", "catalog.json", "catalog.json"]
    checks.clear()
    for _ in range(2):
        assert index.location_context(**seeds[0])
    assert checks == ["a", "a"]


@pytest.mark.parametrize("change", ["source", "catalog", "catalog_path", "source_path"])
def test_mid_batch_changes_discard_every_group_and_clear_diagnostics(tmp_path, monkeypatch, change):
    index, seeds, path, catalog = fixture(tmp_path)
    original = index.location_context
    called = False
    original_binding = index._source_binding
    moved = False

    def binding(sid):
        value = original_binding(sid)
        if moved and sid == "a" and value:
            return tmp_path / "other.pdf", value[1]
        return value

    monkeypatch.setattr(index, "_source_binding", binding)

    def location(*args, **kwargs):
        nonlocal called, moved
        result = original(*args, **kwargs)
        if not called:
            called = True
            if change == "source":
                (tmp_path / "a.pdf").write_bytes(b"changed source bytes")
            elif change == "catalog":
                catalog["blocks"][0]["text"] = "changed extraction"
                path.write_text(json.dumps(catalog), encoding="utf-8")
            elif change == "catalog_path":
                replacement = tmp_path / "replacement.json"
                replacement.write_bytes(path.read_bytes())
                index.catalog_path = replacement
            else:
                moved = True
        return result

    monkeypatch.setattr(index, "location_context", location)
    result = index.physical_contexts(seeds)
    assert len(result) == 2
    assert all(not row["groups"] for row in result)
    for row in result:
        for report in row["diagnostics"]:
            assert report["reason"] == "batch_changed" and report["invalidated"]
            assert not report["bounded"] and not report["omitted"]
            assert not report.get("selected_blocks", 0) and not report.get("selected_locations", 0)
    assert related_evidence._PHYSICAL_BATCH.get() is None


def test_stale_or_missing_catalog_and_source_never_gain_groups(tmp_path):
    index, seeds, path, _ = fixture(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"changed source")
    result = index.physical_contexts(seeds)
    assert not result[0]["groups"] and len(result[1]["groups"]) == 3
    path.write_text("invalid catalog", encoding="utf-8")
    result = index.physical_contexts(seeds)
    assert all(not row["groups"] for row in result)
    assert all(row["diagnostics"][0]["reason"] == "unverified_catalog" for row in result)


def test_hash_validity_cache_includes_resolved_path_even_with_same_stamp(tmp_path, monkeypatch):
    index, seeds, _, _ = fixture(tmp_path)
    assert index.location_context(**seeds[0])
    original_binding = index._source_binding
    replacement = tmp_path / "replacement.pdf"
    replacement.write_bytes(b"unaccepted bytes")

    def binding(sid):
        original = original_binding(sid)
        return (replacement, original[1]) if sid == "a" else original

    monkeypatch.setattr(index, "_source_binding", binding)
    assert index.location_context(**seeds[0]) is None
    assert not index.physical_contexts(seeds)[0]["groups"]


def test_exception_cleans_memo_and_does_not_mask_next_source_change(tmp_path, monkeypatch):
    index, seeds, _, _ = fixture(tmp_path)
    original = index.nearby_contexts

    def fail(*_args, **_kwargs):
        assert related_evidence._PHYSICAL_BATCH.get()["owner"] is index
        raise RuntimeError("test failure")

    monkeypatch.setattr(index, "nearby_contexts", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        index.physical_contexts(seeds)
    assert related_evidence._PHYSICAL_BATCH.get() is None
    monkeypatch.setattr(index, "nearby_contexts", original)
    (tmp_path / "a.pdf").write_bytes(b"changed after exception")
    assert index.location_context(**seeds[0]) is None


def test_thread_and_instance_isolation(tmp_path, monkeypatch):
    first, seeds1, _, _ = fixture(tmp_path / "first")
    second, seeds2, _, _ = fixture(tmp_path / "second")
    barrier = Barrier(2)

    def patch(index):
        original = index.location_context
        first_call = True

        def location(*args, **kwargs):
            nonlocal first_call
            assert related_evidence._PHYSICAL_BATCH.get()["owner"] is index
            if first_call:
                first_call = False
                barrier.wait(timeout=5)
            return original(*args, **kwargs)
        monkeypatch.setattr(index, "location_context", location)

    patch(first)
    patch(second)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(first.physical_contexts, seeds1)
        future2 = executor.submit(second.physical_contexts, seeds2)
        assert all(row["groups"] for row in future1.result(timeout=10))
        assert all(row["groups"] for row in future2.result(timeout=10))
    assert related_evidence._PHYSICAL_BATCH.get() is None


def test_nested_same_instance_batch_rejected_without_leaking_state(tmp_path, monkeypatch):
    index, seeds, _, _ = fixture(tmp_path)
    monkeypatch.setattr(index, "nearby_contexts", lambda *_args, **_kwargs: index.physical_contexts(seeds))
    with pytest.raises(RuntimeError, match="Nested physical evidence batch"):
        index.physical_contexts(seeds)
    assert related_evidence._PHYSICAL_BATCH.get() is None
