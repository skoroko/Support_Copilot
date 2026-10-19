from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

scope_module = importlib.import_module("app.services.evidence_scope")

TITLE = "Hardening Soil(small strain stiffness)"
CHUNK = "Parameter Value G0 120"


def html(title=TITLE, body=None):
    body = body or "<table><tr><th>Parameter</th><th>Value</th></tr><tr><td>G0</td><td>120</td></tr></table>"
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>".encode()


@pytest.fixture
def source(tmp_path, monkeypatch):
    archive = tmp_path / "manual.chm"
    archive.write_bytes(b"archive")
    metadata = {
        "file_name": archive.name, "chm_member": "/models/hss.htm",
        "product": "GTS NX", "version": "unknown",
    }
    raw = html()
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: raw)
    return tmp_path, archive, metadata


def evaluate(source, text=CHUNK):
    directory, _, metadata = source
    return scope_module.evidence_scope(directory, metadata, "chunk-1", text)


def test_title_proves_table_subject_without_heading(source):
    metadata_before = deepcopy(source[2])
    result = evaluate(source)
    assert result["status"] == "exact"
    assert result["subject"] == "Hardening Soil small strain"
    assert result["product"] == "GTS NX"
    assert result["document_version"] is None
    assert result["applicability"] is None
    assert result["section_path"] == ["models", "hss.htm"]
    assert result["title_anchor"] == {
        "title": TITLE, "member_sha256": hashlib.sha256(html()).hexdigest(),
    }
    assert result["chunk_sha256"] == hashlib.sha256(CHUNK.encode()).hexdigest()
    page = scope_module.CHMReader._html_to_text(html())
    span = result["normalized_span"]
    assert page[span["start"]:span["end"]] == CHUNK
    assert source[2] == metadata_before


def test_body_mention_does_not_change_title_subject(source, monkeypatch):
    raw = html(body="<p>Hardening Soil is mentioned here.</p>")
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: raw)
    result = evaluate(source, "Hardening Soil is mentioned here.")
    assert result["subject"] == "Hardening Soil small strain"


@pytest.mark.parametrize("title,subject", [
    ("Hardening Soil", "Hardening Soil"),
    ("Hardening Soil (small strain stiffness)", "Hardening Soil small strain"),
])
def test_alternate_member_title_controls_scope(source, monkeypatch, title, subject):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(title))
    assert evaluate(source)["subject"] == subject


@pytest.mark.parametrize("field,value,reason", [
    ("file_name", "manual.pdf", "unsupported_source"),
    ("product", "unknown", "missing_product"),
    ("product", None, "missing_product"),
    ("chm_member", None, "missing_member"),
    ("chm_member", "../other.htm", "invalid_member"),
])
def test_unverifiable_metadata_fails_closed(source, field, value, reason):
    source[2][field] = value
    result = evaluate(source)
    assert result["status"] == "unknown"
    assert result["reason"] == reason
    assert "subject" not in result
    assert result["chunk_id"] == "chunk-1"
    assert result["chunk_sha256"]


def test_missing_chunk_fails_closed(source):
    assert evaluate(source, "missing text")["reason"] == "chunk_not_found"


def test_duplicate_chunk_fails_closed(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<p>{CHUNK}</p><p>{CHUNK}</p>"))
    assert evaluate(source)["reason"] == "ambiguous_chunk"


def test_unknown_title_cannot_inherit_body_subject(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html("Overview", f"<p>{CHUNK} Hardening Soil</p>"))
    assert evaluate(source)["reason"] == "unsupported_title"


@pytest.mark.parametrize("heading", ["Hardening Soil", "Mohr-Coulomb", "Modified Cam Clay model", "Soft Soil", "Jardine"])
def test_subject_heading_before_chunk_rejects_title_inheritance(source, monkeypatch, heading):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<h2>{heading}</h2><p>{CHUNK}</p>"))
    assert evaluate(source)["reason"] == "conflicting_subject_heading"


def test_heading_inside_selected_span_rejects_title_inheritance(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body="<p>Start</p><h3>Hardening Soil</h3><p>End</p>"))
    assert evaluate(source, "Start Hardening Soil End")["reason"] == "conflicting_subject_heading"


def test_parameter_heading_is_not_a_subject_heading(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<h2>Small strain stiffness parameters</h2><p>{CHUNK}</p>"))
    assert evaluate(source)["status"] == "exact"


def test_unrecognized_section_does_not_inherit_model_title(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<h2>Another constitutive family</h2><p>{CHUNK}</p>"))
    assert evaluate(source)["reason"] == "unrecognized_section_heading"


def test_projection_and_scope_share_snapshot_after_same_size_mtime_replacement(tmp_path, monkeypatch):
    import os

    from app.services import evidence_text

    first = html(body="<table><tr><th>Name</th><th>Value</th></tr><tr><td>G0</td><td>120</td></tr></table>")
    second = html(body="<table><tr><th>Name Value</th><th></th></tr><tr><td>G0 120</td><td></td></tr></table>")
    size = max(len(first), len(second))
    first, second = first.ljust(size, b" "), second.ljust(size, b" ")
    archive = tmp_path / "manual.chm"
    archive.write_bytes(first)
    metadata = {"file_name": archive.name, "chm_member": "/hss.htm", "product": "gts_nx"}
    chunk = scope_module.CHMReader._html_to_text(first)
    assert chunk == scope_module.CHMReader._html_to_text(second)
    reads = []

    class Archive:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, member, **kwargs):
            reads.append(member)
            return self.path.read_bytes()

    monkeypatch.setattr(evidence_text, "ChmArchive", Archive)
    evidence_text._read_member.cache_clear()
    old_projection = evidence_text.chm_context(tmp_path, archive.name, "/hss.htm", chunk)
    assert scope_module.evidence_scope(tmp_path, metadata, "id", chunk, projected_text=old_projection)["status"] == "exact"
    assert len(reads) == 1  # Scope and projection must use the same extraction cache.
    previous = archive.stat()
    replacement = tmp_path / "replacement.chm"
    replacement.write_bytes(second)
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(archive)
    new_projection = evidence_text.chm_context(tmp_path, archive.name, "/hss.htm", chunk)
    assert new_projection != old_projection and len(reads) == 2
    scope = scope_module.evidence_scope(tmp_path, metadata, "id", chunk, projected_text=new_projection)
    assert scope["status"] == "exact"
    assert scope["title_anchor"]["member_sha256"] == hashlib.sha256(second).hexdigest()
    assert scope_module.evidence_scope(tmp_path, metadata, "id", chunk, projected_text=old_projection)["reason"] == "projection_changed"


def test_different_subject_after_chunk_is_irrelevant(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<p>{CHUNK}</p><h2>Hardening Soil</h2>"))
    assert evaluate(source)["status"] == "exact"


def test_repeated_same_title_heading_is_allowed(source, monkeypatch):
    monkeypatch.setattr(scope_module, "_read_member", lambda *args: html(body=f"<h2>{TITLE}</h2><p>{CHUNK}</p>"))
    assert evaluate(source)["status"] == "exact"


def test_archive_missing_or_duplicated_fails_closed(source):
    directory, archive, _ = source
    archive.unlink()
    assert evaluate(source)["reason"] == "unavailable_or_ambiguous_archive"
    archive.write_bytes(b"archive")
    nested = directory / "nested"
    nested.mkdir()
    (nested / archive.name).write_bytes(b"duplicate")
    assert evaluate(source)["reason"] == "unavailable_or_ambiguous_archive"


def test_symlink_outside_corpus_rejected(source, tmp_path_factory):
    _, archive, _ = source
    external = tmp_path_factory.mktemp("external") / "manual.chm"
    external.write_bytes(b"archive")
    archive.unlink()
    try:
        archive.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode")
    assert evaluate(source)["reason"] == "unavailable_or_ambiguous_archive"


def test_archive_replacement_invalidates_extraction_and_parse(source, monkeypatch):
    _, archive, _ = source
    calls = []

    @lru_cache(maxsize=4)
    def cached_reader(path, stamp, size, member):
        calls.append(stamp)
        return archive.read_bytes()

    monkeypatch.setattr(scope_module, "_read_member", cached_reader)
    archive.write_bytes(html())
    assert evaluate(source)["subject"] == "Hardening Soil small strain"
    archive.write_bytes(html("Hardening Soil"))
    assert evaluate(source)["subject"] == "Hardening Soil"
    assert len(calls) == 2


def test_original_chunk_hash_uses_unnormalized_text(source):
    chunk = "Parameter\nValue  G0\t120"
    result = evaluate(source, chunk)
    assert result["status"] == "exact"
    assert result["chunk_sha256"] == hashlib.sha256(chunk.encode()).hexdigest()


def test_explicit_version_retained(source):
    source[2]["version"] = "2024"
    assert evaluate(source)["document_version"] == "2024"
