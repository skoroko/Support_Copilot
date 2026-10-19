"""Attribute legacy CHM chunks to an exact page title, without editing evidence.

This deliberately small title registry proves only a model subject. Member path
components are locators; they establish no model applicability or version.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from bs4 import BeautifulSoup, CData, NavigableString

from app.services.chm_reader import CHMReader
from app.services.evidence_text import (
    _archive_paths,
    _read_member,
    archive_stamp,
    project_table_rows,
)

_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_SUBJECTS = {
    "hardening soil": "Hardening Soil",
    "hardening soil(small strain stiffness)": "Hardening Soil small strain",
    "hardening soil (small strain stiffness)": "Hardening Soil small strain",
}
_OTHER_MODEL = re.compile(
    r"\b(?:mohr[\s-]*coulomb|drucker[\s-]*prager|cam[\s-]*clay|"
    r"hoek[\s-]*brown|modified\s+mohr|linear\s+elastic|"
    r"soft\s+soil|jardine|"
    r"(?:material|constitutive|soil)\s+model)\b", re.IGNORECASE,
)
_FACET_HEADING = re.compile(
    r"(?:description|overview|parameters|(?:small strain stiffness|nonlinear|stiffness|strength|model) parameters|"
    r"описание|обзор|(?:основные |нелинейные )?параметры(?: модели)?)", re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _subject(title: str) -> str | None:
    return _SUBJECTS.get(_normalize(title).casefold())


@lru_cache(maxsize=32)
def _parse(raw: bytes) -> tuple[str, tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    """Cache immutable strings/spans keyed by actual member bytes, never a DOM."""
    soup = BeautifulSoup(CHMReader._decode_html(raw), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    titles = tuple(tag.get_text(" ", strip=True) for tag in soup.find_all("title"))
    offsets: dict[int, tuple[int, int]] = {}
    parts: list[str] = []
    position = 0
    for node in soup.descendants:
        if type(node) not in (NavigableString, CData):
            continue
        value = _normalize(str(node))
        if value:
            offsets[id(node)] = (position, position + len(value))
            parts.append(value)
            position += len(value) + 1
    page = " ".join(parts)
    # Keep the extractor's exact semantics as a hard guard on provenance.
    if page != CHMReader._html_to_text(raw):
        return "", titles, ()
    headings = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        spans = [offsets[id(node)] for node in tag.descendants if id(node) in offsets]
        if spans:
            headings.append((spans[0][0], spans[-1][1], tag.get_text(" ", strip=True)))
    return page, titles, tuple(headings)


def _archive(data_dir: Path, file_name: str) -> Path | None:
    root = data_dir.resolve()
    session = _archive_paths.get()
    candidates = session[1].get(file_name, []) if session and session[0] == data_dir else [
        path for path in data_dir.rglob("*")
        if path.name == file_name and path.suffix.lower() == ".chm"
    ]
    # Count candidates before resolving: duplicate aliases are ambiguous too.
    if len(candidates) != 1:
        return None
    path = candidates[0].resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def evidence_scope(
    data_dir: Path, metadata: Mapping[str, Any], chunk_id: str, chunk_text: str,
    *, projected_text: str | None = None,
) -> dict[str, Any]:
    """Return exact title provenance or an explicit, fail-closed unknown result.

    Spans use whitespace-normalized CHMReader page text and an exclusive end.
    The chunk digest always hashes the original UTF-8 chunk, not its projection.
    """
    result: dict[str, Any] = {
        "status": "unknown", "chunk_id": chunk_id,
        "chunk_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
        "file_name": metadata.get("file_name"),
        "chm_member": metadata.get("chm_member"),
        "product": metadata.get("product"),
        "document_version": None,
        "applicability": None,
    }

    def unknown(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    file_name, member, product = (metadata.get(key) for key in ("file_name", "chm_member", "product"))
    if not isinstance(file_name, str) or Path(file_name).suffix.lower() != ".chm":
        return unknown("unsupported_source")
    if not isinstance(product, str) or not product.strip() or product.strip().casefold() == "unknown":
        return unknown("missing_product")
    if not isinstance(member, str) or not member.strip():
        return unknown("missing_member")
    # CHM absolute member names are valid, but traversal and non-HTML pages are not.
    member_path = PurePosixPath(member.replace("\\", "/"))
    if ".." in member_path.parts or member_path.suffix.lower() not in {".htm", ".html", ".xhtml"}:
        return unknown("invalid_member")
    version = metadata.get("version")
    if isinstance(version, str) and version.strip() and version.strip().casefold() not in {"unknown", "none", "n/a"}:
        result["document_version"] = version
    result["section_path"] = [part for part in member_path.parts if part != "/"]
    try:
        path = _archive(data_dir, file_name)
        if path is None:
            return unknown("unavailable_or_ambiguous_archive")
        before = path.stat()
        # ctime/inode also invalidate extraction when an archive is replaced.
        raw = _read_member(str(path), archive_stamp(before), before.st_size, member)
        after = path.stat()
        if (before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_size) != (
            after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_size,
        ):
            return unknown("archive_changed")
        if not raw or len(raw) > _MAX_MEMBER_BYTES:
            return unknown("invalid_member_size")
        page, titles, headings = _parse(raw)
    except Exception:  # noqa: BLE001 - extraction failures cannot justify a scope
        return unknown("unavailable_member")
    if len(titles) != 1 or not titles[0]:
        return unknown("missing_or_ambiguous_title")
    title = titles[0]
    result["title_anchor"] = {
        "title": title, "member_sha256": hashlib.sha256(raw).hexdigest(),
    }
    subject = _subject(title)
    if subject is None:
        return unknown("unsupported_title")
    chunk = _normalize(chunk_text)
    start = page.find(chunk) if chunk else -1
    if start < 0:
        return unknown("chunk_not_found")
    if page.find(chunk, start + 1) >= 0:
        return unknown("ambiguous_chunk")
    end = start + len(chunk)
    result["normalized_span"] = {"start": start, "end": end}
    for left, _right, heading in headings:
        if left >= end:
            continue
        heading_subject = _subject(heading)
        if heading_subject and heading_subject != subject:
            return unknown("conflicting_subject_heading")
        # Recognize subject headings without interpreting parameter/body words.
        if not heading_subject and (
            re.search(r"\bhardening\s+soil\b", heading, re.IGNORECASE)
            or _OTHER_MODEL.search(heading)
        ):
            return unknown("conflicting_subject_heading")
        if not heading_subject and not _FACET_HEADING.fullmatch(_normalize(heading)):
            return unknown("unrecognized_section_heading")
    if projected_text is not None:
        current_projection = project_table_rows(chunk_text, raw)
        if not _normalize(projected_text) or _normalize(projected_text) not in _normalize(current_projection):
            return unknown("projection_changed")
        result["projection_sha256"] = hashlib.sha256(projected_text.encode("utf-8")).hexdigest()
    result.update(status="exact", subject=subject)
    return result
