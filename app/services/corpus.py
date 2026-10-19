"""Offline, format-neutral evidence catalog. No model calls or invented relations.

Catalog files contain source text and belong in private local ``var/`` storage.
Only :meth:`CorpusCatalog.summary` is suitable for public execution reports.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EXTRACTION_VERSION = "corpus-v1"
SCHEMA_VERSION = 1


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def source_identity(path: Path, data_root: Path) -> tuple[str, str]:
    try:
        relative = path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        # An explicitly uploaded external file must not collide with a corpus file.
        relative = "external:" + path.resolve().as_posix()
    return stable_id("source", relative), relative


@dataclass
class EvidenceBlock:
    block_id: str
    kind: str
    text: str
    ordinal: int
    section_path: list[str] = field(default_factory=list)
    anchor: str | None = None
    bbox: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    relations: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def block(location_id: str, ordinal: int, kind: str, text: str, **kwargs: Any) -> EvidenceBlock:
    return EvidenceBlock(stable_id(location_id, ordinal, kind), kind, text, ordinal, **kwargs)


@dataclass
class CorpusCatalog:
    """A complete source/location ledger, including empty and unreadable units."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    extraction_version: str = EXTRACTION_VERSION
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_documents(
        cls, paths: Iterable[Path], documents: Iterable[Any], *, data_root: Path,
        errors: dict[str, str] | None = None,
    ) -> CorpusCatalog:
        catalog = cls()
        errors = errors or {}
        for path in sorted(paths, key=lambda p: p.as_posix().casefold()):
            sid, relative = source_identity(path, data_root)
            catalog.sources.append({
                "source_id": sid, "relative_path": relative, "source_hash": file_hash(path),
                "format": path.suffix.lower().lstrip("."), "bytes": path.stat().st_size,
                "status": "unreadable" if str(path) in errors else "extracted",
                "error_type": errors.get(str(path)), "extraction_version": EXTRACTION_VERSION,
            })
        for document in documents:
            m = document.metadata
            location_id = m.get("location_id")
            if not location_id:
                continue
            catalog.locations.append({
                key: m.get(key) for key in (
                    "location_id", "source_id", "source_hash", "page", "total_pages", "chm_member",
                    "extraction_status", "extraction_error", "language", "version", "product",
                    "section_path", "title", "page_bbox", "figure_regions", "warnings", "chm_read_method",
                ) if key in m
            })
            for item in m.get("evidence_blocks", []):
                catalog.blocks.append({"location_id": location_id, "source_id": m["source_id"], **item})
            for relation in m.get("relations", []):
                catalog.relations.append({"source_id": m["source_id"], **relation})
        catalog._resolve_relations()
        for source in catalog.sources:
            statuses = Counter(loc["extraction_status"] for loc in catalog.locations
                               if loc["source_id"] == source["source_id"])
            source["location_statuses"] = dict(sorted(statuses.items()))
            if source["status"] != "unreadable" and statuses["unreadable"]:
                source["status"] = "partial"
        return catalog

    @classmethod
    def build(cls, data_root: Path, *, observations: list[dict[str, Any]] | None = None,
              transcriptions: list[dict[str, Any]] | None = None) -> CorpusCatalog:
        # Lazy import keeps record types usable by the readers without a cycle.
        from app.services.ingestion import _load_one_file, supported_files

        paths = supported_files(data_root)
        documents: list[Any] = []
        errors: dict[str, str] = {}
        for path in paths:
            try:
                documents.extend(_load_one_file(path, data_root=data_root))
            except Exception as exc:  # noqa: BLE001 - malformed sources remain in the coverage ledger
                errors[str(path)] = type(exc).__name__
        apply_reviewed_transcriptions(documents, transcriptions or [], data_root=data_root)
        catalog = cls.from_documents(paths, documents, data_root=data_root, errors=errors)
        catalog.apply_observations(observations or [])
        return catalog

    def apply_observations(self, observations: list[dict[str, Any]]) -> None:
        """Attach explicit hash-bound visual review; image presence alone proves nothing."""
        for observation in observations:
            if observation.get("status") not in {"blank_verified", "reviewed"}:
                raise ValueError("Unsupported visual observation status")
            matches = [loc for loc in self.locations
                       if loc.get("source_hash") == observation.get("source_hash")
                       and loc.get("page") == observation.get("page")]
            if len(matches) != 1 or not observation.get("reviewer") or not observation.get("reason"):
                raise ValueError("Visual observations require one exact hash/page match, reviewer and reason")
            matches[0]["visual_review"] = dict(observation)

    def _resolve_relations(self) -> None:
        """Resolve literal local links and generated HTML wrappers, without semantic guesses."""
        import posixpath
        from urllib.parse import unquote, urlsplit

        sources = {s["source_id"]: s for s in self.sources}
        self._link_explicit_pdf_sections(sources)
        self._link_continued_tables()
        by_path = {s["relative_path"]: s["source_id"] for s in self.sources}
        members = {(loc["source_id"], loc.get("chm_member")): loc["location_id"]
                   for loc in self.locations if loc.get("chm_member")}
        html_locations = {loc["source_id"]: loc["location_id"] for loc in self.locations if not loc.get("page")
                          and not loc.get("chm_member")}
        anchors = {(b["location_id"], b.get("anchor")): b["block_id"] for b in self.blocks if b.get("anchor")}
        location_by_id = {loc["location_id"]: loc for loc in self.locations}
        for evidence in self.blocks:
            attrs = evidence.get("attributes", {})
            href = attrs.get("href")
            if not href:
                continue
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue  # retain external URL as source data, never fetch it
            loc = location_by_id[evidence["location_id"]]
            target_location = None
            if loc.get("chm_member"):
                name = posixpath.normpath(posixpath.join(posixpath.dirname(loc["chm_member"]), unquote(parsed.path)))
                if not parsed.path:
                    name = loc["chm_member"]
                target_location = members.get((loc["source_id"], name))
            else:
                relative = sources[loc["source_id"]]["relative_path"]
                target_path = posixpath.normpath(posixpath.join(posixpath.dirname(relative), unquote(parsed.path)))
                if not parsed.path:
                    target_path = relative
                target_location = html_locations.get(by_path.get(target_path))
            if target_location:
                target = anchors.get((target_location, unquote(parsed.fragment))) if parsed.fragment else target_location
                if target:
                    self.relations.append({"kind": "explicit_link", "from_id": evidence["block_id"],
                                           "to_id": target, "source_id": loc["source_id"]})
        # Existing course HTML wrappers explicitly name the source CHM topic in title/h1.
        for loc in self.locations:
            source = sources[loc["source_id"]]
            if source["format"] not in {"html", "htm"}:
                continue
            title = str(loc.get("title") or "")
            if not title.startswith("/") or not title.lower().endswith((".htm", ".html", ".xhtml")):
                continue
            matches = [lid for (_, member), lid in members.items() if member == title]
            if len(matches) == 1:
                self.relations.append({"kind": "derived_from", "from_id": loc["location_id"], "to_id": matches[0],
                                       "source_id": loc["source_id"], "basis": "exact_wrapper_title_member"})
        unique = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in self.relations}
        self.relations = [unique[key] for key in sorted(unique)]

    def _link_explicit_pdf_sections(self, sources: dict[str, dict[str, Any]]) -> None:
        """An actual PDF outline, never physical adjacency, confirms page membership."""
        existing = {b["block_id"] for b in self.blocks}
        blocks_by_location: dict[str, list[dict[str, Any]]] = {}
        for item in self.blocks:
            blocks_by_location.setdefault(item["location_id"], []).append(item)
        for sid, source in sources.items():
            if source["format"] != "pdf":
                continue
            locations = sorted((loc for loc in self.locations if loc["source_id"] == sid), key=lambda loc: loc["page"])
            current_path: list[str] = []
            heading_id = None
            headings: dict[tuple[str, ...], str] = {}
            for loc in locations:
                path = loc.get("section_path") or []
                if not path:
                    current_path = []
                    heading_id = None
                    continue
                if path != current_path:
                    heading_id = stable_id("pdf-outline", sid, loc["page"], *path)
                    heading = {"location_id": loc["location_id"], "source_id": sid, "block_id": heading_id,
                               "kind": "outline_heading", "text": path[-1], "ordinal": -1,
                               "section_path": path, "anchor": None, "bbox": None,
                               "attributes": {"method": "pdf_explicit_outline", "destination_page": loc["page"]},
                               "relations": []}
                    parent = headings.get(tuple(path[:-1]))
                    if parent:
                        heading["relations"].append({"kind": "parent_section", "to_id": parent})
                        self.relations.append({"source_id": sid, "kind": "parent_section", "from_id": heading_id, "to_id": parent})
                    if heading_id not in existing:
                        self.blocks.append(heading)
                        existing.add(heading_id)
                    headings[tuple(path)] = heading_id
                    current_path = path
                for item in blocks_by_location.get(loc["location_id"], []):
                    if item["kind"] == "outline_heading":
                        continue
                    relation = {"kind": "parent_section", "to_id": heading_id}
                    if relation not in item.setdefault("relations", []):
                        item["relations"].append(relation)
                    self.relations.append({"source_id": sid, "from_id": item["block_id"], **relation})

    def _link_continued_tables(self) -> None:
        """Link only numbered, explicitly continued tables with the same source/width."""
        import re
        by_id = {b["block_id"]: b for b in self.blocks}
        locs = {loc["location_id"]: loc for loc in self.locations}
        captions: dict[str, dict[str, Any]] = {}
        for relation in self.relations:
            if relation["kind"] == "table_caption_for" and relation["from_id"] in by_id:
                caption = by_id[relation["from_id"]]
                match = re.match(r"(?i)^(?:table|таблица)\s+(\d+(?:\.\d+)*)", caption["text"])
                if match:
                    captions[relation["to_id"]] = {"number": match.group(1), "block": caption,
                                                  "continued": bool(re.search(r"(?i)continu(?:ed|ation)|продолж", caption["text"]))}
        rows: dict[str, list[dict[str, Any]]] = {}
        for item in self.blocks:
            if item["kind"] in {"table_row", "table_header"}:
                rows.setdefault(item.get("attributes", {}).get("table_id", ""), []).append(item)
        for table_id, caption in captions.items():
            if not caption["continued"] or table_id not in by_id:
                continue
            table = by_id[table_id]
            loc = locs[table["location_id"]]
            if not loc.get("page"):
                continue
            width = {len(r.get("attributes", {}).get("cells", [])) for r in rows.get(table_id, [])}
            candidates = []
            for previous_id, previous in captions.items():
                previous_table = by_id.get(previous_id)
                if previous_id == table_id or not previous_table or previous["number"] != caption["number"]:
                    continue
                previous_loc = locs[previous_table["location_id"]]
                previous_width = {len(r.get("attributes", {}).get("cells", [])) for r in rows.get(previous_id, [])}
                if (previous_loc["source_id"] == loc["source_id"] and previous_loc.get("page") == loc["page"] - 1
                        and len(width) == 1 and width == previous_width and next(iter(width), 0) >= 2):
                    candidates.append(previous_id)
            if len(candidates) == 1:
                relation = {"kind": "continued_table", "to_id": candidates[0]}
                if relation not in table.setdefault("relations", []):
                    table["relations"].append(relation)
                self.relations.append({"source_id": loc["source_id"], "from_id": table_id,
                                       "basis": "same_number_explicit_continuation_same_width_adjacent_page", **relation})

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "extraction_version": self.extraction_version,
            "sources": len(self.sources), "formats": dict(sorted(Counter(s["format"] for s in self.sources).items())),
            "locations": len(self.locations),
            "location_statuses": dict(sorted(Counter(loc["extraction_status"] for loc in self.locations).items())),
            "failed_sources": sum(s["status"] == "unreadable" for s in self.sources),
            "partial_sources": sum(s["status"] == "partial" for s in self.sources),
            "blocks": len(self.blocks), "block_types": dict(sorted(Counter(b["kind"] for b in self.blocks).items())),
            "relations": dict(sorted(Counter(r["kind"] for r in self.relations).items())),
            "visual_reviews": sum("visual_review" in loc for loc in self.locations),
            "untranscribed_image_regions": sum(b["kind"] == "visual_unextracted"
                                               and not b.get("attributes", {}).get("approved_transcription_id")
                                               for b in self.blocks),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> CorpusCatalog:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def valid_reviewed_relation(relation: dict[str, Any], transcript: dict[str, Any],
                            metadata: dict[str, Any], blocks: dict[str, dict[str, Any]]) -> bool:
    """Mechanically validate a separately reviewed, same-page definition link.

    Exact source quotes establish provenance, not the engineering truth of an
    association. A reviewer must explicitly approve the source relationship;
    literal image reading never supplies that approval implicitly.
    """
    attrs = relation.get("attributes", {})
    target = blocks.get(relation["to_id"]) if isinstance(relation.get("to_id"), str) else None
    proof = attrs.get("proof", {}) if isinstance(attrs, dict) else {}
    keys = ("source_id", "source_hash", "location_id", "page", "product", "version")
    return bool(
        relation.get("kind") == "defines" and isinstance(attrs, dict)
        and attrs.get("method") == "reviewed_source_association"
        and attrs.get("approval_status") == "approved"
        and attrs.get("review_scope") == "explicit_source_definition_link"
        and attrs.get("semantic_verified") is False
        and type(metadata.get("page")) is int and metadata["page"] >= 1
        and all(attrs.get(k) == metadata.get(k) and metadata.get(k) is not None for k in keys)
        and all(isinstance(attrs.get(k), str) and attrs[k].strip() for k in ("review_id", "reviewer", "reviewed_at"))
        and transcript.get("kind") in {"formula", "figure_transcription", "definition", "table_row"}
        and transcript.get("attributes", {}).get("method") == "reviewed_visual_transcription"
        and attrs.get("review_id") != transcript.get("attributes", {}).get("review_id")
        and target and target.get("block_id") != transcript.get("block_id")
        and target.get("location_id", metadata.get("location_id")) == metadata.get("location_id")
        and target.get("source_id", metadata.get("source_id")) == metadata.get("source_id")
        and target.get("kind") in {"paragraph", "definition", "formula_candidate", "table_row"}
        and target.get("attributes", {}).get("method") != "reviewed_visual_transcription"
        and isinstance(proof, dict) and set(proof) == {"basis", "formula_quote", "target_quote"}
        and proof.get("basis") in {"explicit_definition", "explicit_reference"}
        and all(isinstance(proof.get(k), str) and proof[k].strip() for k in ("formula_quote", "target_quote"))
        and proof["formula_quote"] in transcript.get("text", "")
        and proof["target_quote"] in target.get("text", "")
    )


def apply_reviewed_relations(document: Any, transcript: dict[str, Any], record: dict[str, Any]) -> bool:
    """Append separately approved definition edges; identical repeats are no-ops.

    Targets are native blocks in this exact physical page Document. Cross-page
    or revised/conflicting associations require a separate reviewed migration.
    All approved records are validated before any metadata is changed.
    """
    rows = record.get("reviewed_relations", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("reviewed_relations must be a list of separately reviewed associations")
    metadata = document.metadata
    blocks = {b["block_id"]: b for b in metadata.get("evidence_blocks", [])}
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("approval_status") != "approved":
            continue
        relation = {"kind": row.get("kind"), "to_id": row.get("to_id"),
                    "attributes": {k: row.get(k) for k in
                                   ("source_id", "source_hash", "location_id", "page", "product", "version",
                                    "approval_status", "review_scope", "review_id", "reviewer", "reviewed_at", "proof")}}
        relation["attributes"].update(method="reviewed_source_association", semantic_verified=False)
        if not valid_reviewed_relation(relation, transcript, metadata, blocks):
            raise ValueError("Reviewed association requires separate approval/proof and an exact same-page native target")
        key = (relation["kind"], relation["to_id"])
        old = [r for r in transcript.get("relations", []) if (r.get("kind"), r.get("to_id")) == key]
        if any(r != relation for r in old) or key in pending and pending[key] != relation:
            raise ValueError("Conflicting reviewed association; explicit revision review required")
        if not old:
            pending[key] = relation
    for relation in pending.values():
        transcript.setdefault("relations", []).append(deepcopy(relation))
        metadata.setdefault("relations", []).append({"from_id": transcript["block_id"], **deepcopy(relation)})
    return bool(pending)


def apply_reviewed_transcriptions(documents: Iterable[Any], records: list[dict[str, Any]], *,
                                 data_root: Path | None = None) -> None:
    """Consume only explicitly approved exact-source visual transcripts.

    A record requires source_hash, physical page, bbox, image_sha256, product,
    version, text, kind, approval_status, reviewer, reviewed_at and review_id.
    Pending OCR output is source material for review, never searchable evidence.
    """
    docs = list(documents)
    for record in records:
        if record.get("approval_status") != "approved":
            continue
        if "visual_locator" in record:
            from app.services.reviewed_visual_regions import (
                apply_reviewed_visual_region,
            )

            if data_root is None:
                raise ValueError("Original permitted data root required for general visual review")
            matches = [document for document in docs if all(document.metadata.get(key) == record.get(key)
                       for key in ("source_id", "source_hash", "location_id", "product", "version"))]
            if len(matches) != 1:
                raise ValueError("Approved visual region requires one exact original source location")
            apply_reviewed_visual_region(matches[0], record, data_root)
            continue
        required = {"source_hash", "page", "bbox", "image_sha256", "product", "version",
                    "text", "kind", "reviewer", "reviewed_at", "review_id"}
        if not required.issubset(record) or not all(record[k] for k in required - {"page", "bbox"}):
            raise ValueError("Approved visual transcription is missing required source/review fields")
        if record["kind"] not in {"formula", "table_row", "figure_transcription", "definition"}:
            raise ValueError("Unsupported reviewed visual evidence kind")
        matches = []
        for document in docs:
            m = document.metadata
            if (m.get("source_hash") == record["source_hash"] and m.get("page") == record["page"]
                    and m.get("product") == record["product"] and m.get("version") == record["version"]):
                for image in m.get("evidence_blocks", []):
                    if (image["kind"] == "visual_unextracted" and image.get("bbox") == record["bbox"]
                            and image.get("attributes", {}).get("image_sha256") == record["image_sha256"]):
                        matches.append((document, image))
        if len(matches) != 1:
            raise ValueError("Approved visual transcription does not match one exact source/page/image region")
        document, image = matches[0]
        m = document.metadata
        transcript = block(m["location_id"], len(m["evidence_blocks"]), record["kind"], record["text"],
                           section_path=image.get("section_path", []), bbox=record["bbox"],
                           attributes={"method": "reviewed_visual_transcription", "verified": True,
                                       "verification_scope": "literal_image_transcription", "semantic_verified": False,
                                       "review_id": record["review_id"], "reviewer": record["reviewer"],
                                       "reviewed_at": record["reviewed_at"], "image_sha256": record["image_sha256"]},
                           relations=[{"kind": "transcription_of", "to_id": image["block_id"]}]).as_dict()
        apply_reviewed_relations(document, transcript, record)
        image["attributes"]["approved_transcription_id"] = transcript["block_id"]
        m["evidence_blocks"].append(transcript)
        m.setdefault("relations", []).append({"kind": "transcription_of", "from_id": transcript["block_id"],
                                              "to_id": image["block_id"]})
        document.set_content(document.text + "\n\n" + record["text"])
        m["extraction_status"] = "text"


def audit_catalog(catalog: CorpusCatalog, *, data_root: Path | None = None, docstore_path: Path | None = None,
                  qdrant_snapshot: Path | None = None) -> dict[str, Any]:
    """Compare saved state without connecting to Qdrant, printing text, or applying repairs."""
    report: dict[str, Any] = {"catalog": catalog.summary(), "mode": "read_only", "docstore": None, "qdrant": None}
    if data_root:
        from app.services.ingestion import supported_files
        current = {source_identity(path, data_root)[0]: file_hash(path) for path in supported_files(data_root)}
        extracted = {source["source_id"]: source["source_hash"] for source in catalog.sources}
        report["sources"] = {"current": len(current), "extracted": len(extracted),
                             "new_sources": len(current.keys() - extracted.keys()),
                             "deleted_sources": len(extracted.keys() - current.keys()),
                             "changed_sources": sum(current[sid] != extracted[sid] for sid in current.keys() & extracted.keys())}
    allowed_hashes = {s["source_id"]: s["source_hash"] for s in catalog.sources}
    source_by_id = {s["source_id"]: s for s in catalog.sources}
    location_by_id = {loc["location_id"]: loc for loc in catalog.locations}
    expected_locations = {loc["location_id"] for loc in catalog.locations if loc["extraction_status"] == "text"}

    def identity(metadata: dict[str, Any], ref_id: str = "") -> tuple[str | None, str | None]:
        sid = metadata.get("source_id")
        if sid not in source_by_id:
            path = str(metadata.get("file_path") or metadata.get("file_name") or "").replace("\\", "/")
            matches = [s["source_id"] for s in catalog.sources
                       if path == s["relative_path"] or path.endswith("/" + s["relative_path"])]
            if not matches:
                name = str(metadata.get("file_name") or "")
                matches = [s["source_id"] for s in catalog.sources if Path(s["relative_path"]).name == name]
            sid = matches[0] if len(matches) == 1 else None
        if sid is None:
            return None, None
        if metadata.get("location_id"):
            return sid, metadata["location_id"]
        source_format = source_by_id[sid]["format"]
        if source_format == "pdf":
            page = metadata.get("page") or metadata.get("page_label") or metadata.get("page_number")
            if page is None:
                import re
                match = re.search(r"_part_(\d+)$", ref_id)
                page = int(match.group(1)) + 1 if match else None
            try:
                return sid, stable_id("location", sid, int(page)) if page is not None else None
            except (ValueError, TypeError):
                return sid, None
        if source_format == "chm":
            member = metadata.get("chm_member")
            return sid, stable_id("location", sid, member) if member else None
        if source_format in {"html", "htm"}:
            return sid, stable_id("location", sid, "html")
        return sid, None
    stored_ids: set[str] = set()
    stored_locations: set[str] = set()
    if docstore_path:
        raw = json.loads(docstore_path.read_text(encoding="utf-8"))
        items = []
        for stored_id, value in raw.get("docstore/data", {}).items():
            node = value.get("__data__", value)
            node = json.loads(node) if isinstance(node, str) else node
            stored_ids.add(stored_id)
            items.append(node)
        metadata = [node.get("metadata", {}) for node in items]
        identities = [identity(m, str(node.get("id_", ""))) for m, node in zip(metadata, items, strict=True)]
        stored_locations = {loc for _, loc in identities if loc}
        legacy = sum(not m.get("extraction_version") for m in metadata)
        stale = sum(bool(m.get("source_id")) and allowed_hashes.get(m["source_id"]) != m.get("source_hash")
                    for m in metadata)
        report["docstore"] = {"documents": len(items), "legacy_documents": legacy,
                              "missing_provenance_ids": sum(not m.get("location_id") for m in metadata),
                              "source_hash_unverified_documents": sum(not m.get("source_hash") for m in metadata),
                              "product_mismatch_documents": sum(loc in location_by_id and m.get("product") != location_by_id[loc].get("product")
                                                                for (_, loc), m in zip(identities, metadata, strict=True)),
                              "stale_or_deleted_source_documents": stale,
                              "missing_text_locations": len(expected_locations - stored_locations),
                              "unexpected_locations": len(stored_locations - expected_locations),
                              "products": dict(sorted(Counter(str(m.get("product", "unknown")) for m in metadata).items())),
                              "reindex_required": bool(legacy or stale or expected_locations - stored_locations
                                                       or stored_locations - expected_locations)}
    if qdrant_snapshot:
        raw = json.loads(qdrant_snapshot.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            points = raw
            complete = False  # a bare list cannot certify a complete Qdrant scroll
        else:
            points = raw.get("points", raw.get("result", {}).get("points", []))
            complete = raw.get("complete") is True
        point_doc_ids = set()
        point_locations = set()
        missing_provenance = 0
        product_mismatches = 0
        formats: Counter[str] = Counter()
        products: Counter[str] = Counter()
        for point in points:
            payload = point.get("payload") or {}
            ref = payload.get("doc_id") or payload.get("ref_doc_id") or payload.get("document_id")
            if ref:
                point_doc_ids.add(str(ref))
            sid, loc = identity(payload, str(ref or ""))
            if loc:
                point_locations.add(loc)
            if sid:
                formats[source_by_id[sid]["format"]] += 1
            products[str(payload.get("product") or "unknown")] += 1
            missing_provenance += not bool(payload.get("location_id"))
            product_mismatches += loc in location_by_id and payload.get("product") != location_by_id[loc].get("product")
        report["qdrant"] = {"points": len(points), "snapshot_complete": complete,
                            "ref_documents": len(point_doc_ids), "text_locations": len(point_locations),
                            "missing_provenance_ids": missing_provenance,
                            "product_mismatch_points": product_mismatches,
                            "formats": dict(sorted(formats.items())), "products": dict(sorted(products.items())),
                            "docstore_ids_missing_in_snapshot": len(stored_ids - point_doc_ids) if docstore_path else None,
                            "snapshot_ids_missing_in_docstore": len(point_doc_ids - stored_ids) if docstore_path else None,
                            "missing_text_locations": len(expected_locations - point_locations),
                            "conclusive": complete,
                            "repair_applied": False}
    return report
