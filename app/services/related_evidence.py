"""Resolve prepared structural evidence without guessing topic or page continuity.

The catalog is an extraction artifact. It can extend a retrieved block only
while its source bytes still match. Every extension gets its own identity and
locator; physical adjacency alone is never a semantic relationship.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Mapping
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from app.services.corpus import valid_reviewed_relation
from app.services.reviewed_visual_regions import valid_literal_render_relation

_RELATIONS = frozenset({"parent_section", "parent_table", "table_header", "table_note_for", "table_caption_for", "defines", "continued_table", "literal_render_of"})
_FORMATS = frozenset({".pdf", ".chm", ".html", ".htm", ".docx", ".md", ".txt"})
_PHYSICAL_BATCH: ContextVar[dict[str, Any] | None] = ContextVar("physical_evidence_batch", default=None)


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_size


def _normalized(value: str) -> str:
    return " ".join(value.split())


def needs_section_context(question: str) -> bool:
    """Classify the requested operation, never the documented subject."""
    return bool(re.search(
        r"\b(?:как|порядок|последовательность|процедур\w*|обзор|расскажи\w*|сравни\w*|"
        r"ограничени\w*|услови\w*|how|procedure|steps|overview|compare|limitations?)\b",
        question, re.IGNORECASE))


class RelatedEvidenceIndex:
    """Thread-safe, lazy local resolver with bounded, separately citable output."""

    def __init__(self, data_root: Path, catalog_path: Path):
        self.data_root = data_root.resolve()
        self.catalog_path = catalog_path
        self._lock = RLock()
        self._catalog_stamp: tuple[int, int, int, int] | None = None
        self._catalog_resolved: Path | None = None
        self._sources: dict[str, dict[str, Any]] = {}
        self._locations: dict[str, dict[str, Any]] = {}
        self._blocks: dict[str, dict[str, Any]] = {}
        self._by_location: dict[str, list[dict[str, Any]]] = {}
        self._pdf_pages: dict[tuple[str, int], str | None] = {}
        self._incoming: dict[str, list[tuple[str, str]]] = {}
        self._source_checks: dict[str, tuple[tuple[Path, tuple[int, int, int, int]], bool]] = {}
        self._table_view: Any = None
        self._table_groups: dict[str, dict[str, Any]] = {}

    def _load(self) -> bool:
        batch = _PHYSICAL_BATCH.get()
        if batch is not None and batch["owner"] is self:
            return True
        try:
            stamp = _stamp(self.catalog_path)
            if stamp == self._catalog_stamp:
                return True
            raw = json.loads(self.catalog_path.read_text(encoding="utf-8-sig"))
            if raw.get("schema_version", 1) != 1:
                return False
            for key in ("sources", "locations", "blocks"):
                if not isinstance(raw.get(key), list) or not all(isinstance(row, dict) for row in raw[key]):
                    return False
            for row in raw["blocks"]:
                if (not isinstance(row.get("attributes", {}), dict)
                        or not isinstance(row.get("relations", []), list)
                        or not all(isinstance(rel, dict) for rel in row.get("relations", []))
                        or not isinstance(row.get("text", ""), str)):
                    return False
                bounds = row.get("bbox")
                if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 4
                                           or not all(isinstance(value, (float, int)) for value in bounds)):
                    return False
            sources = {str(row["source_id"]): row for row in raw["sources"]}
            locations = {str(row["location_id"]): row for row in raw["locations"]}
            blocks = {str(row["block_id"]): row for row in raw["blocks"]}
            if any(len(mapped) != len(raw[key]) for key, mapped in
                   (("sources", sources), ("locations", locations), ("blocks", blocks))):
                return False
            for evidence in blocks.values():
                for relation in evidence.get("relations", []):
                    if relation.get("kind") == "literal_render_of":
                        location = locations.get(str(evidence.get("location_id")), {})
                        if (location.get("source_hash") != sources.get(str(evidence.get("source_id")), {}).get("source_hash")
                                or not valid_literal_render_relation(relation, evidence, location, blocks)):
                            return False
                    if (relation.get("attributes", {}).get("method") == "reviewed_source_association"
                            or relation.get("kind") == "defines" and
                            evidence.get("attributes", {}).get("method") == "reviewed_visual_transcription"):
                        location = locations.get(str(evidence.get("location_id")), {})
                        if (location.get("source_hash") != sources.get(str(evidence.get("source_id")), {}).get("source_hash")
                                or not valid_reviewed_relation(relation, evidence, location, blocks)):
                            return False
            by_location: dict[str, list[dict[str, Any]]] = {}
            incoming: dict[str, list[tuple[str, str]]] = {}
            for block in blocks.values():
                by_location.setdefault(str(block["location_id"]), []).append(block)
                for relation in block.get("relations", []):
                    if relation.get("kind") in {"defines", "table_note_for", "table_caption_for", "parent_section", "parent_table", "literal_render_of"}:
                        incoming.setdefault(str(relation.get("to_id")), []).append(
                            (str(block["block_id"]), "inverse_" + str(relation["kind"])))
            for batch in by_location.values():
                batch.sort(key=lambda row: (int(row.get("ordinal", 0)), row["block_id"]))
            pdf_pages: dict[tuple[str, int], str | None] = {}
            for identity, location in locations.items():
                sid, page = str(location.get("source_id")), location.get("page")
                if (type(page) is int and page > 0
                        and Path(str(sources.get(sid, {}).get("relative_path", ""))).suffix.lower() == ".pdf"):
                    key = (sid, page)
                    pdf_pages[key] = None if key in pdf_pages else identity
            self._sources, self._locations, self._blocks = sources, locations, blocks
            self._pdf_pages = pdf_pages
            self._by_location = by_location
            self._incoming = incoming
            self._catalog_stamp = stamp
            self._catalog_resolved = self.catalog_path.resolve()
            self._source_checks.clear()
            self._table_view = None
            self._table_groups = {}
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # A corrupt or missing catalog leaves ordinary retrieved evidence
            # usable. It can never justify an extension from an older snapshot.
            self._catalog_stamp = None
            return False

    def _source_binding(self, source_id: str) -> tuple[Path, tuple[int, int, int, int]] | None:
        """Check the current resolved path as well as the file stamp."""
        source = self._sources.get(source_id)
        if not source:
            return None
        try:
            relative = Path(str(source["relative_path"]))
            path = (self.data_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(self.data_root) or path.suffix.lower() not in _FORMATS:
                return None
            return path, _stamp(path)
        except (OSError, ValueError, KeyError):
            return None

    def _valid_source(self, source_id: str) -> bool:
        batch = _PHYSICAL_BATCH.get()
        active = batch is not None and batch["owner"] is self
        if active and source_id in batch["sources"]:
            return batch["sources"][source_id][1]
        binding = self._source_binding(source_id)
        valid = False
        if binding is not None:
            previous = self._source_checks.get(source_id)
            if previous is not None and previous[0] == binding:
                valid = previous[1]
            else:
                try:
                    valid = hashlib.sha256(binding[0].read_bytes()).hexdigest() == self._sources[source_id].get("source_hash")
                    self._source_checks[source_id] = (binding, valid)
                except (OSError, ValueError, KeyError):
                    pass
        if active:
            batch["sources"][source_id] = (binding, valid)
        return valid

    def source_relative_path(self, metadata: Mapping[str, Any]) -> str | None:
        """Resolve a portable locator through the exact current source hash.

        Vector payload file_path may name the Windows ingestion host while
        retrieval runs in Linux. Catalog paths are corpus-relative POSIX paths;
        host paths never establish this binding or appear in public citations.
        """
        with self._lock:
            if not self._load():
                return None
            sid = str(metadata.get("source_id") or "")
            source = self._sources.get(sid)
            if (not source or metadata.get("source_hash") != source.get("source_hash")
                    or not self._valid_source(sid)):
                return None
            return str(source["relative_path"])

    def warmup(self, *, inventory: bool = False) -> dict[str, Any]:
        """Prepare the same validated cache used by requests, before readiness.

        File stamps are still checked on every lookup. Preloading is never
        permission to use evidence after its catalog or source bytes change.
        Missing optional catalogs retain the ordinary retrieval fallback.
        """
        with self._lock:
            loaded = self._load()
            valid = sum(self._valid_source(sid) for sid in self._sources) if loaded else 0
            if loaded and inventory:
                self._prepare_table_view()
            return {"catalog_loaded": loaded, "valid_sources": valid,
                    "invalid_sources": len(self._sources) - valid if loaded else 0}

    def _seed_blocks(self, metadata: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
        location_id = str(metadata.get("location_id") or "")
        source_id = str(metadata.get("source_id") or "")
        location = self._locations.get(location_id)
        if not location or location.get("source_id") != source_id:
            return []
        if (metadata.get("source_hash") != location.get("source_hash")
                or location.get("source_hash") != self._sources.get(source_id, {}).get("source_hash")
                or not self._valid_source(source_id)):
            return []
        candidates = self._by_location.get(location_id, [])
        normalized = _normalized(text)
        if not normalized:
            return []
        block_id = str(metadata.get("block_id") or "")
        declared_ids = metadata.get("block_ids")
        if isinstance(declared_ids, list) and 1 < len(declared_ids) <= 64:
            declared = [self._blocks.get(str(identity)) for identity in declared_ids]
            if (all(row and row.get("location_id") == location_id and row.get("source_id") == source_id for row in declared)
                    and normalized in _normalized("\n".join(str(row.get("text") or "") for row in declared))):
                return declared
        if block_id:
            block = self._blocks.get(block_id)
            if block and block.get("location_id") == location_id and block.get("source_id") == source_id:
                related_headers = [self._blocks.get(str(rel.get("to_id")))
                                   for rel in block.get("relations", []) if rel.get("kind") == "table_header"]
                context = "\n".join(str(row.get("text") or "") for row in related_headers
                                     if row and row.get("location_id") == location_id)
                original = str(block.get("text") or "")
                if normalized in _normalized(context + "\n" + original):
                    return [block]
            return []
        # Legacy records without catalog location/hash deliberately do not
        # reach this fallback. Only uniquely attributable new spans qualify.
        matches = [row for row in candidates if normalized in _normalized(str(row.get("text") or ""))]
        return matches if len(matches) == 1 else []

    def _prepare_table_view(self) -> None:
        """Build once per catalog snapshot, under the shared resolver lock."""
        from llama_index.core.schema import TextNode

        from app.services import lexical
        from app.services.catalog_tables import validated_table_group

        if self._table_view is not None:
            return
        members: dict[str, list[dict[str, Any]]] = {}
        for block in self._blocks.values():
            parents = {str(rel.get("to_id")) for rel in block.get("relations", [])
                       if rel.get("kind") == "parent_table"}
            if table_id := (block.get("attributes") or {}).get("table_id"):
                parents.add(str(table_id))
            for parent in parents:
                members.setdefault(parent, []).append(block)
        groups = {}
        for table_id, rows in members.items():
            table = self._blocks.get(table_id, {})
            group = validated_table_group(table, rows,
                source=self._sources.get(str(table.get("source_id")), {}),
                location=self._locations.get(str(table.get("location_id")), {}))
            if group:
                groups[group["id"]] = group
        self._table_groups = groups
        self._table_view = lexical.LexicalBM25Index.build_nodes([
            TextNode(id_=key, text=group["ranking_text"]) for key, group in groups.items()
        ], lexical.lexical_terms)

    def inventory_table(self, queries: list[str], *, product_versions: set[tuple[str, str]],
                        max_chars: int, diagnostics: dict[str, Any],
                        anchor_queries: list[str] | None = None) -> dict[str, Any] | None:
        """Find one exact table outside ordinary chunk top-k; never certify coverage."""
        from app.services import lexical

        diagnostics.update(origin="catalog_inventory", selected_blocks=0, omitted=[], bounded=False)
        with self._lock:
            if not self._load():
                diagnostics["catalog_status"] = "unavailable"
                return None
            diagnostics["catalog_status"] = "loaded"
            self._prepare_table_view()
            allowed = {key for key, group in self._table_groups.items()
                       if group["metadata"]["product"] not in {None, "", "unknown", "unresolved", "all"}
                       and (group["metadata"]["product"], group["metadata"]["version"]) in product_versions}
            # Query only matching product/version partitions, before the bounded
            # top-k, so another product's many tables cannot starve this one.
            scores: dict[str, float] = {}
            for query in list(dict.fromkeys(queries))[:5]:
                for node, score in self._table_view.query(lexical.inventory_subject_terms(query), 8, allowed_ids=allowed):
                    scores[node.node_id] = max(score, scores.get(node.node_id, 0.))
            diagnostics["candidate_count"] = len(scores)
            anchored = []
            for identity, score in scores.items():
                group = self._table_groups[identity]
                metadata = group["metadata"]
                # Product names and operation words cannot establish a topic.
                excluded = set(lexical.lexical_terms("midas " + metadata["product"].replace("_", " ")
                                                     + " " + metadata["version"]))
                heading = set(lexical.inventory_subject_terms(group["anchor_text"])) - excluded
                coverage = max((len(subject & heading) / len(subject)
                    if len(subject & heading) >= 2 else 0.
                    for query in (anchor_queries if anchor_queries is not None else queries)
                    if (subject := set(lexical.inventory_subject_terms(query)) - excluded)), default=0.)
                if coverage >= .5:
                    anchored.append((coverage, score, identity))
            anchored.sort(reverse=True)
            diagnostics["anchored_candidates"] = len(anchored)
            if (len(anchored) > 1 and anchored[0][0] == anchored[1][0]
                    and anchored[1][1] >= anchored[0][1] * .95
                    and self._table_groups[anchored[0][2]]["text"] != self._table_groups[anchored[1][2]]["text"]):
                diagnostics["omitted"].append({"reason": "ambiguous_table_anchor"})
                return None
            for _, _, identity in anchored:
                group = self._table_groups[identity]
                if not self._valid_source(group["metadata"]["source_id"]):
                    diagnostics["omitted"].append({"table_id": group["metadata"]["block_id"], "reason": "stale_source"})
                    continue
                if len(group["text"]) > max_chars:
                    diagnostics["bounded"] = True
                    diagnostics["omitted"].append({"table_id": group["metadata"]["block_id"], "reason": "atomic_table_budget"})
                    continue
                diagnostics["selected_blocks"] = len(group["metadata"]["block_ids"])
                # The cached BM25 view and provenance must never receive query
                # annotations or mutation from one request's citation merge.
                from copy import deepcopy
                return deepcopy(group)
            return None

    def physical_contexts(
        self, seeds: list[dict[str, Any]], *, max_chars: int = 6_000,
    ) -> list[dict[str, Any]]:
        """Build aligned physical groups under one checked local snapshot.

        Catalog/source validity is reused only inside this synchronous locked
        batch. Every touched source and the catalog must retain their resolved
        path and stamp at exit, or all output is discarded. No network work is
        performed here. Standalone location/neighbor calls keep fresh checks.
        """
        def invalidated(reason: str) -> list[dict[str, Any]]:
            return [{"groups": [], "diagnostics": [
                {"origin": "physical_location", "selected_blocks": 0, "omitted": [],
                 "bounded": False, "invalidated": True, "reason": reason},
                {"origin": "physical_neighbor", "selected_locations": 0, "omitted": [],
                 "bounded": False, "invalidated": True, "reason": reason},
            ]} for _ in seeds]

        with self._lock:
            outer = _PHYSICAL_BATCH.get()
            if outer is not None and outer["owner"] is self:
                raise RuntimeError("Nested physical evidence batch")
            try:
                catalog_path = self.catalog_path.resolve()
                if catalog_path != self._catalog_resolved:
                    self._catalog_stamp = None
                if not self._load():
                    return invalidated("unverified_catalog")
                catalog_binding = (catalog_path, self._catalog_stamp)
            except (OSError, ValueError):
                return invalidated("unverified_catalog")
            batch: dict[str, Any] = {"owner": self, "sources": {}}
            token = _PHYSICAL_BATCH.set(batch)
            try:
                output = []
                for seed in seeds:
                    local_report: dict[str, Any] = {}
                    neighbor_report: dict[str, Any] = {}
                    group = self.location_context(seed["metadata"], seed["text"],
                                                  max_chars=max_chars, diagnostics=local_report)
                    groups = [group] if group else []
                    groups.extend(self.nearby_contexts(seed["metadata"], seed["text"],
                                                      max_chars=max_chars, diagnostics=neighbor_report))
                    output.append({"groups": groups, "diagnostics": [local_report, neighbor_report]})
            finally:
                _PHYSICAL_BATCH.reset(token)
            try:
                changed = catalog_binding != (self.catalog_path.resolve(), _stamp(self.catalog_path))
                # Do not short-circuit: every touched source gets an exit check.
                for sid, (binding, _valid) in batch["sources"].items():
                    if self._source_binding(sid) != binding:
                        changed = True
            except (OSError, ValueError):
                changed = True
            return invalidated("batch_changed") if changed else output

    def location_context(
        self, metadata: Mapping[str, Any], text: str, *, max_chars: int = 6_000,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Complete one verified physical location, without asserting relations.

        Only usable extracted text is complete: omitted visuals/unsafe grids are
        recorded explicitly. Large locations have no arbitrary prefix fallback.
        """
        from app.services.catalog_tables import validated_table_group
        from app.services.corpus import stable_id

        report = diagnostics if diagnostics is not None else {}
        report.update(origin="physical_location", selected_blocks=0, omitted=[], bounded=False)
        with self._lock:
            if not self._load() or not self._seed_blocks(metadata, text):
                report["reason"] = "unverified_location"
                return None
            lid, sid = str(metadata["location_id"]), str(metadata["source_id"])
            location, source = self._locations[lid], self._sources[sid]
            if any(metadata.get(key) and metadata[key] != location.get(key, "unknown")
                   for key in ("product", "version")):
                report["reason"] = "scope_mismatch"
                return None
            rows = self._by_location.get(lid, [])
            if any(row.get("source_id") != sid for row in rows):
                report["reason"] = "invalid_location_membership"
                return None
            # Only mechanically validated DOM grids may expose their row text.
            # Parent containers are never repeated alongside their children.
            members: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                if row.get("kind") in {"table_row", "table_header"}:
                    members.setdefault(str((row.get("attributes") or {}).get("table_id")), []).append(row)
            safe_rows: set[str] = set()
            for row in rows:
                if row.get("kind") == "table":
                    group = validated_table_group(row, members.get(str(row["block_id"]), []),
                                                  source=source, location=location)
                    if group:
                        safe_rows.update(group["metadata"]["block_ids"])
            selected = []
            for row in rows:
                kind, attrs = row.get("kind"), row.get("attributes") or {}
                content = str(row.get("text") or "")
                reason = None
                if kind in {"table", "section"}:
                    reason = "structural_container"
                elif kind in {"image", "visual_unextracted", "figure"}:
                    reason = "untranscribed_visual"
                elif attrs.get("review_status") in {"pending_review", "rejected"}:
                    reason = "unreviewed_text"
                elif kind in {"table_row", "table_header"} and row["block_id"] not in safe_rows:
                    reason = "unsafe_grid"
                elif not content.strip():
                    reason = "empty_text"
                if reason:
                    report["omitted"].append({"block_id": row["block_id"], "reason": reason})
                else:
                    selected.append(row)
            content = "\n".join(row["text"] for row in selected)
            if not content or len(selected) > 64 or len(content) > max_chars:
                report.update(reason=("no_usable_text" if not content else "block_budget"
                                      if len(selected) > 64 else "character_budget"), bounded=bool(content),
                              required_chars=len(content), required_blocks=len(selected))
                return None
            ids = [str(row["block_id"]) for row in selected]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            identity = stable_id("physical-location-v1", sid, source["source_hash"], lid, *ids, digest)
            path = str(source["relative_path"])
            locator = path + (f"#page={location['page']}" if location.get("page") is not None
                              else "::" + str(location["chm_member"]) if location.get("chm_member") else "")
            report.update(selected_blocks=len(selected), required_chars=len(content))
            return {"block_id": identity, "text": content, "metadata": {
                "source_id": sid, "source_hash": source["source_hash"], "location_id": lid,
                "block_id": identity, "block_ids": ids, "doc_id": identity,
                "file_name": Path(path).name, "file_path": str(self.data_root / path),
                "source_locator": locator, "page": location.get("page"),
                "chm_member": location.get("chm_member"), "title": location.get("title"),
                "product": location.get("product", "unknown"), "version": location.get("version") or "unknown",
                "language": location.get("language") or "unknown", "section_path": [],
                "figure_regions": deepcopy(location.get("figure_regions") or []),
                "warnings": list(location.get("warnings") or []),
                "extraction_status": location.get("extraction_status"),
                "block_type": "location_text_group", "block_attributes": {
                    "complete_text_location": True, "extraction_only": True,
                    "omitted_block_count": len(report["omitted"]),
                    "visual_review_required": any((row.get("attributes") or {}).get("visual_review_required")
                                                  for row in rows),
                    "visual_region_count": len(location.get("figure_regions") or []),
                    "untranscribed_visual_regions_present": bool(location.get("figure_regions")) or any(
                        row["reason"] == "untranscribed_visual" for row in report["omitted"])},
                "atomic_evidence": True, "extraction_version": "corpus-location-group-v1",
                "origin": "physical_location", "text_sha256": digest,
            }}

    def nearby_contexts(
        self, metadata: Mapping[str, Any], text: str, *, max_chars: int = 6_000,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return whole physical PDF pages ±1, never semantic continuations.

        Each independently attributed page passes location_context's rendering
        and source-byte checks. Ambiguous page numbers never select an alias.
        The caller separately allocates the request's shared context budget.
        """
        report = diagnostics if diagnostics is not None else {}
        report.update(origin="physical_neighbor", selected_locations=0, omitted=[], bounded=False)
        with self._lock:
            if not self._load() or not self._seed_blocks(metadata, text):
                report["reason"] = "unverified_seed"
                return []
            sid, lid = str(metadata["source_id"]), str(metadata["location_id"])
            location, source = self._locations[lid], self._sources[sid]
            page = location.get("page")
            if (Path(str(source["relative_path"])).suffix.lower() != ".pdf"
                    or type(page) is not int or page < 1):
                report["reason"] = "unsupported_source"
                return []
            if any(metadata.get(key) and metadata[key] != location.get(key, "unknown")
                   for key in ("product", "version")):
                report["reason"] = "scope_mismatch"
                return []
            if self._pdf_pages.get((sid, page)) != lid:
                report["reason"] = "ambiguous_seed_page"
                return []
            output = []
            for neighbor_page in (page - 1, page + 1):
                key = (sid, neighbor_page)
                if key not in self._pdf_pages:
                    continue
                neighbor_id = self._pdf_pages[key]
                if neighbor_id is None:
                    report["omitted"].append({"page": neighbor_page, "reason": "ambiguous_page"})
                    continue
                neighbor = self._locations[neighbor_id]
                if (neighbor.get("source_hash") != location.get("source_hash")
                        or any(neighbor.get(key, "unknown") != location.get(key, "unknown")
                               for key in ("product", "version"))):
                    report["omitted"].append({"page": neighbor_page, "reason": "scope_mismatch"})
                    continue
                rows = self._by_location.get(neighbor_id, [])
                anchor = next((row for row in rows if str(row.get("text") or "").strip()), None)
                if anchor is None:
                    report["omitted"].append({"page": neighbor_page, "reason": "no_usable_text"})
                    continue
                local_report: dict[str, Any] = {}
                group = self.location_context({**neighbor, "block_id": anchor["block_id"]},
                                              anchor["text"], max_chars=max_chars, diagnostics=local_report)
                if group is None:
                    report["bounded"] |= bool(local_report.get("bounded"))
                    report["omitted"].append({"page": neighbor_page,
                                              "reason": local_report.get("reason", "unverified_location")})
                    continue
                group["metadata"].update(origin="physical_neighbor", related_to=[])
                output.append(group)
            report["selected_locations"] = len(output)
            return output

    @staticmethod
    def _navigation_id(binding: Mapping[str, Any], page: int) -> str:
        from app.services.corpus import stable_id

        return stable_id("physical-navigation-v1", *(str(binding[key]) for key in
                         ("source_id", "source_hash", "location_id", "product", "version")), str(page))

    def navigation(
        self, seeds: list[dict[str, Any]], *, max_chars: int = 10_000, max_sources: int = 2,
    ) -> dict[str, Any]:
        """Non-citable PDF page hints, enabled only by two verified seed pages.

        max_chars bounds the compact JSON list of model-visible fields; raw
        binding metadata stays on the server. Hints are literal first prose,
        excluding exact paragraphs repeated on at least three physical pages.
        Neither the manifest nor a page hint establishes engineering facts.
        """
        result: dict[str, Any] = {"items": [], "truncated": False}
        if max_chars < 2 or max_sources < 1:
            result["truncated"] = bool(seeds)
            return result
        with self._lock:
            if not self._load():
                return result
            scopes: dict[tuple[str, str, str, str], set[str]] = {}
            for seed in seeds:
                if not isinstance(seed, dict) or not isinstance(seed.get("metadata"), Mapping):
                    continue
                metadata, text = seed["metadata"], seed.get("text")
                if not isinstance(text, str) or not self._seed_blocks(metadata, text):
                    continue
                sid, lid = str(metadata["source_id"]), str(metadata["location_id"])
                location, source = self._locations[lid], self._sources[sid]
                page = location.get("page")
                if (Path(str(source["relative_path"])).suffix.lower() != ".pdf"
                        or type(page) is not int or page < 1 or self._pdf_pages.get((sid, page)) != lid
                        or any(metadata.get(key) and metadata[key] != location.get(key, "unknown")
                               for key in ("product", "version"))):
                    continue
                scope = (sid, str(source["source_hash"]), str(location.get("product", "unknown")),
                         str(location.get("version", "unknown")))
                scopes.setdefault(scope, set()).add(lid)
            # A single PDF source cannot silently combine conflicting seed scopes.
            scope_counts: dict[str, int] = {}
            for scope in scopes:
                scope_counts[scope[0]] = scope_counts.get(scope[0], 0) + 1
            eligible = [scope for scope, locations in scopes.items()
                        if len(locations) >= 2 and scope_counts[scope[0]] == 1]
            if len(eligible) > min(max_sources, 2):
                result["truncated"] = True
            visible: list[dict[str, Any]] = []
            considered = 0
            for sid, digest, product, version in eligible[:min(max_sources, 2)]:
                source = self._sources[sid]
                pages = sorted((page, lid) for (source_id, page), lid in self._pdf_pages.items()
                               if source_id == sid and lid is not None
                               and self._locations[lid].get("source_hash") == digest
                               and str(self._locations[lid].get("product", "unknown")) == product
                               and str(self._locations[lid].get("version", "unknown")) == version)
                if len(pages) > 100 - considered:
                    result["truncated"] = True
                pages = pages[:max(0, 100 - considered)]
                considered += len(pages)
                paragraphs: dict[str, list[str]] = {}
                occurrences: dict[str, set[str]] = {}
                for _page, lid in pages:
                    prose = [row["text"] for row in self._by_location.get(lid, [])
                             if row.get("source_id") == sid and row.get("kind") == "paragraph"
                             and str(row.get("text") or "").strip()
                             and (row.get("attributes") or {}).get("review_status") not in {"pending_review", "rejected"}]
                    paragraphs[lid] = prose
                    for paragraph in prose:
                        occurrences.setdefault(paragraph, set()).add(lid)
                for page, lid in pages:
                    hint = next((paragraph[:160] for paragraph in paragraphs[lid]
                                 if len(occurrences[paragraph]) < 3), "")
                    if not hint:
                        continue
                    binding = {"source_id": sid, "source_hash": digest, "location_id": lid,
                               "product": product, "version": version}
                    item = {"id": self._navigation_id(binding, page), "metadata": binding,
                            "page": page, "file_name": Path(source["relative_path"]).name,
                            "hint": hint, "not_citable": True}
                    projection = {key: value for key, value in item.items() if key != "metadata"}
                    if len(json.dumps([*visible, projection], ensure_ascii=False, separators=(",", ":"))) > max_chars:
                        result["truncated"] = True
                        continue
                    visible.append(projection)
                    result["items"].append(item)
            return result

    def read_navigation(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read one exact manifest binding; callers enforce manifest membership.

        The ID is a stable provenance binding, not an authorization token.
        Source bytes, scopes, page uniqueness and rendering are revalidated.
        """
        with self._lock:
            if not self._load() or not isinstance(item, Mapping) or item.get("not_citable") is not True:
                return None
            binding = item.get("metadata")
            if (not isinstance(binding, Mapping)
                    or set(binding) != {"source_id", "source_hash", "location_id", "product", "version"}):
                return None
            sid, lid = str(binding["source_id"]), str(binding["location_id"])
            source, location = self._sources.get(sid), self._locations.get(lid)
            if not source or not location:
                return None
            page = location.get("page")
            canonical = {"source_id": sid, "source_hash": source.get("source_hash"), "location_id": lid,
                         "product": str(location.get("product", "unknown")),
                         "version": str(location.get("version", "unknown"))}
            if (dict(binding) != canonical or location.get("source_id") != sid
                    or location.get("source_hash") != source.get("source_hash")
                    or Path(str(source["relative_path"])).suffix.lower() != ".pdf"
                    or type(page) is not int or page < 1 or self._pdf_pages.get((sid, page)) != lid
                    or item.get("page") != page or item.get("file_name") != Path(source["relative_path"]).name
                    or item.get("id") != self._navigation_id(canonical, page) or not self._valid_source(sid)):
                return None
            anchor = next((row for row in self._by_location.get(lid, []) if str(row.get("text") or "").strip()), None)
            if anchor is None:
                return None
            return self.location_context({**location, "block_id": anchor["block_id"]}, anchor["text"], max_chars=6_000)

    def resolve(
        self, metadata: Mapping[str, Any], text: str, *, question: str,
        max_blocks: int = 8, max_chars: int = 12_000,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact linked blocks; no source text is fabricated or truncated."""
        report = diagnostics if diagnostics is not None else {}
        report.update({"seed_block_id": metadata.get("block_id"), "catalog_status": "not_checked",
                       "selected_blocks": 0, "omitted": [], "bounded": False})
        if max_blocks <= 0 or max_chars <= 0:
            report.update({"bounded": True, "omitted": [{"reason": "expansion_budget_exhausted"}]})
            return []
        with self._lock:
            if not self._load():
                report["catalog_status"] = "unavailable"
                return []
            report["catalog_status"] = "loaded"
            seeds = self._seed_blocks(metadata, text)
            if not seeds:
                report["seed_status"] = "unverified"
                return []
            report["seed_status"] = "verified"
            source_id = str(metadata["source_id"])
            seed_ids = {str(row["block_id"]) for row in seeds}
            queue: list[tuple[int, int, str, str, int]] = []
            next_order = 0

            def enqueue(entries: Any) -> None:
                nonlocal next_order
                for identity, relation, depth in entries:
                    priority = (-1 if relation in {"parent_table", "parent_section", "continued_table"}
                                else 0 if relation in {"table_header", "table_note_for", "inverse_table_note_for", "defines", "inverse_defines", "literal_render_of", "inverse_literal_render_of"}
                                else 1 if "caption" in relation else 3)
                    heapq.heappush(queue, (priority, next_order, identity, relation, depth))
                    next_order += 1

            for seed in seeds:
                enqueue((str(rel.get("to_id") or ""), str(rel.get("kind")), 0)
                             for rel in seed.get("relations", []) if rel.get("kind") in _RELATIONS)
                enqueue((identity, relation, 0) for identity, relation in self._incoming.get(str(seed["block_id"]), [])
                             if relation != "inverse_parent_section" or needs_section_context(question))
                section = seed.get("section_path")
                if needs_section_context(question) and section:
                    enqueue((str(row["block_id"]), "same_section", 0)
                                 for row in self._by_location.get(str(seed["location_id"]), [])
                                 if row.get("section_path") == section)
            seen = set(seed_ids)
            output: list[dict[str, Any]] = []
            used = 0
            while queue:
                _, _, block_id, relation, depth = heapq.heappop(queue)
                if block_id in seen:
                    continue
                seen.add(block_id)
                block = self._blocks.get(block_id)
                if not block or block.get("source_id") != source_id:
                    continue
                location = self._locations.get(str(block.get("location_id")))
                if not location or location.get("source_id") != source_id:
                    continue
                if location.get("source_hash") != metadata.get("source_hash"):
                    continue
                # Structural parents can legitimately have no searchable text.
                # Follow their explicit links before skipping their empty body.
                if depth < 2:
                    enqueue((str(rel.get("to_id") or ""), str(rel.get("kind")), depth + 1)
                                 for rel in block.get("relations", []) if rel.get("kind") in _RELATIONS)
                    enqueue((identity, rel, depth + 1) for identity, rel in
                                 self._incoming.get(str(block["block_id"]), [])
                                 if rel != "inverse_parent_section" or needs_section_context(question))
                content = str(block.get("text") or "")
                # Image placeholders and unresolved transcriptions are useful
                # diagnostics, never textual evidence of a diagram/formula.
                if (not content.strip() or block.get("kind") in {"image", "visual_unextracted", "figure"}
                        or (block.get("attributes") or {}).get("review_status") in {"pending_review", "rejected"}):
                    continue
                if used + len(content) > max_chars:
                    report["bounded"] = True
                    if len(report["omitted"]) < 16:
                        report["omitted"].append({"block_id": block_id, "relation": relation, "reason": "character_budget"})
                    continue
                if len(output) >= max_blocks:
                    report["bounded"] = True
                    report["omitted"].append({"block_id": block_id, "relation": relation, "reason": "block_budget"})
                    report["remaining_queue_unexamined"] = True
                    break
                path = str(self._sources[source_id]["relative_path"])
                locator = path
                if location.get("page") is not None:
                    locator += f"#page={location['page']}"
                elif location.get("chm_member"):
                    locator += "::" + str(location["chm_member"])
                if block.get("anchor"):
                    locator += "#" + str(block["anchor"])
                output.append({
                    "block_id": block_id, "text": content,
                    "metadata": {
                        "source_id": source_id, "source_hash": metadata["source_hash"],
                        "location_id": block["location_id"], "block_id": block_id,
                        "file_name": Path(path).name, "file_path": str(self.data_root / path),
                        "page": location.get("page"), "chm_member": location.get("chm_member"),
                        "product": location.get("product", "unknown"), "version": location.get("version", "unknown"),
                        "language": location.get("language", "unknown"), "section_path": block.get("section_path", []),
                        "source_locator": locator, "block_bbox": block.get("bbox"), "block_type": block.get("kind"),
                        "block_attributes": block.get("attributes") or {},
                        "block_relations": block.get("relations") or [],
                        "atomic_evidence": block.get("kind") in {"table_row", "table_header", "formula", "formula_candidate", "definition"},
                        "extraction_version": "corpus-linked-v1", "origin": "structural_relation",
                        "relation": relation, "related_to": sorted(seed_ids),
                        "text_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    },
                })
                used += len(content)
            report["selected_blocks"] = len(output)
            return output
