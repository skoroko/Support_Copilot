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
from pathlib import Path
from threading import RLock
from typing import Any

from app.services.corpus import valid_reviewed_relation
from app.services.reviewed_visual_regions import valid_literal_render_relation

_RELATIONS = frozenset({"parent_section", "parent_table", "table_header", "table_note_for", "table_caption_for", "defines", "continued_table", "literal_render_of"})
_FORMATS = frozenset({".pdf", ".chm", ".html", ".htm", ".docx", ".md", ".txt"})


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
        self._sources: dict[str, dict[str, Any]] = {}
        self._locations: dict[str, dict[str, Any]] = {}
        self._blocks: dict[str, dict[str, Any]] = {}
        self._by_location: dict[str, list[dict[str, Any]]] = {}
        self._incoming: dict[str, list[tuple[str, str]]] = {}
        self._source_checks: dict[str, tuple[tuple[int, int, int, int], bool]] = {}

    def _load(self) -> bool:
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
            self._sources, self._locations, self._blocks = sources, locations, blocks
            self._by_location = by_location
            self._incoming = incoming
            self._catalog_stamp = stamp
            self._source_checks.clear()
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # A corrupt or missing catalog leaves ordinary retrieved evidence
            # usable. It can never justify an extension from an older snapshot.
            self._catalog_stamp = None
            return False

    def _valid_source(self, source_id: str) -> bool:
        source = self._sources.get(source_id)
        if not source:
            return False
        try:
            relative = Path(str(source["relative_path"]))
            path = (self.data_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(self.data_root) or path.suffix.lower() not in _FORMATS:
                return False
            stamp = _stamp(path)
            previous = self._source_checks.get(source_id)
            if previous is not None and previous[0] == stamp:
                return previous[1]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            valid = digest == source.get("source_hash")
            self._source_checks[source_id] = (stamp, valid)
            return valid
        except (OSError, ValueError, KeyError):
            return False

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

    def warmup(self) -> dict[str, Any]:
        """Prepare the same validated cache used by requests, before readiness.

        File stamps are still checked on every lookup. Preloading is never
        permission to use evidence after its catalog or source bytes change.
        Missing optional catalogs retain the ordinary retrieval fallback.
        """
        with self._lock:
            loaded = self._load()
            valid = sum(self._valid_source(sid) for sid in self._sources) if loaded else 0
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
