"""Format-aware readers with physical coverage and literal source evidence."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
    TransformComponent,
)

from app.services.corpus import EXTRACTION_VERSION, block, stable_id
from app.services.corpus_html import parse_html


class StructuredHTMLReader(BaseReader):
    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        location = stable_id("location", (extra_info or {}).get("source_id", str(Path(file).resolve())), "html")
        extracted = parse_html(Path(file).read_bytes(), location)
        metadata = dict(extra_info or {})
        metadata.update({"location_id": location, "extraction_version": EXTRACTION_VERSION,
                         "extraction_status": "text" if extracted["text"].strip() else "no_text",
                         "evidence_blocks": extracted["blocks"], "relations": extracted["relations"],
                         "title": extracted["title"], "language": extracted["language"],
                         "section_path": extracted["section_path"]})
        return [Document(text=extracted["text"], metadata=metadata)]


def _bbox(obj: dict[str, Any]) -> list[float]:
    return [round(float(obj[k]), 4) for k in ("x0", "top", "x1", "bottom")]


def _inside(word: dict[str, Any], boxes: list[list[float]]) -> bool:
    x, y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return any(left <= x <= right and top <= y <= bottom for left, top, right, bottom in boxes)


def _layout_lines(page: Any, table_boxes: list[list[float]]) -> list[dict[str, Any]]:
    """Separate a repeated geometric gutter; never join sentences across columns.

    This is a two-column candidate detector, not an exhaustive reading-order model.
    At least three rows and a majority of body rows must support a wide empty gap.
    Full-width lines split independent top-to-bottom bands around the columns.
    """
    def from_chars(chars: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
        chars = sorted(chars, key=lambda c: (c["x0"], c["top"]))
        typical_width = median(max(.1, c["x1"] - c["x0"]) for c in chars)
        text, previous = "", None
        for char in chars:
            if previous is not None and char["x0"] - previous["x1"] > typical_width * .35 and not text.endswith(" "):
                text += " "
            text += char["text"]
            previous = char
        return {**base, "text": text.strip(), "chars": chars,
                "x0": min(c["x0"] for c in chars), "x1": max(c["x1"] for c in chars),
                "top": min(c["top"] for c in chars), "bottom": max(c["bottom"] for c in chars)}

    lines = []
    for original in page.extract_text_lines(layout=False, strip=True, return_chars=True):
        chars = original.get("chars", [])
        selected = [char for char in chars if not _inside(char, table_boxes)]
        if selected:
            lines.append(from_chars(selected, original) if len(selected) != len(chars) else dict(original))
    words = [word for word in page.extract_words() if not _inside(word, table_boxes)]
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if rows and abs(rows[-1][0]["top"] - word["top"]) <= 3:
            rows[-1].append(word)
        else:
            rows.append([word])
    gaps: list[tuple[float, float, int]] = []
    for index, row in enumerate(rows):
        row.sort(key=lambda item: item["x0"])
        for position, (left, right) in enumerate(pairwise(row)):
            # Two words on each side avoid treating list/definition indentation as columns.
            if position < 1 or len(row) - position - 1 < 2:
                continue
            start, end = float(left["x1"]), float(right["x0"])
            if end - start >= max(28.0, float(page.width) * .07) and .2 * page.width < (start + end) / 2 < .8 * page.width:
                gaps.append((start, end, index))
    candidates = [(start + end) / 2 for start, end, _ in gaps]
    gutter = None
    if candidates:
        support = lambda value: {index for start, end, index in gaps if start + 4 <= value <= end - 4}
        candidate = max(candidates, key=lambda value: (len(support(value)), -abs(value - page.width / 2)))
        if len(support(candidate)) >= 3 and len(support(candidate)) >= max(1, len(rows)) * .55:
            gutter = candidate
    if gutter is None:
        for line in lines:
            line["column_detection"] = "no_confirmed_gutter"
            line["column_candidate"] = None
        return lines

    full: list[dict[str, Any]] = []
    divided: list[dict[str, Any]] = []
    for line in lines:
        line_words = [w for w in words if line["top"] - 2 <= (w["top"] + w["bottom"]) / 2 <= line["bottom"] + 2]
        crosses = any(w["x0"] < gutter < w["x1"] for w in line_words)
        chars = line.get("chars", [])
        if crosses or not chars:
            line["column_detection"] = "full_width_band"
            line["column_candidate"] = None
            full.append(line)
            continue
        for column in (0, 1):
            selected = [c for c in chars if ((c["x0"] + c["x1"]) / 2 < gutter) == (column == 0)]
            if not selected:
                continue
            divided.append(from_chars(selected, {"column_candidate": column,
                                                "column_detection": "repeated_geometric_gutter",
                                                "gutter_x": round(gutter, 4)}))
    full.sort(key=lambda line: (line["top"], line["x0"]))
    output = []
    for band in range(len(full) + 1):
        top = full[band - 1]["bottom"] if band else float('-inf')
        bottom = full[band]["top"] if band < len(full) else float('inf')
        output.extend(sorted((line for line in divided if top - 2 <= line["top"] < bottom),
                             key=lambda line: (line["column_candidate"], line["top"], line["x0"])))
        if band < len(full):
            output.append(full[band])
    return output


def _outline_sections(reader: Any) -> dict[int, list[str]]:
    """Only PDF's explicit outline destinations establish sections."""
    sections: dict[int, list[str]] = {}
    def visit(items: list[Any], parents: list[str]) -> None:
        last: list[str] = parents
        for item in items:
            if isinstance(item, list):
                visit(item, last)
                continue
            title = str(getattr(item, "title", "") or "")
            if not title:
                continue
            page = reader.get_destination_page_number(item)
            if page is not None and page >= 0:
                last = parents + [title]
                sections[page + 1] = last
    try:
        visit(reader.outline, [])
    except Exception:  # noqa: BLE001 - broken optional outlines cannot erase physical coverage
        return sections
    return sections


class StructuredPDFReader(BaseReader):
    """Keep all physical pages; record layout/tables/regions, without claiming OCR."""
    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        import pdfplumber
        from pypdf import PdfReader

        source = (extra_info or {}).get("source_id", str(Path(file).resolve()))
        reader = PdfReader(str(file))
        total = len(reader.pages)
        sections = _outline_sections(reader)
        documents: list[Document] = []
        current_section: list[str] = []
        with pdfplumber.open(file) as pdf:
            for number in range(1, total + 1):
                location = stable_id("location", source, number)
                if number in sections:
                    current_section = sections[number]
                metadata = dict(extra_info or {})
                metadata.update({"location_id": location, "page": number, "total_pages": total,
                                 "extraction_version": EXTRACTION_VERSION, "language": "unknown",
                                 "section_path": list(current_section), "relations": [], "warnings": []})
                for neighbor, kind in ((number - 1, "previous_page"), (number + 1, "next_page")):
                    if 1 <= neighbor <= total:
                        metadata["relations"].append({"kind": kind, "from_id": location,
                                                      "to_id": stable_id("location", source, neighbor)})
                try:
                    page = pdf.pages[number - 1]
                    metadata["page_bbox"] = [0.0, 0.0, float(page.width), float(page.height)]
                    items, figures, warnings = self._extract_page(page, location, current_section)
                    metadata["evidence_blocks"] = items
                    metadata["figure_regions"] = figures
                    metadata["warnings"].extend(warnings)
                    for item in items:
                        metadata["relations"].extend({"from_id": item["block_id"], **rel}
                                                     for rel in item.get("relations", []))
                    text = "\n\n".join(item["text"] for item in items if item["text"])
                    if not text.strip():
                        # pypdf is a text fallback, not OCR, and cannot infer missing formula symbols.
                        text = reader.pages[number - 1].extract_text() or ""
                        if text.strip():
                            metadata["evidence_blocks"] = [block(location, 0, "paragraph", text,
                                                                  section_path=list(current_section),
                                                                  attributes={"method": "pypdf_fallback", "layout": "unavailable"}).as_dict()]
                            metadata["warnings"].append("layout_text_unavailable_pypdf_fallback")
                    metadata["extraction_status"] = "text" if text.strip() else "no_text"
                    if not text.strip() and figures:
                        metadata["warnings"].append("image_regions_present_visual_review_required")
                    # Release pdfminer page caches during full-corpus processing.
                    page.close()
                except Exception as exc:  # noqa: BLE001 - keep per-page failures in the ledger
                    text = ""
                    metadata.update({"extraction_status": "unreadable", "extraction_error": type(exc).__name__,
                                     "evidence_blocks": []})
                documents.append(Document(text=text, metadata=metadata))
        return documents

    @staticmethod
    def _extract_page(page: Any, location: str, section: list[str]) -> tuple[list[dict[str, Any]], list[Any], list[str]]:
        items = []
        warnings: list[str] = []
        figures = []
        for image in page.images:
            stream = image.get("stream")
            try:
                image_hash = hashlib.sha256(stream.get_data()).hexdigest() if stream is not None else None
                image_error = None
            except Exception as exc:  # noqa: BLE001 - an image decoder cannot erase text/region coverage
                image_hash = None
                image_error = type(exc).__name__
            region = {"kind": "image", "bbox": _bbox(image), "image_sha256": image_hash,
                      "interpretation": "not_performed", "visual_review_required": True,
                      "image_decode_error": image_error}
            figures.append(region)
            items.append(block(location, len(items), "visual_unextracted", "", section_path=list(section),
                               bbox=region["bbox"], attributes={**region, "quality_status": "visual_content_unextracted",
                                                             "ocr_performed": False}).as_dict())
        # Vector geometry is recorded as regions, never automatically labelled a semantic figure/formula.
        for shape in [*page.curves, *page.rects]:
            if float(shape["x1"]) - float(shape["x0"]) > 4 and float(shape["bottom"]) - float(shape["top"]) > 4:
                figures.append({"kind": "vector_region", "bbox": _bbox(shape), "interpretation": "not_performed"})
        try:
            tables = page.find_tables()
        except Exception as exc:  # noqa: BLE001 - record detector failure without losing prose
            tables = []
            warnings.append("table_detection_failed:" + type(exc).__name__)
        table_boxes = []
        for table in tables:
            rows = table.extract()
            if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2 or not any(any(c for c in row) for row in rows):
                continue
            bounds = [round(float(v), 4) for v in table.bbox]
            table_boxes.append(bounds)
            parent = block(location, len(items), "table", "", section_path=list(section), bbox=bounds,
                           attributes={"structure": "pdfplumber_lines", "header_status": "unknown",
                                       "quality_status": "extracted_unverified", "visual_review_required": True})
            items.append(parent.as_dict())
            for row_index, row in enumerate(rows):
                text = " | ".join(f"Column {col + 1}={value or '(empty)'}" for col, value in enumerate(row))
                row_bbox = table.rows[row_index].bbox if row_index < len(table.rows) else table.bbox
                item = block(location, len(items), "table_row", text, section_path=list(section),
                             bbox=[round(float(v), 4) for v in row_bbox],
                             attributes={"table_id": parent.block_id, "row": row_index,
                                         "cells": [{"column": col, "text": value or ""} for col, value in enumerate(row)],
                                         "header_status": "unknown", "quality_status": "extracted_unverified",
                                         "visual_review_required": True},
                             relations=[{"kind": "parent_table", "to_id": parent.block_id}])
                items.append(item.as_dict())
        # Lines retain exact extracted glyph transcription and a source region.
        for line in _layout_lines(page, table_boxes):
            text = line["text"]
            formula_candidate = bool(re.search(r"[=∑∫√≠≤≥]", text))
            chars = line.get("chars", [])
            typography_variation = len({round(float(c.get("size", 0)), 1) for c in chars}) > 1
            formula_candidate = formula_candidate or typography_variation
            attributes = {"method": "pdfplumber_text_line", "raw_char_text": "".join(c["text"] for c in chars),
                          "transcription": "source_text_only", "verified": False,
                          "reading_order": "layout_candidate_unverified", "visual_review_required": formula_candidate}
            attributes.update({"column_candidate": line.get("column_candidate"),
                               "column_detection": line.get("column_detection"), "gutter_x": line.get("gutter_x")})
            if formula_candidate:
                attributes["glyphs"] = [{"text": c["text"], "font": c.get("fontname"), "size": c.get("size"),
                                         "bbox": _bbox(c)} for c in chars]
            item = block(location, len(items), "formula_candidate" if formula_candidate else "paragraph", text,
                         section_path=list(section), bbox=_bbox(line),
                         attributes=attributes)
            items.append(item.as_dict())
        # Spatially explicit captions/notes may be related; no first-row/header guessing.
        for item in items:
            if item["kind"] != "paragraph" or not item.get("bbox"):
                continue
            candidates = [t for t in items if t["kind"] == "table" and t.get("bbox")
                          and t["bbox"][0] - 20 <= item["bbox"][0] <= t["bbox"][2]]
            if re.match(r"(?i)^(?:table|таблица)\s+\d", item["text"]):
                candidates = [t for t in candidates if 0 <= t["bbox"][1] - item["bbox"][3] <= 40]
                kind = "table_caption_for"
            elif re.match(r"(?i)^(?:note|notes|примечание|примечания)\s*[:.]", item["text"]):
                candidates = [t for t in candidates if 0 <= item["bbox"][1] - t["bbox"][3] <= 40]
                kind = "table_note_for"
            else:
                continue
            if len(candidates) == 1:
                item["relations"].append({"kind": kind, "to_id": candidates[0]["block_id"]})
        return items, figures, warnings


class EvidenceSplitter(TransformComponent):
    """Split prose, keep table rows/formulas atomic with literal header context."""
    chunk_size: int = 1024
    chunk_overlap: int = 64

    def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> list[TextNode]:
        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                    include_metadata=False)
        output: list[TextNode] = []
        for document in nodes:
            blocks = document.metadata.get("evidence_blocks") or []
            by_id = {item["block_id"]: item for item in blocks}
            if not blocks:
                blocks = [{"block_id": stable_id(document.id_, "text"), "kind": "paragraph",
                           "text": document.text, "relations": []}]
            # Adjacent prose lines are one split unit; this avoids one vector per PDF line.
            units: list[dict[str, Any]] = []
            for item in blocks:
                if not item.get("text", "").strip():
                    continue
                if (item["kind"] == "paragraph" and units and units[-1]["kind"] == "paragraph"
                        and units[-1].get("section_path") == item.get("section_path")
                        and units[-1].get("attributes", {}).get("column_candidate") == item.get("attributes", {}).get("column_candidate")):
                    units[-1]["text"] += "\n" + item["text"]
                    units[-1].setdefault("block_ids", [units[-1]["block_id"]]).append(item["block_id"])
                else:
                    units.append(dict(item))
            ordinal = 0
            for item in units:
                text = item["text"]
                headers = [by_id[rel["to_id"]]["text"] for rel in item.get("relations", [])
                           if rel["kind"] == "table_header" and rel["to_id"] in by_id]
                if headers:
                    text = "\n".join([*headers, text])
                atomic = item["kind"] in {"table_row", "table_header", "formula", "formula_candidate", "definition"}
                pieces = [text] if atomic else splitter.split_text(text)
                for piece in pieces:
                    metadata = {k: v for k, v in document.metadata.items() if k != "evidence_blocks"}
                    metadata.update({"block_id": item["block_id"], "block_type": item["kind"],
                                     "block_ids": item.get("block_ids", [item["block_id"]]),
                                     "block_bbox": item.get("bbox"), "block_attributes": item.get("attributes", {}),
                                     "section_path": item.get("section_path", metadata.get("section_path", [])),
                                     "block_relations": item.get("relations", []), "atomic_evidence": atomic})
                    excluded = sorted(set(document.excluded_embed_metadata_keys) | set(metadata))
                    node = TextNode(text=piece, id_=str(uuid5(NAMESPACE_URL, f"{document.id_}:chunk:{ordinal}")),
                                    metadata=metadata, excluded_embed_metadata_keys=excluded,
                                    excluded_llm_metadata_keys=sorted(metadata),
                                    relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=document.id_)})
                    output.append(node)
                    ordinal += 1
        return output
