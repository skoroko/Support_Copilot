"""One literal DOM-to-evidence parser for standalone HTML and CHM members."""
from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from app.services.corpus import EvidenceBlock, block

_CHARSET_RE = re.compile(br"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.IGNORECASE)


def decode_html(raw: bytes) -> str:
    match = _CHARSET_RE.search(raw[:4096])
    encodings = ([match.group(1).decode("ascii")] if match else []) + ["utf-8", "windows-1251", "latin-1"]
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def parse_html(raw: bytes, location_id: str) -> dict[str, Any]:
    soup = BeautifulSoup(decode_html(raw), "html.parser")
    for hidden in soup(["script", "style", "noscript"]):
        hidden.decompose()
    title = _text(soup.title) if soup.title else ""
    language = str(soup.html.get("lang") or "unknown") if soup.html else "unknown"
    blocks: list[EvidenceBlock] = []
    headings: list[tuple[int, str, str]] = []
    relations: list[dict[str, str]] = []

    def emit(kind: str, text: str, tag: Tag, attributes: dict[str, Any] | None = None) -> EvidenceBlock:
        item = block(location_id, len(blocks), kind, text,
                     section_path=[h[1] for h in headings], anchor=str(tag.get("id") or tag.get("name") or "") or None,
                     attributes=attributes or {})
        if headings and kind != "heading":
            item.relations.append({"kind": "parent_section", "to_id": headings[-1][2]})
        blocks.append(item)
        return item

    def table(tag: Tag) -> None:
        rows = [row for row in tag.find_all("tr") if row.find_parent("table") is tag]
        if not rows:
            return
        # One-cell layout wrappers carry prose, not a data table.
        if (not any(len(r.find_all(["td", "th"], recursive=False)) > 1 for r in rows)
                and not tag.find("th") and not tag.find("caption", recursive=False)
                and (len(rows) == 1 or tag.find("table"))):
            for row in rows:
                for cell in row.find_all(["td", "th"], recursive=False):
                    walk(cell)
            return
        caption = tag.find("caption", recursive=False)
        table_block = emit("table", _text(caption) if caption else "", tag,
                           {"structure": "html_dom", "header_inference": "explicit_th_only"})
        spans: dict[int, tuple[str, int, str, int]] = {}
        headers: list[EvidenceBlock] = []
        for row_number, row in enumerate(rows):
            grid = dict(spans)
            spans = {col: (value[0], value[1], value[2], value[3] - 1)
                     for col, value in spans.items() if value[3] > 1}
            column = 0
            cells_data = []
            invalid = False
            for cell in row.find_all(["td", "th"], recursive=False):
                while column in grid:
                    column += 1
                try:
                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))
                except (ValueError, TypeError):
                    invalid = True
                    break
                if not (1 <= colspan <= 32 and 1 <= rowspan <= 1000 and column + colspan <= 32):
                    invalid = True
                    break
                text = _text(cell)
                origin = str(cell.get("id") or f"{table_block.block_id}:r{row_number}:c{column}")
                value = (text, row_number, origin, rowspan)
                cells_data.append({"column": column, "text": text, "colspan": colspan,
                                   "rowspan": rowspan, "header": cell.name == "th", "origin": origin})
                for index in range(column, column + colspan):
                    if index in grid:
                        invalid = True
                        break
                    grid[index] = value
                    if rowspan > 1:
                        spans[index] = (text, row_number, origin, rowspan - 1)
                column += colspan
            if invalid:
                emit("unresolved_table", _text(row), row, {"table_id": table_block.block_id,
                                                          "reason": "invalid_or_overlapping_cell_spans"})
                continue
            is_header = bool(cells_data) and all(c["header"] for c in cells_data)
            text = " | ".join(f"Column {col + 1}={grid[col][0] or '(empty)'}" for col in sorted(grid))
            item = emit("table_header" if is_header else "table_row", text, row,
                        {"table_id": table_block.block_id, "row": row_number,
                         "cells": cells_data, "resolved_cells": [{"column": col, "text": v[0],
                         "origin_row": v[1], "origin": v[2]} for col, v in sorted(grid.items())]})
            item.relations.append({"kind": "parent_table", "to_id": table_block.block_id})
            if is_header:
                headers.append(item)
            else:
                for header in headers:
                    item.relations.append({"kind": "table_header", "to_id": header.block_id})
            # Extract literal links/images inside cells as separate evidence too.
            for child in row.find_all(["a", "img", "math"]):
                walk(child)
        described_by = str(tag.get("aria-describedby") or "").split()
        for identity in described_by:
            note = soup.find(id=identity)
            if note:
                item = emit("table_note", _text(note), note, {"table_id": table_block.block_id})
                item.relations.append({"kind": "table_note_for", "to_id": table_block.block_id})

    def walk(tag: Tag) -> None:
        if tag.name == "figure":
            first = len(blocks)
            for child in tag.children:
                if isinstance(child, Tag):
                    walk(child)
            images = [b for b in blocks[first:] if b.kind == "image"]
            captions = [b for b in blocks[first:] if b.kind == "caption"]
            for image in images:
                for caption in captions:
                    image.relations.append({"kind": "figure_caption", "to_id": caption.block_id})
            return
        if re.fullmatch(r"h[1-6]", tag.name or ""):
            level = int(tag.name[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            item = emit("heading", _text(tag), tag, {"level": level})
            if headings:
                item.relations.append({"kind": "parent_section", "to_id": headings[-1][2]})
            headings.append((level, item.text, item.block_id))
            item.section_path = [h[1] for h in headings]
            return
        if tag.name == "table":
            table(tag)
            return
        if tag.name == "img":
            emit("image", str(tag.get("alt") or ""), tag,
                 {"src": str(tag.get("src") or ""), "title": str(tag.get("title") or ""),
                  "interpretation": "not_performed"})
            return
        if tag.name == "a":
            emit("link" if tag.get("href") else "anchor", _text(tag), tag,
                 {"href": str(tag.get("href") or "")})
            return
        if tag.name == "math" or tag.has_attr("data-formula"):
            emit("formula", tag.get_text("", strip=False), tag,
                 {"markup": str(tag), "transcription": "source_text_only", "verified": False})
            return
        if tag.name == "dl":
            term: EvidenceBlock | None = None
            for child in tag.children:
                if not isinstance(child, Tag):
                    continue
                if child.name == "dt":
                    term = emit("definition_term", _text(child), child)
                elif child.name == "dd":
                    item = emit("definition", _text(child), child)
                    if term:
                        item.relations.append({"kind": "defines", "to_id": term.block_id})
            return
        if tag.name in {"p", "li", "pre", "figcaption", "caption"}:
            text = tag.get_text("", strip=False) if tag.name == "pre" else _text(tag)
            if text.strip():
                emit("caption" if tag.name in {"figcaption", "caption"} else "paragraph", text, tag)
            for child in tag.find_all(["a", "img", "math"]):
                walk(child)
            return
        for child in tag.children:
            if isinstance(child, Tag):
                walk(child)
            elif isinstance(child, NavigableString) and str(child).strip():
                emit("paragraph", " ".join(str(child).split()), tag)

    walk(soup.body or soup)
    for item in blocks:
        for relation in item.relations:
            relations.append({"from_id": item.block_id, **relation})
    return {"title": title, "language": language, "blocks": [item.as_dict() for item in blocks],
            "relations": relations, "text": "\n\n".join(item.text for item in blocks if item.text),
            # Each block has its own heading ancestry; a whole topic does not inherit its final heading.
            "section_path": []}
