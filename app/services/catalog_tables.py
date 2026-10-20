"""Literal, bounded HTML table groups; no inferred headers or topic relations."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.corpus import stable_id


def validated_table_group(table: dict[str, Any], members: list[dict[str, Any]], *,
                          source: dict[str, Any], location: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct the DOM grid and check every recorded cell/span and row.

    The caller must separately validate current source bytes. A successful
    check establishes literal table membership, never engineering correctness
    or semantic verification of the first row's labels.
    """
    sid, lid, tid = source.get("source_id"), location.get("location_id"), table.get("block_id")
    attributes = table.get("attributes") or {}
    if (not all(isinstance(value, str) and value for value in (sid, lid, tid, source.get("source_hash")))
            or source.get("format") not in {"html", "htm", "chm"}
            or location.get("source_id") != sid or location.get("source_hash") != source["source_hash"]
            or table.get("source_id") != sid or table.get("location_id") != lid
            or table.get("kind") != "table" or attributes.get("structure") != "html_dom"
            or attributes.get("visual_review_required") or attributes.get("review_status") in {"pending_review", "rejected"}
            or not 2 <= len(members) <= 64):
        return None
    try:
        rows = sorted(members, key=lambda row: row["attributes"]["row"])
        if [row["attributes"]["row"] for row in rows] != list(range(len(rows))):
            return None
        spans: dict[int, tuple[str, str, int, int]] = {}
        origins: set[str] = set()
        width = None
        for number, row in enumerate(rows):
            attrs = row["attributes"]
            if (row.get("kind") not in {"table_row", "table_header"}
                    or row.get("source_id") != sid or row.get("location_id") != lid
                    or type(attrs["row"]) is not int or attrs.get("table_id") != tid
                    or [rel.get("to_id") for rel in row.get("relations", [])
                        if rel.get("kind") == "parent_table"] != [tid]
                    or attrs.get("visual_review_required")
                    or attrs.get("review_status") in {"pending_review", "rejected"}):
                return None
            grid = {column: value for column, value in spans.items() if value[3] > number}
            spans = dict(grid)
            cells = attrs["cells"]
            if not isinstance(cells, list) or not cells:
                return None
            column = 0
            for cell in cells:
                while column in grid:
                    column += 1
                colspan, rowspan = cell["colspan"], cell["rowspan"]
                origin, text = cell["origin"], cell["text"]
                if (type(cell["column"]) is not int or cell["column"] != column
                        or type(colspan) is not int or type(rowspan) is not int
                        or not 1 <= colspan <= 32 or not 1 <= rowspan <= len(rows) - number
                        or column + colspan > 32 or not isinstance(text, str)
                        or not isinstance(origin, str) or not origin or origin in origins
                        or type(cell["header"]) is not bool):
                    return None
                origins.add(origin)
                value = (text, origin, number, number + rowspan)
                for target in range(column, column + colspan):
                    if target in grid:
                        return None
                    grid[target] = value
                    if rowspan > 1:
                        spans[target] = value
                column += colspan
            if set(grid) != set(range(len(grid))) or (width is not None and len(grid) != width):
                return None
            width = len(grid)
            if width < 2:
                return None
            resolved = [{"column": col, "text": grid[col][0], "origin": grid[col][1],
                         "origin_row": grid[col][2]} for col in range(width)]
            rendered = " | ".join(f"Column {col + 1}={grid[col][0] or '(empty)'}" for col in range(width))
            if attrs["resolved_cells"] != resolved or row["text"] != rendered:
                return None
        text = "\n".join(row["text"] for row in rows)
        # Bound both indexing work and a single atomic evidence supplement.
        if len(text) > 24_000 or len({row["block_id"] for row in rows}) != len(rows):
            return None
        identity = stable_id("catalog-table-v1", sid, source["source_hash"], tid,
                             hashlib.sha256(text.encode()).hexdigest())
        path = str(source["relative_path"])
        count_lines = []
        for column in range(width):
            counts = Counter(row['attributes']['resolved_cells'][column]['text'] for row in rows)
            if len(counts) > 8:
                continue
            line = json.dumps({'column': column + 1, 'counts': dict(counts)}, ensure_ascii=False)
            if len('\n'.join([*count_lines, line])) <= 1600:
                count_lines.append(line)
        metadata = {
            "source_id": sid, "source_hash": source["source_hash"], "location_id": lid,
            "block_id": tid, "block_ids": [row["block_id"] for row in rows],
            "block_type": "table_group", "block_attributes": {
                "structure": "html_dom", "header_inference": attributes.get("header_inference"),
                "group_row_count": len(rows),
                # Mechanical counts include row zero and imply no header or
                # engineering meaning. Only complete validated columns count.
                "literal_column_counts": '\n'.join(count_lines)},
            "doc_id": identity, "file_name": Path(path).name,
            "page": location.get("page"), "chm_member": location.get("chm_member"),
            "product": location.get("product", "unknown"), "version": location.get("version") or "unknown",
            "language": location.get("language") or "unknown", "title": location.get("title"),
            "section_path": table.get("section_path") or [], "anchor": table.get("anchor"),
            "atomic_evidence": True, "extraction_version": "corpus-table-group-v1",
            "origin": "catalog_inventory", "relation": "parent_table",
            "related_to": [tid], "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        # Ranking context is literal metadata, not an asserted nearby-intro link.
        anchor = "\n".join(value for value in [location.get("title") or "",
            *(table.get("section_path") or []), table.get("text") or "", rows[0]["text"]]
            if isinstance(value, str))
        return {"id": identity, "text": text, "anchor_text": anchor,
                "ranking_text": anchor + "\n" + text, "metadata": metadata}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
