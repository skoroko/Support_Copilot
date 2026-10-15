"""Present legacy CHM tables without inventing links between flattened cells.

The frozen vector chunks remain unchanged. Only complete rows that can be
mapped to the selected chunk are presented; neighboring rows are never added.
"""
from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup, CData, NavigableString
from pylibmspack import ChmArchive

from app.services.chm_reader import CHMReader


def _normalized(text: str) -> str:
    return " ".join(text.split())


def behavior_context(text: str, question: str) -> str:
    """Keep prose, not an applicability grid, as evidence of material behavior.

    Flattened PDF checkmarks describe element compatibility, not constitutive
    behavior. Keep those grids for compatibility questions; for behavior
    questions retain the preceding explicit description. No facts are added.
    """

    if not re.search(r"упруг|поведен|elastic|behavio[ur]", question, re.IGNORECASE):
        return text
    if re.search(r"конечн|тип\w*\s+элемент|применим|element|applicab|compatib", question, re.IGNORECASE):
        return text
    caption = re.search(
        r"(?:Таблица\s+\d+(?:\.\d+)*\.?\s+Применение\s+моделей\s+материалов\s+"
        r"в\s+зависимости\s+от\s+типа\s+конечного\s+элемента"
        r"|Table\s+\d+(?:\.\d+)*\.?\s+(?:Application|Applicability)\s+of\s+material\s+models\s+"
        r"(?:depending\s+on|by|for)\s+(?:the\s+)?(?:finite\s+)?element\s+type)",
        text,
        re.IGNORECASE,
    )
    return text[:caption.start()].rstrip() if caption else text


@lru_cache(maxsize=64)
def _read_member(path: str, _mtime: int, _size: int, member: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="support-copilot-evidence-") as directory:
        archive_path = CHMReader._ascii_compatible_path(Path(path), Path(directory))
        return ChmArchive(str(archive_path)).read(member, max_size=16 * 1024 * 1024)


def chm_context(data_dir: Path, file_name: str, member: str, text: str) -> str:
    """Read only the original selected member; unavailable structure fails closed."""

    matches = [path for path in data_dir.rglob("*.chm") if path.name == file_name]
    if len(matches) != 1:
        return ""
    path = matches[0]
    try:
        stat = path.stat()
        raw = _read_member(str(path), stat.st_mtime_ns, stat.st_size, member)
        return project_table_rows(text, raw)
    except Exception:  # noqa: BLE001 - an unavailable archive cannot justify table facts
        return ""


@lru_cache(maxsize=64)
def project_table_rows(chunk: str, html: bytes) -> str:
    """Replace intersecting flattened tables with complete explicit column rows.

    Repeated values from row/column spans retain their originating cell's
    position. A row is excluded when any originating cell falls outside the
    chunk, including a row truncated before its final behavior column.
    """

    soup = BeautifulSoup(CHMReader._decode_html(html), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Help pages use outer one-cell layout tables around all prose. Only
    # innermost tables with multiple rows/columns carry a grid to recover.
    tables = [
        table for table in soup.find_all("table")
        if table.find("table") is None
        and len(table.find_all("tr")) > 1
        and any(len(row.find_all(["td", "th"], recursive=False)) > 1 for row in table.find_all("tr"))
    ]
    text = _normalized(chunk)
    if not tables:
        return text
    # DOM positions disambiguate a table from identical text quoted earlier in
    # prose. Searching for the flattened table string would rewrite the quote.
    strings = [
        node for node in soup.descendants
        if type(node) in (NavigableString, CData) and _normalized(str(node))
    ]
    offsets: dict[int, tuple[int, int]] = {}
    parts: list[str] = []
    position = 0
    for node in strings:
        value = _normalized(str(node))
        offsets[id(node)] = (position, position + len(value))
        parts.append(value)
        position += len(value) + 1
    page = " ".join(parts)
    start = page.find(text)
    if start < 0 or page.find(text, start + 1) >= 0:
        # There is no unique provenance mapping for this legacy chunk.
        return ""
    end = start + len(text)
    replacements: list[tuple[int, int, str]] = []
    for table in tables:
        positions = [offsets[id(node)] for node in table.descendants if id(node) in offsets]
        if not positions:
            continue
        table_start, table_end = positions[0][0], positions[-1][1]
        if table_end <= start or table_start >= end:
            continue
        rendered = _complete_rows(table, offsets, start, end)
        replacements.append((max(table_start, start) - start, min(table_end, end) - start, rendered))
    for left, right, rendered in reversed(replacements):
        text = text[:left].rstrip() + "\n" + rendered + "\n" + text[right:].lstrip()
    return text.strip()


def _complete_rows(table, offsets: dict[int, tuple[int, int]], start: int, end: int) -> str:
    # Each slot stores (cell text, origin start, origin end, rows remaining).
    spans: dict[int, tuple[str, int, int, int]] = {}
    rows: list[list[tuple[str, int, int, int]]] = []
    row_spans: list[tuple[int, int]] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["td", "th"], recursive=False)
        positions = [offsets[id(node)] for node in row.descendants if id(node) in offsets]
        if not cells or not positions:
            spans = {column: (*cell[:3], cell[3] - 1) for column, cell in spans.items() if cell[3] > 1}
            continue
        row_start, row_end = positions[0][0], positions[-1][1]
        if row_start >= end:
            break
        grid = {column: cell for column, cell in spans.items()}
        spans = {column: (*cell[:3], cell[3] - 1) for column, cell in spans.items() if cell[3] > 1}
        column = 0
        for cell in cells:
            while column in grid:
                column += 1
            try:
                colspan, rowspan = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
            except (TypeError, ValueError):
                return ""
            if not (1 <= colspan <= 32 and 1 <= rowspan <= 1000 and column + colspan <= 32):
                return ""
            value = (_normalized(cell.get_text(" ", strip=True)), row_start, row_end, rowspan - 1)
            for index in range(column, column + colspan):
                if index in grid:
                    return ""
                grid[index] = value
                if rowspan > 1:
                    spans[index] = value
            column += colspan
        if set(grid) != set(range(len(grid))):
            return ""
        rows.append([grid[index] for index in range(len(grid))])
        row_spans.append((row_start, row_end))
    if not rows or row_spans[0][0] < start or row_spans[0][1] > end:
        # Without the table's heading cells, column meanings are ambiguous.
        return ""
    width = max(map(len, rows))
    complete = [
        row for row, (left, right) in zip(rows, row_spans, strict=True)
        if len(row) == width and start <= left and right <= end
        and all(start <= cell[1] and cell[2] <= end for cell in row)
    ]
    return "\n".join(
        " | ".join(f"Column {column}={cell[0] or '(empty)'}" for column, cell in enumerate(row, 1))
        for row in complete
    )
