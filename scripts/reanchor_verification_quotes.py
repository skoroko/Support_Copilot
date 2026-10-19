"""Re-anchor verification-set exact_quotes onto the current corpus serialization.

After a corpus rebuild or extraction-format change, ``exact_quote`` slots may
stop matching verbatim even though the same evidence is retrievable. This tool
maps each stale quote to the corresponding passage in the persisted docstore
and proposes a verbatim window that (a) covers the same evidence and
(b) sits inside one indexed chunk, so the slot metric keeps measuring recall
rather than serialization drift.

Usage (from ``HelpDesk_Assistant/``)::

    .venv/Scripts/python scripts/reanchor_verification_quotes.py --dry-run
    .venv/Scripts/python scripts/reanchor_verification_quotes.py --write
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SET = ROOT / "tests" / "eval" / "verification_set.json"
DEFAULT_DOCSTORE = ROOT / "var" / "rag_eval_chunk_1024_docstore.json"

MIN_QUOTE = 200
MAX_QUOTE = 700


def norm(text: object) -> str:
    return " ".join(str(text or "").split())


def load_docs(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    for key, value in payload.get("docstore/data", {}).items():
        text = (value.get("__data__", {}) or {}).get("text", "") if isinstance(value, dict) else ""
        if text:
            docs[key.replace("\\", "/")] = text
    return docs


def doc_key_for(docs: dict[str, str], relpath: str, page: int | None) -> str | None:
    tail = relpath.replace("\\", "/").rstrip("/")
    name = tail.split("/")[-1]
    part = None if page is None else page - 1
    if part is not None:
        for key in docs:
            if name in key and key.endswith(f"_part_{part}"):
                return key
        return None
    return None


def search_docs(docs: dict[str, str], relpath: str, probes: list[str]) -> str | None:
    name = relpath.replace("\\", "/").rstrip("/").split("/")[-1]
    for probe in probes:
        if not probe:
            continue
        hits = [key for key, text in docs.items() if name in key and probe in norm(text)]
        if hits:
            return hits[0]
    return None


def quote_probes(old_norm: str) -> list[str]:
    """Distinctive mid-quote phrases, longest first."""
    size = len(old_norm)
    probes = []
    for length in (60, 45, 30):
        start = max(0, size // 2 - length // 2)
        probes.append(old_norm[start:start + length])
    return probes


def find_window(old_norm: str, new_norm: str) -> tuple[int, int] | None:
    """Locate the span in ``new_norm`` aligned with ``old_norm``.

    Uses difflib matching blocks: the new serialization re-orders/re-formats
    lines, so the window spans from the first to the last aligned region and is
    then trimmed to verbatim bounds.
    """
    matcher = difflib.SequenceMatcher(None, old_norm, new_norm, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= 12]
    if not blocks:
        return None
    start = blocks[0].b
    end = blocks[-1].b + blocks[-1].size
    if end - start < MIN_QUOTE:
        center = (start + end) // 2
        start = max(0, center - MIN_QUOTE // 2)
        end = min(len(new_norm), start + MIN_QUOTE)
    if end - start > MAX_QUOTE:
        # Keep the densest ~MAX_QUOTE region around the largest matching block.
        biggest = max(blocks, key=lambda b: b.size)
        center = biggest.b + biggest.size // 2
        start = max(0, center - MAX_QUOTE // 2)
        end = min(len(new_norm), start + MAX_QUOTE)
    # Trim to whitespace boundaries so the quote has clean edges.
    while start > 0 and not new_norm[start - 1].isspace() and start < end:
        start -= 1
    while end < len(new_norm) and not new_norm[end].isspace():
        end += 1
    return start, end


def propose(item: dict, slot_index: int, docs: dict[str, str]) -> dict:
    slot = item["required_evidence_slots"][slot_index]
    rel = slot.get("relative_path", "")
    page = slot.get("physical_page")
    old_q = norm(slot["exact_quote"])
    key = doc_key_for(docs, rel, page)
    if key is None:
        key = search_docs(docs, rel, quote_probes(old_q))
    if key is None:
        return {"status": "no_document", "slot": slot_index}
    new_text = norm(docs[key])
    window = find_window(old_q, new_text)
    if window is None:
        return {"status": "no_alignment", "slot": slot_index, "doc": key}
    start, end = window
    return {
        "status": "ok",
        "slot": slot_index,
        "doc": key,
        "old_len": len(old_q),
        "new_quote": docs[key] and new_text[start:end],
        "new_len": end - start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--docstore", type=Path, default=DEFAULT_DOCSTORE)
    parser.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    parser.add_argument("--write", action="store_true", help="update the set file in place")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.set.read_text(encoding="utf-8"))
    items = payload.get("items", payload.get("cases", []))
    docs = load_docs(args.docstore)
    wanted = set(filter(None, args.ids.split(","))) or None

    proposals: list[dict] = []
    for item in items:
        if wanted and item["id"] not in wanted:
            continue
        for index, slot in enumerate(item.get("required_evidence_slots") or []):
            result = propose(item, index, docs)
            result["question_id"] = item["id"]
            proposals.append(result)

    changed = 0
    for result in proposals:
        if result["status"] != "ok":
            print(json.dumps(result, ensure_ascii=False))
            continue
        item = next(i for i in items if i["id"] == result["question_id"])
        slot = item["required_evidence_slots"][result["slot"]]
        # preserve the quote's original serialization (non-normalized text is
        # what _slot_hit normalizes anyway); store the verbatim window.
        slot["exact_quote"] = result["new_quote"]
        changed += 1
        print(json.dumps({k: v for k, v in result.items() if k != "new_quote"},
                         ensure_ascii=False))

    if args.write and changed:
        args.set.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"proposed": changed, "written": bool(args.write and changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
