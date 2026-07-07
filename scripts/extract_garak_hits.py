from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract successful attack examples from a garak JSONL report.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hits = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if _is_hit(entry):
            hits.append(entry)

    for entry in hits[: args.limit]:
        print(json.dumps(_summarize(entry), ensure_ascii=False))
    return 0


def _is_hit(entry: dict[str, Any]) -> bool:
    if entry.get("entry_type") != "attempt" or entry.get("status") != 2:
        return False
    detector_results = entry.get("detector_results") or {}
    for values in detector_results.values():
        if isinstance(values, list) and values and float(values[0]) >= 0.5:
            return True
    return False


def _summarize(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe": entry.get("probe_classname") or entry.get("probe"),
        "prompt": entry.get("prompt") or entry.get("messages"),
        "outputs": entry.get("outputs"),
        "detector_results": entry.get("detector_results"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
