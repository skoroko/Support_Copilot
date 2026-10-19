"""Offline source/extraction/index reconciliation. Read-only unless --write-catalog.

No service settings, environment files, embedding providers or live databases are read.
The console report contains counts and status only, never manual text or credentials.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.corpus import CorpusCatalog, audit_catalog


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--catalog", type=Path, help="Audit saved extraction catalog instead of extracting anew")
    parser.add_argument("--audit", type=Path, metavar="DOCSTORE", help="Read a persisted ingestion docstore")
    parser.add_argument("--qdrant-snapshot", type=Path, help="Read serialized point metadata; never connects to Qdrant")
    parser.add_argument("--observations", type=Path, help="Hash-bound human visual review observations JSON list")
    parser.add_argument("--transcriptions", type=Path, help="Approved exact-image visual transcripts JSON list")
    parser.add_argument("--write-catalog", type=Path, help="Explicitly save private evidence catalog (contains manual text)")
    parser.add_argument("--summary", type=Path, help="Explicitly save the text-free summary JSON")
    return parser.parse_args(argv)


def _records(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if path.name.casefold().startswith(".env"):
        raise ValueError("Environment files are not corpus audit inputs")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, list):
        raise TypeError("Review inputs must be JSON lists")
    return result


def run(args: argparse.Namespace) -> dict:
    for path in (args.catalog, args.audit, args.qdrant_snapshot):
        if path and path.name.casefold().startswith(".env"):
            raise ValueError("Environment files are not corpus audit inputs")
    if args.catalog:
        if args.transcriptions:
            raise ValueError("Transcriptions require fresh source extraction")
        catalog = CorpusCatalog.load(args.catalog)
        catalog.apply_observations(_records(args.observations))
    else:
        catalog = CorpusCatalog.build(args.data_dir, observations=_records(args.observations),
                                      transcriptions=_records(args.transcriptions))
    report = audit_catalog(catalog, data_root=args.data_dir, docstore_path=args.audit,
                           qdrant_snapshot=args.qdrant_snapshot)
    if args.write_catalog:
        catalog.persist(args.write_catalog)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        # Third-party parsers may print document fragments in diagnostics.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = run(args)
    except Exception as exc:  # noqa: BLE001 - never expose parser text in the public report
        # Parser/library errors may contain manual fragments; expose only their type.
        print(json.dumps({"mode": "read_only", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
