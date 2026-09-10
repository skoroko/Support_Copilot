"""Opt-in corpus downloader.

The project intentionally ships no hard-coded third-party URLs.  The owner
provides a reviewed JSON manifest, which keeps licences and document ownership
visible instead of silently downloading an arbitrary corpus.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a reviewed Support Copilot corpus manifest")
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        help="JSON list: [{\"url\": ..., \"category\": ..., \"file_name\": ...}]",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Confirm network download of the reviewed manifest URLs",
    )
    parser.add_argument(
        "--from-chm",
        type=Path,
        help="Extract safe, offline HTML pages from an existing CHM file instead of downloading",
    )
    parser.add_argument("--limit", type=int, default=60, help="Maximum number of CHM pages to materialize")
    return parser.parse_args()


def _safe_file_name(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."}:
        raise ValueError("file_name must be a simple file name")
    return name


def download(manifest: Path, output_dir: Path, *, allow_download: bool) -> list[Path]:
    if not allow_download:
        raise PermissionError("Refusing network download. Re-run with --allow-download after reviewing the manifest.")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a JSON list")
    downloaded: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest item must be an object")
        url = str(entry.get("url", ""))
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError(f"Only http(s) URLs are allowed: {url!r}")
        category = _safe_file_name(str(entry.get("category", "uncategorized")))
        file_name = _safe_file_name(str(entry.get("file_name", Path(urlparse(url).path).name)))
        target = output_dir / category / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:  # nosec B310 - explicitly opt-in reviewed URLs
            target.write_bytes(response.read())
        downloaded.append(target)
    return downloaded


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").casefold()
    return normalized[:80] or "topic"


def extract_chm_to_html(
    source_chm: Path,
    output_dir: Path,
    *,
    limit: int = 60,
    reader: Any | None = None,
) -> list[Path]:
    """Materialize readable CHM topics as harmless, deterministic HTML files.

    This is deliberately an offline conversion of an already owned manual, not
    a downloader and not generated knowledge.  Text is HTML-escaped, so an
    original topic cannot introduce scripts or active markup into the corpus.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    if not source_chm.is_file():
        raise FileNotFoundError(f"CHM file does not exist: {source_chm}")
    if reader is None:
        from app.services.chm_reader import CHMReader

        reader = CHMReader()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for number, document in enumerate(reader.load_data(source_chm), start=1):
        if len(files) >= limit:
            break
        text = document.text.strip()
        if not text:
            continue
        member = str((document.metadata or {}).get("chm_member") or f"topic-{number}")
        target = output_dir / f"{number:03d}-{_slug(member)}.html"
        safe_title = html.escape(member, quote=True)
        safe_text = html.escape(text, quote=False)
        target.write_text(
            "<!doctype html>\n"
            "<html lang=\"ru\"><head><meta charset=\"utf-8\">"
            f"<title>{safe_title}</title></head><body>"
            f"<h1>{safe_title}</h1><pre>{safe_text}</pre></body></html>\n",
            encoding="utf-8",
        )
        files.append(target)
    if not files:
        raise ValueError(f"No readable CHM topics found in {source_chm}")
    return files


if __name__ == "__main__":
    args = parse_args()
    if args.from_chm:
        if args.manifest:
            raise SystemExit("Pass either a manifest or --from-chm, not both")
        files = extract_chm_to_html(args.from_chm, args.output_dir, limit=args.limit)
        print(f"Extracted {len(files)} safe HTML files locally")
    elif args.manifest:
        files = download(args.manifest, args.output_dir, allow_download=args.allow_download)
        print(f"Downloaded {len(files)} files")
    else:
        raise SystemExit("Pass a reviewed manifest or --from-chm")
