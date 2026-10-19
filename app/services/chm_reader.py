from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from pylibmspack import ChmArchive
from pylibmspack.errors import ChmError

from app.services.corpus import EXTRACTION_VERSION, stable_id
from app.services.corpus_html import parse_html

_HTML_SUFFIXES = {".htm", ".html", ".xhtml"}
_CHARSET_RE = re.compile(br"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.IGNORECASE)


class CHMReader(BaseReader):
    """Read Microsoft Compiled HTML Help pages as LlamaIndex documents."""

    def __init__(
        self,
        *,
        archive_factory: Callable[[str], Any] = ChmArchive,
        max_member_size: int = 16 * 1024 * 1024,
    ) -> None:
        self._archive_factory = archive_factory
        self._max_member_size = max_member_size

    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        source_path = Path(file)
        if source_path.suffix.lower() != ".chm":
            raise ValueError(f"CHMReader supports only .chm files: {source_path}")

        with tempfile.TemporaryDirectory(prefix="support-copilot-chm-") as temp_dir:
            archive_path = self._ascii_compatible_path(source_path, Path(temp_dir))
            archive = self._archive_factory(str(archive_path))
            members = sorted(archive.files(include_system=False), key=lambda item: str(item["name"]))
            documents: list[Document] = []
            for member in members:
                member_name = str(member["name"])
                if Path(member_name).suffix.lower() not in _HTML_SUFFIXES:
                    continue
                metadata = dict(extra_info or {})
                location = stable_id("location", metadata.get("source_id", str(source_path.resolve())), member_name)
                metadata.update(
                    {
                        "file_name": source_path.name,
                        "file_path": str(source_path),
                        "chm_member": member_name,
                        "source": f"{source_path.name}::{member_name.lstrip('/')}",
                        "location_id": location,
                        "extraction_version": EXTRACTION_VERSION,
                    }
                )
                try:
                    try:
                        raw = archive.read(member_name, max_size=self._max_member_size)
                        metadata["chm_read_method"] = "pylibmspack"
                    except FileNotFoundError:
                        # On Windows pylibmspack's native writer can produce a
                        # mojibake output filename for a Unicode member name.
                        # Retry the SAME member into a fixed ASCII destination;
                        # never select a similarly named topic as a substitute.
                        if not isinstance(archive, ChmArchive) or member_name.isascii():
                            raise
                        raw = self._read_exact_member_ascii_output(archive, member_name, member.get("size"))
                        metadata["chm_read_method"] = "pylibmspack_ascii_output"
                    extracted = parse_html(raw, location)
                    text = extracted["text"]
                    metadata.update({"extraction_status": "text" if text.strip() else "no_text",
                                     "evidence_blocks": extracted["blocks"], "relations": extracted["relations"],
                                     "title": extracted["title"], "language": extracted["language"] if extracted["language"] != "unknown"
                                     else metadata.get("language", "unknown"),
                                     "section_path": extracted["section_path"]})
                except (ChmError, FileNotFoundError, OSError) as exc:
                    # Retain the exact member and failure in the source ledger.
                    text = ""
                    metadata.update({"extraction_status": "unreadable", "extraction_error": type(exc).__name__,
                                     "evidence_blocks": [], "relations": [], "language": "unknown"})
                documents.append(Document(text=text, metadata=metadata))

        if not documents:
            raise ValueError(f"CHM archive contains no HTML pages: {source_path}")
        return documents

    def _read_exact_member_ascii_output(self, archive: ChmArchive, member_name: str, expected_size: int | None) -> bytes:
        """Isolate pylibmspack's native output-path limitation, retaining lookup.

        The library currently exposes no public destination override on read().
        Its bundled native extractor accepts an exact member and output path;
        keep this small compatibility boundary explicit and size bounded.
        """
        from pylibmspack import _cab

        with tempfile.TemporaryDirectory(prefix="support-copilot-member-") as temp_dir:
            target = Path(temp_dir) / "member.bin"
            if not str(target).isascii():
                raise OSError("CHM compatibility extraction requires an ASCII temporary directory")
            if archive._data is not None:
                error = _cab.chm_extract_file_bytes(archive._data, member_name, str(target), self._max_member_size)
            else:
                error = _cab.chm_extract_file(archive.path, member_name, str(target), self._max_member_size)
            if error != getattr(_cab, "MSPACK_ERR_OK", 0):
                raise ChmError(f"exact member extraction failed: libmspack error {error}")
            size = target.stat().st_size
            if size > self._max_member_size or (expected_size is not None and size != expected_size):
                raise ChmError("exact member extraction size mismatch or limit exceeded")
            return target.read_bytes()

    @staticmethod
    def _ascii_compatible_path(source: Path, temp_dir: Path) -> Path:
        try:
            str(source).encode("ascii")
        except UnicodeEncodeError:
            copied = temp_dir / "manual.chm"
            shutil.copy2(source, copied)
            return copied
        return source

    @staticmethod
    def _decode_html(raw: bytes) -> str:
        match = _CHARSET_RE.search(raw[:4096])
        candidates = [match.group(1).decode("ascii", errors="ignore")] if match else []
        candidates.extend(["utf-8", "windows-1251", "latin-1"])
        decoded = ""
        for encoding in candidates:
            try:
                decoded = raw.decode(encoding)
                break
            except (LookupError, UnicodeDecodeError):
                continue
        if not decoded:
            decoded = raw.decode("utf-8", errors="replace")
        return decoded

    @staticmethod
    def _html_to_text(raw: bytes) -> str:
        soup = BeautifulSoup(CHMReader._decode_html(raw), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())
