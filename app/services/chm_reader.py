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
            members = archive.files(include_system=False)
            documents: list[Document] = []
            for member in members:
                member_name = str(member["name"])
                if Path(member_name).suffix.lower() not in _HTML_SUFFIXES:
                    continue
                try:
                    raw = archive.read(member_name, max_size=self._max_member_size)
                except (ChmError, FileNotFoundError, OSError):
                    # Some legacy CHM archives contain member names that the
                    # underlying Windows extraction API cannot materialize.
                    # A broken page must not make the remaining help unavailable.
                    continue
                text = self._html_to_text(raw)
                if not text:
                    continue
                metadata = dict(extra_info or {})
                metadata.update(
                    {
                        "file_name": source_path.name,
                        "file_path": str(source_path),
                        "chm_member": member_name,
                        "source": f"{source_path.name}::{member_name.lstrip('/')}",
                    }
                )
                documents.append(Document(text=text, metadata=metadata))

        if not documents:
            raise ValueError(f"CHM archive contains no readable HTML pages: {source_path}")
        return documents

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
    def _html_to_text(raw: bytes) -> str:
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

        soup = BeautifulSoup(decoded, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())
