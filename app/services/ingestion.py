"""Incremental, metadata-aware ingestion for the Support Copilot corpus.

The service deliberately keeps loading, metadata enrichment and indexing in one
place.  It does not send document fragments to a hosted embedding provider
until the caller explicitly confirms that transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import FlatReader, HTMLTagReader, MarkdownReader, PyMuPDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.services.chm_reader import CHMReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".html", ".htm", ".md", ".txt", ".chm"})

# These values are useful for citations and filtering, but adding paths,
# timestamps, author names or document versions to a chunk embedding harms
# semantic retrieval and makes otherwise identical pages look different.
EXCLUDED_EMBED_METADATA_KEYS = frozenset(
    {
        "source",
        "file_name",
        "file_path",
        "created_at",
        "last_modified",
        "author",
        "category",
        "product",
        "version",
        "page_label",
        "page",
        "total_pages",
        "chm_member",
    }
)
_VERSION_RE = re.compile(r"(?ix)(?:^|[._\-\s])v(?:ersion)?[._\-\s]*(\d+(?:[._\-]\d+){0,3})(?=$|[._\-\s])")
_PRODUCT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gts_nx", re.compile(r"(?i)(?:\bgts\s+nx\b|\bgtsnx\b)")),
    ("fea_nx", re.compile(r"(?i)(?:\bfea\s+nx\b|\bfeanx\b)")),
    ("civil", re.compile(r"(?i)\b(?:midas\s+)?civil\b")),
    ("gen", re.compile(r"(?i)\b(?:midas\s+)?gen\b")),
)
_GENERIC_MIDAS_PRODUCT_RE = re.compile(r"(?i)\bmidas\s+([a-z0-9]+(?:\s+nx)?)\b")
_PRODUCT_ALIASES = {
    "gts": "gts_nx",
    "gtsnx": "gts_nx",
    "gts_nx": "gts_nx",
    "fea": "fea_nx",
    "feanx": "fea_nx",
    "fea_nx": "fea_nx",
    "midas_civil": "civil",
    "midas_gen": "gen",
}


class ExternalDataTransferRequired(PermissionError):
    """Raised before a hosted provider receives a document fragment."""


class DocumentLoadError(RuntimeError):
    """Raised when a supported source file contains no readable text."""


class PythonDocxReader(BaseReader):
    """DOCX reader backed by the already required ``python-docx`` package.

    LlamaIndex's bundled ``DocxReader`` additionally depends on ``docx2txt``.
    Using the project dependency directly keeps DOCX ingestion available in the
    standard compose image and preserves table text as searchable content.
    """

    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        from docx import Document as DocxDocument

        docx = DocxDocument(str(file))
        paragraphs = [paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip()]
        table_rows = [
            " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            for table in docx.tables
            for row in table.rows
        ]
        text = "\n".join(paragraphs + [row for row in table_rows if row])
        return [Document(text=text, metadata=dict(extra_info or {}))]


@dataclass(frozen=True)
class IngestionConfig:
    """Settings required by :class:`IngestionService`.

    ``docstore_path`` is persisted alongside the local application state.  It
    is the state that lets a new CLI process distinguish changed source files
    from unchanged ones while Qdrant holds the corresponding chunks.
    """

    data_dir: Path = PROJECT_ROOT / "data"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "support_copilot"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chunk_size: int = 1024
    chunk_overlap: int = 64
    default_product: str = "gts_nx"
    docstore_path: Path = PROJECT_ROOT / "var" / "ingestion_docstore.json"

    def __post_init__(self) -> None:
        if not self.qdrant_url.strip():
            raise ValueError("qdrant_url must not be blank")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be blank")
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if not canonical_product(self.default_product):
            raise ValueError("default_product must not be blank")

    @classmethod
    def from_settings(cls, settings: Any) -> "IngestionConfig":
        """Build configuration without forcing unrelated app settings changes.

        It intentionally reuses the existing RAG names until a future settings
        migration introduces dedicated ingestion variables.
        """

        data_dir = Path(getattr(settings, "rag_data_dir", PROJECT_ROOT / "data"))
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        docstore_path = Path(
            getattr(
                settings,
                "rag_docstore_path",
                PROJECT_ROOT / "var" / "rag_eval_chunk_1024_docstore.json",
            )
        )
        if not docstore_path.is_absolute():
            docstore_path = PROJECT_ROOT / docstore_path
        secret = getattr(settings, "qdrant_api_key", None)
        qdrant_api_key = secret.get_secret_value() if secret else None
        openai_secret = getattr(settings, "openai_api_key", None)
        openai_api_key = openai_secret.get_secret_value() if openai_secret else None
        return cls(
            data_dir=data_dir,
            qdrant_url=getattr(settings, "qdrant_url", None) or "http://localhost:6333",
            qdrant_api_key=qdrant_api_key,
            collection_name=getattr(settings, "rag_collection", "rag_eval_chunk_1024"),
            embedding_model=getattr(settings, "rag_embedding_model", "text-embedding-3-large"),
            embedding_dimensions=getattr(settings, "rag_embedding_dimensions", 1024),
            openai_api_key=openai_api_key,
            openai_base_url=getattr(settings, "openai_base_url", None),
            chunk_size=getattr(settings, "rag_chunk_size", 1024),
            chunk_overlap=getattr(settings, "rag_chunk_overlap", 64),
            default_product=getattr(settings, "rag_product", "gts_nx"),
            docstore_path=docstore_path,
        )


@dataclass(frozen=True)
class IngestionResult:
    """A concise, machine-friendly report for one ingest invocation."""

    source_files: int
    loaded_documents: int
    changed_documents: int
    unchanged_documents: int
    indexed_nodes: int
    failed_files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed_count(self) -> int:
        return len(self.failed_files)

    @property
    def summary(self) -> str:
        return (
            f"{self.changed_documents} changed, {self.unchanged_documents} unchanged, "
            f"{self.failed_count} failed"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_files": self.source_files,
            "loaded_documents": self.loaded_documents,
            "changed_documents": self.changed_documents,
            "unchanged_documents": self.unchanged_documents,
            "indexed_nodes": self.indexed_nodes,
            "failed_files": list(self.failed_files),
            "summary": self.summary,
        }


def supported_files(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    """Return supported files in deterministic order, skipping prior failures."""

    if isinstance(path_or_paths, (str, Path)):
        candidates: Sequence[Path | str] = [path_or_paths]
    else:
        candidates = list(path_or_paths)
    result: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = [item for item in path.rglob("*") if item.is_file()]
        else:
            raise FileNotFoundError(f"Ingestion path does not exist: {path}")
        result.extend(
            item
            for item in files
            if item.suffix.lower() in SUPPORTED_SUFFIXES and not item.name.endswith(".failed")
        )
    # A file can be supplied directly as well as through its parent directory.
    return sorted({item.resolve() for item in result}, key=lambda item: str(item).casefold())


def category_for_path(path: Path, *, data_root: Path) -> str:
    """Map ``data/<category>/file`` to the category, with a safe fallback."""

    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return path.parent.name or "uncategorized"
    return relative.parts[0] if len(relative.parts) > 1 else "uncategorized"


def version_from_filename(file_name: str) -> str:
    """Extract explicit semantic versions such as ``manual-v2.1.pdf``."""

    match = _VERSION_RE.search(Path(file_name).stem)
    return match.group(1).replace("_", ".").replace("-", ".") if match else "unknown"


def canonical_product(value: str | None) -> str:
    """Return a stable product key suitable for metadata and filters."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return _PRODUCT_ALIASES.get(normalized, normalized)


def infer_product(values: Iterable[object], *, default_product: str = "gts_nx") -> str:
    """Infer a MIDAS product from metadata-like values.

    Explicit product markers always win.  Files without a marker inherit the
    configured corpus product, which keeps generic chapter names usable while
    still separating documents explicitly labelled for Civil, FEA NX, etc.
    """

    evidence = " ".join(str(value) for value in values if value is not None)
    searchable = re.sub(r"[_\\/\-.]+", " ", evidence)
    for product, pattern in _PRODUCT_RULES:
        if pattern.search(searchable):
            return product
    if match := _GENERIC_MIDAS_PRODUCT_RE.search(searchable):
        return canonical_product(match.group(1))
    return canonical_product(default_product) or "unknown"


def product_for_path(path: Path, *, default_product: str = "gts_nx") -> str:
    """Classify a source by product without depending on one exact filename."""

    return infer_product(path.parts, default_product=default_product)


def _optional_author(path: Path) -> str | None:
    try:
        if path.suffix.lower() == ".docx":
            from docx import Document as DocxDocument

            author = DocxDocument(str(path)).core_properties.author
            return str(author).strip() or None
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            metadata = PdfReader(str(path)).metadata
            author = metadata.author if metadata else None
            return str(author).strip() or None
    except Exception:
        # Source metadata must not make a readable document fail ingestion.
        return None
    return None


def enrich_metadata(
    document: Document,
    *,
    source_path: Path,
    data_root: Path,
    default_product: str = "gts_nx",
) -> Document:
    """Attach stable citation metadata while excluding it from embeddings."""

    stat = source_path.stat()
    metadata = dict(document.metadata or {})
    metadata.update(
        {
            "source": source_path.name,
            "file_name": source_path.name,
            "file_path": str(source_path.resolve()),
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "category": category_for_path(source_path, data_root=data_root),
            "product": product_for_path(source_path, default_product=default_product),
            "version": version_from_filename(source_path.name),
        }
    )
    author = _optional_author(source_path)
    if author:
        metadata["author"] = author
    document.metadata = metadata
    document.excluded_embed_metadata_keys = sorted(
        set(document.excluded_embed_metadata_keys) | EXCLUDED_EMBED_METADATA_KEYS
    )
    # Retrieval code reads citation fields directly from ``node.metadata``.
    # They therefore do not need to occupy context or reduce split capacity.
    document.excluded_llm_metadata_keys = sorted(
        set(document.excluded_llm_metadata_keys) | set(metadata)
    )
    return document


def _file_extractors() -> dict[str, Any]:
    """The explicit mapping is part of the assignment's format contract."""

    html_reader = HTMLTagReader(tag="body", ignore_no_id=False)
    return {
        ".pdf": PyMuPDFReader(),
        ".docx": PythonDocxReader(),
        ".html": html_reader,
        ".htm": html_reader,
        ".md": MarkdownReader(),
        ".txt": FlatReader(),
        ".chm": CHMReader(),
    }


def _preserve_pdf_page_metadata(documents: Sequence[Document], path: Path) -> None:
    """Attach 1-based pages when PyMuPDF returns one Document per PDF page."""

    if path.suffix.lower() != ".pdf":
        return
    total_pages = len(documents)
    for page, document in enumerate(documents, start=1):
        metadata = dict(document.metadata or {})
        metadata["page"] = page
        metadata.setdefault("total_pages", total_pages)
        document.metadata = metadata


def _load_one_file(
    path: Path,
    *,
    data_root: Path,
    default_product: str = "gts_nx",
) -> list[Document]:
    reader = SimpleDirectoryReader(
        input_files=[path],
        file_extractor=_file_extractors(),
        filename_as_id=True,
        raise_on_error=True,
    )
    documents = reader.load_data()
    _preserve_pdf_page_metadata(documents, path)
    nonempty = [
        document
        for document in documents
        if document.get_content(metadata_mode=MetadataMode.NONE).strip()
    ]
    if not nonempty:
        raise DocumentLoadError(f"No readable text found in {path.name}")
    return [
        enrich_metadata(
            document,
            source_path=path,
            data_root=data_root,
            default_product=default_product,
        )
        for document in nonempty
    ]


def _failed_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.failed")
    attempt = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.failed.{attempt}")
        attempt += 1
    return candidate


def _is_local_qdrant_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    return host in {"", "localhost", "127.0.0.1", "::1", "qdrant"}


class IngestionService:
    """Load and incrementally upsert corpus files into Qdrant through LlamaIndex."""

    def __init__(
        self,
        config: IngestionConfig,
        *,
        client: QdrantClient | None = None,
        embed_model: Any | None = None,
        docstore: SimpleDocumentStore | None = None,
        requires_external_data_transfer: bool | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._embed_model = embed_model
        self._docstore = docstore or self._load_docstore(config.docstore_path)
        self._requires_external_data_transfer = (
            embed_model is None
            if requires_external_data_transfer is None
            else requires_external_data_transfer
        )
        self._pipeline: IngestionPipeline | None = None

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> "IngestionService":
        return cls(IngestionConfig.from_settings(settings), **kwargs)

    @property
    def docstore(self) -> SimpleDocumentStore:
        return self._docstore

    def ingest(
        self,
        path_or_paths: Path | str | Iterable[Path | str] | None = None,
        *,
        allow_external_data_transfer: bool | None = None,
    ) -> IngestionResult:
        """Load files and UPSERT changed documents only.

        A hosted embedding model is an external data transfer.  The explicit
        flag makes that decision visible at every CLI and upload call site.
        """

        allowed = bool(allow_external_data_transfer)
        if self._requires_external_data_transfer and not allowed:
            raise ExternalDataTransferRequired(
                "Refusing to send document fragments to the embedding provider. "
                "Pass allow_external_data_transfer=True after explicit approval."
            )
        if not self._requires_external_data_transfer and not _is_local_qdrant_url(self.config.qdrant_url) and not allowed:
            raise ExternalDataTransferRequired(
                "Qdrant URL is not local. Pass allow_external_data_transfer=True after explicit approval."
            )

        source_files = supported_files(self.config.data_dir if path_or_paths is None else path_or_paths)
        documents: list[Document] = []
        failures: list[str] = []
        for path in source_files:
            try:
                documents.extend(
                    _load_one_file(
                        path,
                        data_root=self.config.data_dir,
                        default_product=self.config.default_product,
                    )
                )
            except Exception:
                logger.exception("Could not load corpus file %s", path)
                moved = _failed_path(path)
                try:
                    path.rename(moved)
                except OSError:
                    logger.exception("Could not rename failed corpus file %s", path)
                    failures.append(str(path))
                else:
                    failures.append(str(moved))

        changed, unchanged = self._change_counts(documents)
        if not documents:
            return IngestionResult(
                source_files=len(source_files),
                loaded_documents=0,
                changed_documents=0,
                unchanged_documents=0,
                indexed_nodes=0,
                failed_files=tuple(failures),
            )

        self._ensure_collection()
        nodes = self._pipeline_or_create().run(documents=documents, show_progress=False)
        self._persist_docstore()
        return IngestionResult(
            source_files=len(source_files),
            loaded_documents=len(documents),
            changed_documents=changed,
            unchanged_documents=unchanged,
            indexed_nodes=len(nodes),
            failed_files=tuple(failures),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _embedding_model(self) -> Any:
        if self._embed_model is None:
            self._embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                api_key=self.config.openai_api_key,
                api_base=self.config.openai_base_url,
            )
        return self._embed_model

    def _client_or_create(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)
        return self._client

    def _ensure_collection(self) -> None:
        client = self._client_or_create()
        if not client.collection_exists(self.config.collection_name):
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=self.config.embedding_dimensions, distance=Distance.COSINE),
            )
            return
        vectors = client.get_collection(self.config.collection_name).config.params.vectors
        if isinstance(vectors, dict) or int(vectors.size) != self.config.embedding_dimensions:
            raise ValueError(
                f"Collection {self.config.collection_name!r} must use one "
                f"{self.config.embedding_dimensions}-dimension vector"
            )
        if vectors.distance != Distance.COSINE:
            raise ValueError(f"Collection {self.config.collection_name!r} must use cosine distance")

    def _pipeline_or_create(self) -> IngestionPipeline:
        if self._pipeline is None:
            vector_store = QdrantVectorStore(
                client=self._client_or_create(),
                collection_name=self.config.collection_name,
                flat_metadata=True,
            )
            self._pipeline = IngestionPipeline(
                transformations=[
                    # Citation metadata stays on the node, but must not consume
                    # the semantic chunk budget (or become embedding input).
                    SentenceSplitter(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                        # Keep citation fields on produced nodes.  They remain
                        # excluded from embedding/LLM text by the per-document
                        # exclusion lists set in ``enrich_metadata``.
                        include_metadata=True,
                    ),
                    self._embedding_model(),
                ],
                vector_store=vector_store,
                docstore=self._docstore,
                docstore_strategy=DocstoreStrategy.UPSERTS,
            )
        return self._pipeline

    def _change_counts(self, documents: Sequence[Document]) -> tuple[int, int]:
        changed = sum(
            self._docstore.get_document_hash(document.id_) != document.hash for document in documents
        )
        return changed, len(documents) - changed

    @staticmethod
    def _load_docstore(path: Path) -> SimpleDocumentStore:
        if path.exists():
            return SimpleDocumentStore.from_persist_path(str(path))
        return SimpleDocumentStore()

    def _persist_docstore(self) -> None:
        self.config.docstore_path.parent.mkdir(parents=True, exist_ok=True)
        self._docstore.persist(str(self.config.docstore_path))


# The alternate name reads naturally in FastAPI composition roots and keeps the
# public contract stable for the homework description.
IngestionPipelineService = IngestionService
