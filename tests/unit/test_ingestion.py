from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as DocxDocument
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import Document
from qdrant_client import QdrantClient

from app.services.ingestion import (
    EXCLUDED_EMBED_METADATA_KEYS,
    ExternalDataTransferRequired,
    IngestionConfig,
    IngestionService,
    _file_extractors,
    _load_one_file,
    category_for_path,
    infer_product,
    product_for_path,
    supported_files,
    version_from_filename,
)
from scripts.download_data import extract_chm_to_html


def make_service(tmp_path: Path) -> IngestionService:
    return IngestionService(
        IngestionConfig(
            data_dir=tmp_path / "data",
            qdrant_url="http://qdrant:6333",
            collection_name="ingestion_test",
            embedding_dimensions=8,
            chunk_size=32,
            chunk_overlap=4,
            docstore_path=tmp_path / "docstore.json",
        ),
        client=QdrantClient(":memory:"),
        embed_model=MockEmbedding(embed_dim=8),
    )


def test_loaders_and_metadata_cover_markdown_html_docx_and_txt(tmp_path: Path) -> None:
    data = tmp_path / "data"
    category = data / "licensing"
    category.mkdir(parents=True)
    (category / "guide-v2.1.md").write_text("# Лицензия\nКлюч активируется в личном кабинете.", encoding="utf-8")
    (category / "faq.html").write_text("<html><body><p>Срок ответа — один день.</p></body></html>", encoding="utf-8")
    (category / "notes.txt").write_text("Текстовый файл базы знаний.", encoding="utf-8")
    docx_path = category / "install.docx"
    docx = DocxDocument()
    docx.core_properties.author = "MIDAS Support"
    docx.add_paragraph("Установка завершена успешно.")
    docx.save(docx_path)

    service = make_service(tmp_path)
    result = service.ingest(data)

    assert result.source_files == 4
    assert result.failed_files == ()
    assert result.changed_documents >= 4
    assert result.indexed_nodes >= 4
    stored_documents = service.docstore.docs
    assert stored_documents
    metadata = next(node.metadata for node in stored_documents.values() if node.metadata["file_name"] == "install.docx")
    assert metadata["source"] == "install.docx"
    assert metadata["category"] == "licensing"
    assert metadata["author"] == "MIDAS Support"
    assert set(EXCLUDED_EMBED_METADATA_KEYS).issubset(
        next(node.excluded_embed_metadata_keys for node in stored_documents.values())
    )
    points, _ = service._client.scroll(
        "ingestion_test",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    assert points
    assert all((point.payload or {}).get("file_name") for point in points)


def test_second_ingest_is_idempotent_and_persists_docstore(tmp_path: Path) -> None:
    data = tmp_path / "data" / "faq"
    data.mkdir(parents=True)
    (data / "return-v1.md").write_text("Возврат возможен в течение 14 дней.", encoding="utf-8")

    first = make_service(tmp_path)
    initial = first.ingest(tmp_path / "data")
    # Recreate the service to exercise the persisted SimpleDocumentStore.
    reopened = IngestionService(
        first.config,
        client=first._client,
        embed_model=MockEmbedding(embed_dim=8),
    )
    repeated = reopened.ingest(tmp_path / "data")

    assert initial.changed_documents == 1
    assert repeated.changed_documents == 0
    assert repeated.unchanged_documents == 1
    assert repeated.indexed_nodes == 0
    assert repeated.summary.startswith("0 changed, 1 unchanged")


def test_failure_is_renamed_and_does_not_abort_other_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    good = data / "good.md"
    bad = data / "bad.md"
    good.write_text("Корректный текст.", encoding="utf-8")
    bad.write_text("Не будет прочитан.", encoding="utf-8")
    original = __import__("app.services.ingestion", fromlist=["_load_one_file"])._load_one_file

    def fail_bad(path: Path, **kwargs: object):
        if path.name == "bad.md":
            raise RuntimeError("fixture failure")
        return original(path, **kwargs)

    monkeypatch.setattr("app.services.ingestion._load_one_file", fail_bad)
    result = make_service(tmp_path).ingest(data)

    assert result.changed_documents == 1
    assert len(result.failed_files) == 1
    assert not bad.exists()
    assert (data / "bad.md.failed").exists()


def test_external_embedding_requires_explicit_confirmation(tmp_path: Path) -> None:
    config = IngestionConfig(data_dir=tmp_path, docstore_path=tmp_path / "docstore.json")

    with pytest.raises(ExternalDataTransferRequired):
        IngestionService(config).ingest()


def test_path_helpers_are_stable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    path = data / "manuals" / "guide-v3.2.pdf"
    path.parent.mkdir(parents=True)
    path.write_text("placeholder", encoding="utf-8")

    assert category_for_path(path, data_root=data) == "manuals"
    assert version_from_filename(path.name) == "3.2"
    assert supported_files(data) == [path.resolve()]


def test_product_metadata_uses_explicit_product_markers_and_corpus_default(tmp_path: Path) -> None:
    assert product_for_path(tmp_path / "03_midas GTS NX_elements.pdf") == "gts_nx"
    assert product_for_path(tmp_path / "manuals" / "midas Civil guide.pdf") == "civil"
    assert product_for_path(tmp_path / "manuals" / "midas NFX guide.pdf") == "nfx"
    assert infer_product(["generic User Manual Chapter 5.pdf"], default_product="gts_nx") == "gts_nx"


def test_pdf_page_numbers_are_preserved_before_metadata_enrichment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "GTS_NX_manual.pdf"
    pdf.write_bytes(b"fixture")

    class FakeDirectoryReader:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load_data(self) -> list[Document]:
            return [
                Document(text="Первая страница"),
                Document(text="Вторая страница"),
                Document(text="Третья страница"),
            ]

    monkeypatch.setattr("app.services.ingestion.SimpleDirectoryReader", FakeDirectoryReader)

    documents = _load_one_file(pdf, data_root=tmp_path)

    assert [document.metadata["page"] for document in documents] == [1, 2, 3]
    assert all(document.metadata["total_pages"] == 3 for document in documents)
    assert all(document.metadata["product"] == "gts_nx" for document in documents)


def test_pdf_ingestion_uses_pypdf_reader() -> None:
    assert type(_file_extractors()[".pdf"]).__name__ == "PDFReader"


def test_chm_pages_can_be_materialized_as_safe_deterministic_html(tmp_path: Path) -> None:
    source = tmp_path / "manual.chm"
    source.write_bytes(b"fixture")

    class Reader:
        def load_data(self, _path: Path):
            return [
                type("Document", (), {"text": "<script>ignore()</script>Свободы", "metadata": {"chm_member": "/topic.htm"}})(),
            ]

    output = tmp_path / "data" / "gts_nx_help_html"
    files = extract_chm_to_html(source, output, reader=Reader())

    assert [item.name for item in files] == ["001-topic-htm.html"]
    rendered = files[0].read_text(encoding="utf-8")
    assert "&lt;script&gt;ignore()&lt;/script&gt;" in rendered
    assert "<script>ignore()" not in rendered
