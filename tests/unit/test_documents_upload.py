from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.documents import router


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Path], bool]] = []

    def ingest(self, paths, *, allow_external_data_transfer: bool):
        normalized = [Path(path) for path in paths]
        self.calls.append((normalized, allow_external_data_transfer))
        return SimpleNamespace(summary="1 changed, 0 unchanged, 0 failed")


def build_app(tmp_path: Path, *, allow_external: bool = True) -> tuple[FastAPI, FakeIngestionService]:
    api = FastAPI()
    api.include_router(router)
    service = FakeIngestionService()
    api.state.ingestion_service = service
    api.state.settings = SimpleNamespace(
        rag_data_dir=tmp_path,
        rag_allow_external_data_transfer=allow_external,
    )
    return api, service


def test_upload_persists_document_and_schedules_incremental_ingestion(tmp_path: Path) -> None:
    api, service = build_app(tmp_path)

    response = TestClient(api).post(
        "/documents/upload",
        data={"category": "manuals"},
        files={"file": ("returns-v2.md", b"# Return policy", "text/markdown")},
    )

    assert response.status_code == 202
    assert response.json()["indexing"] == "scheduled"
    target = tmp_path / "manuals" / "returns-v2.md"
    assert target.read_bytes() == b"# Return policy"
    assert service.calls == [([target], True)]


def test_upload_rejects_unsupported_format(tmp_path: Path) -> None:
    api, service = build_app(tmp_path)

    response = TestClient(api).post(
        "/documents/upload",
        files={"file": ("payload.exe", b"x", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert service.calls == []


def test_upload_requires_document_transfer_approval(tmp_path: Path) -> None:
    api, service = build_app(tmp_path, allow_external=False)

    response = TestClient(api).post(
        "/documents/upload",
        files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 409
    assert service.calls == []
    assert not (tmp_path / "uploads" / "policy.pdf").exists()
