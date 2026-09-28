from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.documents import IngestionQueueBudget, router


ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token-000000000000000002"}

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
        headers=ADMIN_HEADERS,
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
        headers=ADMIN_HEADERS,
        files={"file": ("payload.exe", b"x", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert service.calls == []


def test_upload_rejects_multiple_files(tmp_path: Path) -> None:
    api, service = build_app(tmp_path)
    response = TestClient(api).post(
        "/documents/upload",
        headers=ADMIN_HEADERS,
        files=[
            ("file", ("one.pdf", b"%PDF", "application/pdf")),
            ("file", ("two.pdf", b"%PDF", "application/pdf")),
        ],
    )
    assert response.status_code == 400
    assert service.calls == []


def test_upload_requires_document_transfer_approval(tmp_path: Path) -> None:
    api, service = build_app(tmp_path, allow_external=False)

    response = TestClient(api).post(
        "/documents/upload",
        headers=ADMIN_HEADERS,
        files={"file": ("policy.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 409
    assert service.calls == []
    assert not (tmp_path / "uploads" / "policy.pdf").exists()


async def _exercise_queue_budget() -> None:
    budget = IngestionQueueBudget(1)
    assert await budget.try_acquire() is True
    assert await budget.try_acquire() is False
    await budget.release()
    assert await budget.try_acquire() is True


def test_ingestion_queue_budget_rejects_excess_pending_work() -> None:
    import asyncio

    asyncio.run(_exercise_queue_budget())
