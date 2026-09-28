from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile

from app.admin.auth import require_admin


router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_CATEGORY_RE = re.compile(r"^[A-Za-z0-9А-Яа-яЁё][A-Za-z0-9А-Яа-яЁё._-]{0,63}$")


class IngestionQueueBudget:
    """Bound pending background ingestion work per application process."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._pending = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._pending >= self.capacity:
                return False
            self._pending += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._pending = max(0, self._pending - 1)


def _ingestion_budget(request: Request, capacity: int) -> IngestionQueueBudget:
    budget = getattr(request.app.state, "ingestion_queue_budget", None)
    if budget is None:
        budget = IngestionQueueBudget(capacity)
        request.app.state.ingestion_queue_budget = budget
    return budget


def _safe_name(raw_name: str | None) -> str:
    name = Path(raw_name or "").name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_filename", "message": "A non-empty filename is required"},
        )
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_document_format",
                "message": "Supported formats: PDF, DOCX, HTML and Markdown",
            },
        )
    return name


def _safe_category(raw_category: str) -> str:
    category = raw_category.strip()
    if not _CATEGORY_RE.fullmatch(category):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_category", "message": "Category must be a simple directory name"},
        )
    return category


async def _persist_upload(upload: UploadFile, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.uploading")
    written = 0
    try:
        async with aiofiles.open(temporary, "wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "document_too_large",
                            "message": f"Document must not exceed {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
                        },
                    )
                await destination.write(chunk)
        os.replace(temporary, target)
        return written
    finally:
        await upload.close()
        temporary.unlink(missing_ok=True)


def _index_uploaded_document(service: Any, target: Path, allow_external_data_transfer: bool) -> None:
    try:
        result = service.ingest(
            [target],
            allow_external_data_transfer=allow_external_data_transfer,
        )
        logger.info("document_indexed", extra={"file_name": target.name, "result": result.summary})
    except Exception:
        logger.exception("document_indexing_failed", extra={"file_name": target.name})


async def _authenticated_upload_form(
    request: Request,
    _admin: None = Depends(require_admin),
) -> AsyncIterator[FormData]:
    # FastAPI parses File/Form parameters before dependencies. This dependency
    # authenticates first and closes temporary files after request processing.
    async with request.form(max_files=1, max_fields=1) as form:
        yield form


async def _run_indexing_task(
    budget: IngestionQueueBudget,
    service: Any,
    target: Path,
    allow_external_data_transfer: bool,
) -> None:
    try:
        await run_in_threadpool(_index_uploaded_document, service, target, allow_external_data_transfer)
    finally:
        await budget.release()


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document and schedule incremental RAG indexing",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "category": {"type": "string", "default": "uploads"},
                        },
                        "required": ["file"],
                    },
                },
            },
        },
    },
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    form: FormData = Depends(_authenticated_upload_form),
) -> dict[str, Any]:
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=422, detail={"code": "file_required", "message": "One file is required"})
    category = form.get("category", "uploads")
    if not isinstance(category, str):
        raise HTTPException(status_code=422, detail={"code": "invalid_category", "message": "Category must be text"})
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "ingestion_unavailable", "message": "Document ingestion is not configured"},
        )

    settings = request.app.state.settings
    allow_external = bool(getattr(settings, "rag_allow_external_data_transfer", False))
    if not allow_external:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "external_data_transfer_not_approved",
                "message": "Set RAG_ALLOW_EXTERNAL_DATA_TRANSFER=true after document-owner approval",
            },
        )

    filename = _safe_name(file.filename)
    safe_category = _safe_category(category)
    capacity = int(getattr(settings, "ingestion_queue_capacity", 4))
    budget = _ingestion_budget(request, capacity)
    if not await budget.try_acquire():
        await file.close()
        raise HTTPException(
            status_code=429,
            detail={"code": "ingestion_queue_full", "message": "Document ingestion queue is full"},
        )

    data_dir = Path(settings.rag_data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[2] / data_dir
    target = data_dir / safe_category / filename
    try:
        size = await _persist_upload(file, target)
    except Exception:
        await budget.release()
        raise

    background_tasks.add_task(_run_indexing_task, budget, service, target, allow_external)
    return {
        "status": "accepted",
        "file_name": filename,
        "category": safe_category,
        "size_bytes": size,
        "indexing": "scheduled",
    }
