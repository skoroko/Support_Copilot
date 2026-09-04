from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from fastapi import UploadFile


MAX_DOCUMENT_CHARS = 30_000
MAX_PDF_PAGES = 50
SCAN_PDF_MIN_PAGES = 5
SCAN_PDF_MAX_TEXT_CHARS = 100
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class MediaProcessingError(ValueError):
    """Base error for media that cannot be converted into an LLM content part."""


class UnsupportedMediaTypeError(MediaProcessingError):
    pass


class MediaExtractionError(MediaProcessingError):
    pass


async def media_to_part(
    media: UploadFile,
    *,
    client: Any | None = None,
    transcription_model: str | None = None,
) -> dict[str, Any]:
    """Convert one uploaded file into an OpenAI Chat Completions content part."""

    mime = (media.content_type or "").lower().strip()
    data = await media.read()
    if not data:
        raise MediaProcessingError("Загруженный файл пуст.")

    if mime.startswith("image/"):
        encoded = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    if mime.startswith("audio/") or mime == "application/ogg":
        if client is None:
            raise MediaProcessingError("OpenAI client is required for audio transcription.")
        if not transcription_model:
            raise MediaProcessingError("TRANSCRIPTION_MODEL is required for audio transcription.")
        transcript = await whisper_transcribe(
            data,
            media.filename or "audio.ogg",
            client=client,
            model=transcription_model,
        )
        return {
            "type": "text",
            "text": f"[пользователь сказал голосом]:\n{transcript}",
        }

    if mime == "application/pdf":
        extracted = extract_pdf_text(data)
        return {"type": "text", "text": f"[документ PDF]:\n{extracted[:MAX_DOCUMENT_CHARS]}"}

    if mime == DOCX_MIME or mime.endswith("wordprocessingml.document"):
        extracted = extract_docx_text(data)
        return {"type": "text", "text": f"[документ DOCX]:\n{extracted[:MAX_DOCUMENT_CHARS]}"}

    raise UnsupportedMediaTypeError(f"Unsupported media type: {mime or 'unknown'}")


async def whisper_transcribe(
    audio_bytes: bytes,
    filename: str,
    *,
    client: Any,
    model: str,
) -> str:
    """Transcribe a supported audio file without local conversion or FFmpeg."""

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename
    result = await client.audio.transcriptions.create(model=model, file=audio_file)
    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        raise MediaProcessingError("Не удалось распознать речь в аудиофайле.")
    return text


def extract_pdf_text(data: bytes, *, max_pages: int = MAX_PDF_PAGES) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is installed from requirements.txt
        raise MediaExtractionError("Для обработки PDF установите пакет pypdf.") from exc

    try:
        reader = PdfReader(BytesIO(data))
        selected_pages = reader.pages[:max_pages]
        page_texts = [(page.extract_text() or "").strip() for page in selected_pages]
    except Exception as exc:  # noqa: BLE001 - parser errors must become a user-facing validation error
        raise MediaExtractionError("Не удалось прочитать PDF-документ.") from exc

    extracted = "\n\n".join(text for text in page_texts if text).strip()
    if len(reader.pages) >= SCAN_PDF_MIN_PAGES and len(extracted) < SCAN_PDF_MAX_TEXT_CHARS:
        raise MediaExtractionError("PDF похож на скан: для него требуется OCR.")
    if not extracted:
        raise MediaExtractionError("В PDF не найден извлекаемый текст.")
    return extracted


def extract_docx_text(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency is installed from requirements.txt
        raise MediaExtractionError("Для обработки DOCX установите пакет python-docx.") from exc

    try:
        document = Document(BytesIO(data))
        blocks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append("\t".join(cells))
    except Exception as exc:  # noqa: BLE001 - parser errors must become a user-facing validation error
        raise MediaExtractionError("Не удалось прочитать DOCX-документ.") from exc

    extracted = "\n".join(blocks).strip()
    if not extracted:
        raise MediaExtractionError("В DOCX не найден извлекаемый текст.")
    return extracted
