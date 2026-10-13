from __future__ import annotations

from fastapi import FastAPI

from app.chat.routes import router as chat_router
from app.core.upload_boundary import CHAT_MEDIA_MAX_FILES
from app.routers.documents import router as documents_router


def _multipart_schema(path: str) -> tuple[dict, dict]:
    api = FastAPI()
    api.include_router(documents_router)
    api.include_router(chat_router)
    body = api.openapi()["paths"][path]["post"]["requestBody"]
    return body, body["content"]["multipart/form-data"]["schema"]


def test_document_upload_openapi_describes_required_file_and_optional_category() -> None:
    body, schema = _multipart_schema("/documents/upload")

    assert body["required"] is True
    assert schema["type"] == "object"
    assert schema["required"] == ["file"]
    assert schema["properties"]["file"] == {"type": "string", "format": "binary"}
    assert schema["properties"]["category"] == {"type": "string", "default": "uploads"}


def test_chat_message_openapi_describes_required_content_and_optional_media() -> None:
    body, schema = _multipart_schema("/chats/{chat_id}/messages")

    assert body["required"] is True
    assert schema["type"] == "object"
    assert schema["required"] == ["content"]
    assert schema["properties"]["content"] == {"type": "string", "minLength": 1}
    assert schema["properties"]["media"] == {
        "type": "array",
        "items": {"type": "string", "format": "binary"},
        "maxItems": CHAT_MEDIA_MAX_FILES,
    }
