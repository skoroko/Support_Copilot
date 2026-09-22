from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.observability.pii import redact_pii


MAX_CHAT_MESSAGE_LENGTH = 8000


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value

    def __repr_args__(self):
        return [("role", self.role), ("content", redact_pii(self.content))]


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Как проверить срок действия лицензии?"}],
                    "temperature": 0.2,
                    "max_tokens": 500,
                },
                {
                    "messages": [{"role": "user", "content": "Сформулируй короткий ответ клиенту."}],
                    "model": "gpt-5.4-mini",
                    "temperature": 0,
                    "max_tokens": 300,
                },
            ]
        }
    )

    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    model: str | None = Field(default=None, min_length=1, max_length=100)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, raw_usage: Any) -> "Usage":
        if raw_usage is None:
            return cls()
        return cls(
            prompt_tokens=int(_read_field(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(_read_field(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(_read_field(raw_usage, "total_tokens", 0) or 0),
        )


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Usage
    finish_reason: str | None = None
    cached: bool = False

    @classmethod
    def from_openai(cls, response: Any, *, requested_model: str, cached: bool = False) -> "ChatResponse":
        choices = _read_field(response, "choices", []) or []
        first_choice = choices[0] if choices else None
        message = _read_field(first_choice, "message") if first_choice is not None else None
        content = _read_field(message, "content", "") if message is not None else ""
        finish_reason = _read_field(first_choice, "finish_reason") if first_choice is not None else None
        return cls(
            content=str(content or ""),
            model=str(_read_field(response, "model", requested_model) or requested_model),
            usage=Usage.from_openai(_read_field(response, "usage")),
            finish_reason=finish_reason,
            cached=cached,
        )


class ChatDelta(BaseModel):
    content: str | None = None
    usage: Usage | None = None
