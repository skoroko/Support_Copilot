from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RAGHistoryMessage(BaseModel):
    """Caller-supplied dialogue context; persistence remains outside RAG."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=8_000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class RAGQuery(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    history: list[RAGHistoryMessage] | None = Field(default=None, max_length=20)
    standalone_question: str | None = Field(default=None, min_length=3, max_length=4000)


class RAGSource(BaseModel):
    id: str
    chunk_id: str | None = None
    document_id: str | None = None
    file_name: str
    # ``page`` is the 1-based physical page in the PDF.  ``printed_page`` is
    # the optional page number printed inside the manual and can differ because
    # of covers/front matter.
    page: int | None = None
    printed_page: int | None = None
    score: float
    snippet: str
    confident: bool


class RAGResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource] = Field(default_factory=list)


class RAGRetrievalResponse(BaseModel):
    status: Literal["ok", "no_evidence"]
    question: str
    sources: list[RAGSource] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    telemetry: dict = Field(default_factory=dict)
