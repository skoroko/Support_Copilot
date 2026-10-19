from __future__ import annotations

from typing import Any, Literal

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
    product: str = "unknown"
    version: str = "unknown"
    language: str = "unknown"
    chm_member: str | None = None
    source_locator: str | None = None
    section: str | list[str] = "unknown"
    section_path: list[str] = Field(default_factory=list)
    source_id: str | None = None
    source_hash: str | None = None
    parent_source_id: str | None = None
    canonical_origin: str | None = None
    derived_from: str | dict[str, Any] | None = None
    location_id: str | None = None
    extraction_version: str | None = None
    extraction_status: str | None = None
    extraction_warnings: list[str] = Field(default_factory=list)
    figure_regions: list[dict[str, Any]] = Field(default_factory=list)
    title: str | None = None
    anchor: str | None = None
    heading: str | None = None
    html_anchor: str | None = None
    relations: list[dict[str, Any]] = Field(default_factory=list)
    block_id: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    block_type: str | None = None
    block_bbox: list[float] | None = None
    block_attributes: dict[str, Any] = Field(default_factory=dict)
    block_relations: list[dict[str, Any]] = Field(default_factory=list)
    atomic_evidence: bool | None = None
    origin: str | None = None
    relation: str | None = None
    related_to: list[str] = Field(default_factory=list)
    text_sha256: str | None = None


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
