from __future__ import annotations

from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


class RAGSource(BaseModel):
    text: str
    source: str
    score: float


class RAGResponse(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource] = Field(min_length=3)
