from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int = Field(ge=0)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int | None = None


class Source(BaseModel):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int = Field(ge=0)
