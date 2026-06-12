"""Core data types shared across the Ragnite pipeline."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Document(BaseModel):
    """A source document before chunking."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    doc_id: str
    text: str
    index: int = 0
    source: str | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for embedding and keyword indexing (contextual prefix + body)."""
        if self.context:
            return f"{self.context.strip()}\n\n{self.text}"
        return self.text


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    origin: Literal["dense", "bm25", "hybrid", "rerank"] = "dense"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    source: str | None = None
    snippet: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[ScoredChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    cached: bool = False  # True when served from the AnswerCache (no LLM call)


class StreamEvent(BaseModel):
    """Event emitted by streaming answer generation."""

    type: Literal["delta", "answer"]
    text: str = ""
    answer: Answer | None = None


class IngestStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    embedded: bool = False
    contextualized: bool = False
